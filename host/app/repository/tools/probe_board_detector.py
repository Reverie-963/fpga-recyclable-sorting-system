"""Stage a new read-only candidate and prove board->PC detector forwarding."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import shlex
import sys

from stage_board_control import dependency_files, require_idle, sha

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT/'pynq_km1_vision'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--connection-module', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--remote-root', required=True)
    p.add_argument('--station-token-file', required=True)
    p.add_argument('--frames', type=int, choices=(1, 3), default=1)
    args = p.parse_args()
    remote, output = PurePosixPath(args.remote_root), args.output.resolve()
    if remote.parent != PurePosixPath('/home/xilinx') or not remote.name.startswith('km1-detector-proof-'):
        raise ValueError('new_isolated_detector_candidate_required')
    if not output.is_relative_to(ROOT/'local-only'):
        raise ValueError('local_only_output_required')
    station_token = PurePosixPath(args.station_token_file)
    if (station_token.parent.parent != PurePosixPath('/home/xilinx') or
            not station_token.parent.name.startswith('km1-station-vision-') or station_token.name != 'token.txt'):
        raise ValueError('existing_station_token_path_required')
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    sys.path.insert(0, str(PROJECT))
    from src.board_detector_tunnel import DetectorTunnel
    from src.sorting_known import bound_file
    from src.sorting_relocated_run import current_source_config
    pointer = ROOT/'local-only/station-grasp-integration/active.json'
    pointer_sha = sha(pointer)
    binding = bound_file(pointer.parent, json.loads(pointer.read_text(encoding='utf-8-sig')))
    record = json.loads(binding.read_text(encoding='utf-8-sig'))
    config_path = bound_file(binding.parent, record['files']['config'])
    config = current_source_config(config_path)
    source = dict(pointer_sha256=pointer_sha, binding_sha256=sha(binding), config_sha256=sha(config_path))
    fixture = dict(config=config, source=source)
    require_idle(fixture)
    output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('private_connection', args.connection_module.resolve())
    connection = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(connection)
    client = connection.connect()
    manifest = dict(kind='board-detector-proof-deployment-v1', status='prepared', remote_root=str(remote),
                    source=source, execution_enabled=False, physical_motion=False)

    def command(code, timeout=15):
        _, out, err = client.exec_command(code, timeout=timeout)
        stdout, stderr = out.read().decode(), err.read().decode()
        return out.channel.recv_exit_status(), stdout, stderr

    try:
        resource = ('import os,json; m={k:int(v.split()[0]) for k,v in '
                    '(line.split(":",1) for line in open("/proc/meminfo"))}; '
                    'print(json.dumps(dict(available_kb=m["MemAvailable"],load=os.getloadavg()[0]))); '
                    'assert m["MemAvailable"]>=200000 and os.getloadavg()[0]<2.5,"board_busy_defer"')
        code, raw, err = command('python3 -c '+shlex.quote(resource))
        if code:
            raise RuntimeError('board_resources_unavailable_defer')
        manifest['board_resources_before'] = json.loads(raw)
        files = dependency_files('src.board_detector_probe')
        hashes = {name:sha(path) for name,path in files.items()}
        manifest['python_sha256'] = hashes
        # A dedicated SSH connection owns this forward and is always closed.
        with DetectorTunnel(connection.connect, secrets.token_hex(32), config['detector_url']) as tunnel:
            migrated = copy.deepcopy(config)
            migrated['detector_url'] = tunnel.board_url
            migrated['migration_source'] = source
            config_file, token_file = output/'probe-config.json', output/'relay-token.txt'
            config_file.write_text(json.dumps(migrated), encoding='utf-8')
            token_file.write_text(tunnel.token, encoding='ascii')
            sftp = client.open_sftp()
            try:
                sftp.mkdir(str(remote), mode=0o700)
                for parent in sorted({str(PurePosixPath(n).parent) for n in files} - {'.'}):
                    sftp.mkdir(str(remote/parent))
                transfers = dict(files, **{'probe-config.json':config_file, 'relay-token.txt':token_file})
                for name,path in transfers.items():
                    sftp.put(str(path), str(remote/name))
                sftp.chmod(str(remote/'relay-token.txt'), 0o600)
                for name,path in transfers.items():
                    with sftp.open(str(remote/name), 'rb') as stream:
                        if hashlib.sha256(stream.read()).hexdigest() != sha(path):
                            raise ValueError('detector_candidate_hash_mismatch:'+name)
            finally:
                sftp.close()
            listener = ('import subprocess,json; rows=subprocess.check_output(["ss","-lntH"],text=True).splitlines(); '
                        'print(json.dumps([r.split()[3] for r in rows if r.split()[3].endswith('+
                        repr(':'+str(tunnel.port))+')]))')
            code, raw, err = command('python3 -c '+shlex.quote(listener))
            require_idle(fixture)
            if code or json.loads(raw) != ['127.0.0.1:'+str(tunnel.port)]:
                raise ValueError('detector_reverse_listener_not_exclusively_loopback')
            manifest['forwarded_port'] = tunnel.port
            probe = ('cd '+shlex.quote(str(remote))+' && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '
                     'timeout 60s nice -n 19 /usr/local/share/pynq-venv/bin/python3 -m src.board_detector_probe '
                     '--config probe-config.json --relay-token-file relay-token.txt --station-token-file '+
                     shlex.quote(str(station_token))+' --output proof --frames '+str(args.frames))
            code, raw, err = command(probe, timeout=65)
            manifest.update(worker_exit_code=code, worker_stdout=raw[-2000:], worker_stderr=err[-2000:])
            sftp = client.open_sftp()
            try:
                sftp.get(str(remote/'proof/result.json'), str(output/'result.json'))
                result = json.loads((output/'result.json').read_text())
                for i, row in enumerate(result['frames']):
                    for name in (f'frame-{i}.png', f'observation-{i}.json'):
                        sftp.get(str(remote/'proof'/name), str(output/name))
            finally:
                sftp.close()
            manifest['status'] = 'readonly_inference_verified_not_active' if code == 0 and result['status'] == 'passed' else 'readonly_check_failed_not_active'
        manifest['tunnel_closed'] = tunnel.state == 'closed'
        manifest['source_files_unchanged'] = all(sha(path) == hashes[name] for name,path in files.items())
        manifest['active_pointer_unchanged'] = sha(pointer) == pointer_sha
    except Exception as exc:
        manifest.update(status='failed_not_active', error_type=type(exc).__name__, reason=str(exc))
        raise
    finally:
        client.close()
        (output/'deployment.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(dict(status=manifest['status'], output=str(output), tunnel_closed=manifest['tunnel_closed'])))
    return 0 if manifest['status'] == 'readonly_inference_verified_not_active' else 1


if __name__ == '__main__':
    raise SystemExit(main())
