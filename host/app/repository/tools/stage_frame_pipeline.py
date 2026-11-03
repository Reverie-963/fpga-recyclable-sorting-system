"""Copy a paired frame-pipeline candidate to a fresh board directory. No startup."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import secrets
import shlex

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--connection-module', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--vision-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--remote-root', required=True)
    args = parser.parse_args()
    output, build = args.output.resolve(), args.build.resolve()
    if not output.is_relative_to(ROOT/'local-only'):
        raise ValueError('local_only_stage_report_required')
    remote = PurePosixPath(args.remote_root)
    if remote.parent != PurePosixPath('/home/xilinx') or not remote.name.startswith('km1-frame-pipeline-'):
        raise ValueError('isolated_frame_pipeline_directory_required')
    report = json.loads((build/'build-result.json').read_text(encoding='utf-8-sig'))
    if report.get('board_programmed') is not False or report.get('toolchain') != '2026.1':
        raise ValueError('fresh_reviewed_build_required')
    for name in ('frame_pipeline.bit', 'frame_pipeline.hwh'):
        if hashlib.sha256((build/'board'/name).read_bytes()).hexdigest() != report['assets'][name]:
            raise ValueError('paired_build_asset_changed')
    config = json.loads(args.vision_config.read_text(encoding='utf-8-sig'))
    if config.get('synthetic_fixture') is not False or config['backend'] != 'pl_board_local':
        raise ValueError('existing_reviewed_station_config_required')
    config.update(pipeline_contract='station-frame-pipeline-v1', preview_enabled=True,
        bitfile='../board/frame_pipeline.bit',
        assets={'station_vision.bit': report['assets']['frame_pipeline.bit'],
                'station_vision.hwh': report['assets']['frame_pipeline.hwh']},
        purpose='full_frame_candidate_preserving_original_rgb_evidence')
    output.mkdir(parents=True, exist_ok=False)
    (output/'frame_pipeline.local.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    (output/'token.txt').write_text(secrets.token_hex(32), encoding='ascii')
    launcher = """from pathlib import Path
import os,sys
root=Path(__file__).resolve().parent
os.chdir(root);sys.path.insert(0,str(root))
os.environ['KM1_STATION_VISION_TOKEN']=(root/'token.txt').read_text().strip()
sys.argv=['frame_pipeline_server','--config',str(root/'config/frame_pipeline.local.json'),'--port','8784']
from src.frame_pipeline_server import main
main()
"""
    config['token_env'] = 'KM1_STATION_VISION_TOKEN'
    (output/'frame_pipeline.local.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    (output/'launch.py').write_text(launcher, encoding='utf-8')
    files = {str(remote/'launch.py'): output/'launch.py', str(remote/'token.txt'): output/'token.txt',
             str(remote/'config/frame_pipeline.local.json'): output/'frame_pipeline.local.json'}
    for name in ('__init__.py', 'station_vision.py', 'station_vision_server.py', 'frame_pipeline.py',
                 'frame_pipeline_selftest.py', 'frame_pipeline_server.py'):
        files[str(remote/'src'/name)] = ROOT/'pynq_km1_vision/src'/name
    files[str(remote/'board/km1_frame_pipeline.py')] = ROOT/'pynq_km1_vision/board/km1_frame_pipeline.py'
    files[str(remote/'probe.py')] = ROOT/'tools/probe_frame_pipeline.py'
    for name in ('frame_pipeline.bit', 'frame_pipeline.hwh'):
        files[str(remote/'board'/name)] = build/'board'/name
    spec = importlib.util.spec_from_file_location('frame_private_connection', args.connection_module.resolve())
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    client = module.connect()
    try:
        sftp = client.open_sftp()
        try:
            sftp.mkdir(str(remote))
            for sub in ('src', 'board', 'config'):
                sftp.mkdir(str(remote/sub))
            for target, path in files.items():
                sftp.put(str(path), target)
                if target.endswith('token.txt'):
                    sftp.chmod(target, 0o600)
        finally:
            sftp.close()
        hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                  for name, path in files.items() if not name.endswith('token.txt')}
        probe = 'import hashlib,json; files='+repr(hashes)+'; print(json.dumps([p for p,h in files.items() if hashlib.sha256(open(p,"rb").read()).hexdigest()!=h]))'
        _, stdout, _ = client.exec_command('python3 -c '+shlex.quote(probe), timeout=20)
        mismatches = json.loads(stdout.read().decode())
        if stdout.channel.recv_exit_status() != 0 or mismatches:
            raise RuntimeError('staged_candidate_hash_mismatch')
        manifest = dict(status='staged_not_started', remote_root=str(remote),
            assets=config['assets'], files=hashes, remote_hashes_verified=True,
            board_programmed=False, camera_service_changed=False, mechanical_commands_sent=False)
        (output/'stage-result.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(json.dumps(dict(status=manifest['status'], remote_root=str(remote))))
    finally:
        client.close()


if __name__ == '__main__':
    main()
