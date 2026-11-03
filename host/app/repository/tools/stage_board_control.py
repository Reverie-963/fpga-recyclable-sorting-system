"""Snapshot and stage the PS control migration in a fresh isolated directory.

Transfers only the worker's Python dependency closure and frozen route inputs.
Does not replace a service, bind production evidence, program PL or move hardware.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / 'pynq_km1_vision'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dependency_files(entry='src.board_control_shadow'):
    pending, found = [entry], {}
    while pending:
        module = pending.pop()
        if module in found:
            continue
        path = PROJECT.joinpath(*module.split('.')).with_suffix('.py')
        if not path.is_file():
            continue
        found[module] = path
        package = module.rsplit('.', 1)[0].split('.')
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = '.'.join(package[:len(package)-node.level+1]) if node.level else ''
                name = '.'.join(v for v in (base, node.module or '') if v)
                names = [name] + [name+'.'+a.name for a in node.names]
            else:
                continue
            pending.extend(n for n in names if n.startswith(('src.', 'openmv_gripper.')))
    # Keep package initialization identical to the repository.
    paths = set(found.values())
    paths.add(PROJECT/'src/__init__.py')
    return {p.relative_to(PROJECT).as_posix(): p for p in sorted(paths)}


def make_fixture():
    sys.path.insert(0, str(PROJECT))
    from src.sorting_known import bound_file
    from src.sorting_relocated_run import current_source_config
    from src.sorting_unified_plan import validate_policy
    from src.board_control_shadow import plan_case
    pointer = ROOT/'local-only/station-grasp-integration/active.json'
    active = json.loads(pointer.read_text(encoding='utf-8-sig'))
    binding = bound_file(pointer.parent, active)
    b = json.loads(binding.read_text(encoding='utf-8-sig'))
    config_path = bound_file(binding.parent, b['files']['config'])
    policy_path = bound_file(binding.parent, b['files']['unified_policy'])
    config = current_source_config(config_path)
    policy = json.loads(policy_path.read_text(encoding='utf-8-sig'))
    validate_policy(policy, config)
    cases = []
    for label in ('paper', 'metal', 'plastic'):
        for angle in range(0, 360, 90):
            for xy in ([0, 5], [-30, 20], [30, 20]):
                case = dict(id=f'{label}-{angle}-{xy[0]}-{xy[1]}', label=label,
                            axis_deg=angle, card_xy=xy, slot_index=0)
                expected, _ = plan_case(config, policy, case)
                cases.append(dict(case, expected_sha256=expected))
    # These numeric snapshots retain flags verbatim but are never a live binding.
    return dict(kind='board-control-shadow-fixture-v1', config=config, policy=policy, cases=cases,
                source=dict(binding_sha256=sha(binding), pointer_sha256=sha(pointer),
                            config_sha256=sha(config_path), policy_sha256=sha(policy_path)),
                grants_execution=False)


def require_idle(fixture):
    """Defer bounded board work while the shared production controller owns it."""
    from src.sorting_known import bound_file
    pointer = ROOT/'local-only/station-grasp-integration/active.json'
    if sha(pointer) != fixture['source']['pointer_sha256']:
        raise ValueError('active_binding_changed_defer')
    binding = bound_file(pointer.parent, json.loads(pointer.read_text(encoding='utf-8-sig')))
    b = json.loads(binding.read_text(encoding='utf-8-sig'))
    base = bound_file(binding.parent, b['files']['config']).parent
    c = fixture['config']
    ledger = (base/c['journal_dir']).resolve()
    protected = [base/c['known']['device_lease'], ledger/'controller.lock',
                 ledger/'pending.json', ledger/'batch-pending.json']
    if any(p.exists() for p in protected):
        raise ValueError('station_busy_or_unresolved_defer')


def execute(client, command, timeout=330):
    _, out, err = client.exec_command(command, timeout=timeout)
    stdout, stderr = out.read().decode(), err.read().decode()
    code = out.channel.recv_exit_status()
    if code:
        raise RuntimeError(f'board_command_failed:{code}:{stderr[-1500:]}:{stdout[-1000:]}')
    return stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--connection-module', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--remote-root', required=True)
    parser.add_argument('--check', action='store_true', help='Run isolated no-device checks at nice 19')
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT/'local-only'):
        raise ValueError('local_only_output_required')
    remote = PurePosixPath(args.remote_root)
    if remote.parent != PurePosixPath('/home/xilinx') or not remote.name.startswith('km1-control-shadow-'):
        raise ValueError('isolated_remote_directory_required')
    output.mkdir(parents=True, exist_ok=False)
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    fixture = make_fixture()
    fixture_path = output/'fixture.json'
    fixture_path.write_text(json.dumps(fixture, indent=2, allow_nan=False), encoding='utf-8')
    files = dependency_files()
    files['fixture.json'] = fixture_path
    source_hashes = {name: sha(path) for name, path in files.items()}
    manifest = dict(kind='board-control-shadow-deployment-v1', remote_root=str(remote),
                    status='prepared', files=source_hashes, source=fixture['source'],
                    execution_enabled=False, physical_motion=False, live_services_changed=False)
    spec = importlib.util.spec_from_file_location('private_connection', args.connection_module.resolve())
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    client = module.connect()
    try:
        sftp = client.open_sftp()
        try:
            sftp.mkdir(str(remote), mode=0o700)
            for parent in sorted({str(PurePosixPath(n).parent) for n in files} - {'.'}):
                sftp.mkdir(str(remote/parent))
            for name, path in files.items():
                sftp.put(str(path), str(remote/name))
            # Verify streamed bytes, without starting any uploaded code.
            for name, digest in source_hashes.items():
                with sftp.open(str(remote/name), 'rb') as stream:
                    if hashlib.sha256(stream.read()).hexdigest() != digest:
                        raise ValueError('remote_hash_mismatch:'+name)
            manifest.update(status='staged_not_active', remote_hashes_verified=True)
        finally:
            sftp.close()
        if args.check:
            require_idle(fixture)
            # Reserve headroom for the camera/PL service. No package install,
            # service shutdown, serial enumeration/open or overlay work.
            resource_probe = ('import os; m={k:int(v.split()[0]) for k,v in '
                              '(line.split(":",1) for line in open("/proc/meminfo"))}; '
                              'assert m["MemAvailable"]>=200000 and os.getloadavg()[0]<2.5, '
                              '"board_busy_defer"')
            execute(client, 'python3 -c '+shlex.quote(resource_probe), timeout=15)
            command = ('cd '+shlex.quote(str(remote))+' && '
                       'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '
                       'timeout 300s nice -n 19 /usr/local/share/pynq-venv/bin/python3 '
                       '-m src.board_control_shadow --fixture fixture.json --output check-result.json')
            manifest['check_summary'] = json.loads(execute(client, command))
            sftp = client.open_sftp()
            try:
                sftp.get(str(remote/'check-result.json'), str(output/'check-result.json'))
            finally:
                sftp.close()
            result = json.loads((output/'check-result.json').read_text(encoding='utf-8'))
            if result['status'] != 'passed' or result['location'] != 'pynq_ps':
                raise ValueError('board_shadow_checks_not_passed')
            manifest['status'] = 'board_shadow_verified_not_active'
        # Detect source edits made by other ongoing tasks while staging.
        manifest['source_files_unchanged'] = all(sha(path) == source_hashes[name] for name, path in files.items())
        manifest['active_pointer_unchanged'] = sha(ROOT/'local-only/station-grasp-integration/active.json') == fixture['source']['pointer_sha256']
        if not manifest['source_files_unchanged'] or not manifest['active_pointer_unchanged']:
            manifest['status'] = 'stale_snapshot_not_active'
    except Exception as exc:
        manifest.update(status='failed_not_active', error_type=type(exc).__name__)
        raise
    finally:
        client.close()
        (output/'deployment.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(dict(status=manifest['status'], files=len(files), remote_root=str(remote),
                         source_files_unchanged=manifest['source_files_unchanged'])))


if __name__ == '__main__':
    main()
