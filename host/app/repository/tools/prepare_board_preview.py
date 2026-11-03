"""Freeze a preview candidate offline, without connecting to any service.

This preserves the current policy; it grants neither deployment nor execution.
The serving tool uses the same snapshot builder after its live idle checks.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

from stage_board_control import ROOT, PROJECT, dependency_files, sha
from export_board_control import Exporter


def freeze_preview_inputs(output, pointer, policy_source=None, trial_scope_source=None):
    output = Path(output).resolve()
    pointer = Path(pointer).resolve()
    if not output.is_relative_to(ROOT/'local-only'):
        raise ValueError('local_only_output_required')
    bundle = Exporter(output/'bundle').export(pointer)
    files = {name: output/'bundle'/name for name in bundle['output_sha256']}
    for name, source in dependency_files('src.board_control_service').items():
        target = 'pynq_km1_vision/'+name
        if target in files and sha(files[target]) != sha(source):
            raise ValueError('preview_loaded_and_bound_code_mismatch')
        files[target] = source
    timing_source = PROJECT/'src/sorting_prompted_motion.py'
    timing_sha = sha(timing_source)
    motion_source = PROJECT/'config/prompted_motion_policy.json'
    policy_source=Path(policy_source or PROJECT/'config/yoloe_taco_recyclables.json').resolve()
    trial_source=Path(trial_scope_source or ROOT/'local-only/yoloe-recyclables/calibration-20261030/deployment.json').resolve()
    if any(not source.is_relative_to(ROOT) or source.suffix!='.json'
           for source in (policy_source,trial_source)):
        raise ValueError('preview_policy_and_scope_must_be_repository_json')
    files.update({
        'prompted-policy.json': policy_source,
        'pynq_km1_vision/config/prompted_motion_policy.json': motion_source,
        'prompted-trial-scope.json': trial_source,
        'pynq_km1_vision/src/board_serial_inventory.py': PROJECT/'src/board_serial_inventory.py',
        'host-sources/sorting_prompted_motion.py': timing_source,
    })
    # Freeze the transition adapter as well as the PS worker for later review.
    for name in ('prepare_board_preview.py', 'serve_board_prompted.py',
                 'board_control_gateway.py', 'activate_board_console.py'):
        files['host-sources/tools/'+name] = ROOT/'tools'/name
    for name in ('board_prompted_console.py', 'board_control_gateway_client.py',
                 'sorting_prompted_console.py', 'monitor_prompted_operation.py'):
        files['host-sources/src/'+name] = PROJECT/'src'/name
    originals = dict(files)
    digests = {name: sha(path) for name, path in originals.items()}
    snapshot = output/'package'
    snapshot.mkdir()
    for name, path in originals.items():
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digests[name]:
            raise ValueError('board_preview_input_changed_during_snapshot')
        target = snapshot/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        files[name] = target
    motion_options = json.loads(files['pynq_km1_vision/config/prompted_motion_policy.json'].read_text(encoding='utf-8'))
    timing = dict(max_age_ms=motion_options['frame_max_age_ms'],
                  frame_timeout_s=motion_options['frame_timeout_s'],
                  source_sha256=timing_sha, source_module=timing_source.name)
    timing_file = snapshot/'observation-timing.json'
    timing_file.write_text(json.dumps(timing, indent=2), encoding='utf-8')
    files['observation-timing.json'] = timing_file
    source_hashes = dict(bundle['original_sha256'])
    source_hashes[pointer.relative_to(ROOT).as_posix()] = bundle['source_pointer_sha256']
    source_hashes.update({path.relative_to(ROOT).as_posix(): digests[name]
                         for name, path in originals.items()})
    manifest = dict(kind='board-preview-offline-package-v1',
        status='offline_frozen_not_deployed', execution_enabled=False,
        physical_motion=False, live_settings_verified=False, cutover_ready=False,
        binding=bundle['binding'], original_sha256=source_hashes,
        output_sha256={name: sha(path) for name, path in files.items()},
        observation_timing=timing, motion_options=motion_options,
        caveat='Recheck live settings, original review records, ownership and resources before deployment.')
    (output/'offline-package.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    verify_preview_inputs(output)
    return dict(bundle=bundle, files=files, originals=originals, digests=digests,
                timing=timing, timing_sha=timing_sha, motion_options=motion_options,
                motion_options_sha=digests['pynq_km1_vision/config/prompted_motion_policy.json'],
                trial_scope_sha=digests['prompted-trial-scope.json'], manifest=manifest)


def verify_preview_inputs(output):
    """Reject changed snapshots and stale sources; never access the board."""
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT/'local-only'):
        raise ValueError('local_only_output_required')
    manifest = json.loads((output/'offline-package.json').read_text(encoding='utf-8'))
    if (manifest.get('kind') != 'board-preview-offline-package-v1' or
            any(manifest.get(key) is not False for key in
                ('execution_enabled', 'physical_motion', 'live_settings_verified', 'cutover_ready'))):
        raise ValueError('invalid_offline_package_contract')
    for base, table, error in ((output/'package', manifest['output_sha256'], 'offline_package_changed'),
                               (ROOT, manifest['original_sha256'], 'offline_source_changed')):
        for name, digest in table.items():
            parts = PurePosixPath(name.replace('\\', '/'))
            path = (base/name).resolve()
            if parts.is_absolute() or '..' in parts.parts or not path.is_relative_to(base.resolve()):
                raise ValueError('offline_package_path_escape')
            if not path.is_file() or sha(path) != digest:
                raise ValueError(error+':'+name)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--policy-source',type=Path)
    parser.add_argument('--trial-scope-source',type=Path)
    args = parser.parse_args()
    if args.verify:
        manifest = verify_preview_inputs(args.output)
    else:
        if not args.output.resolve().is_relative_to(ROOT/'local-only'):
            raise ValueError('local_only_output_required')
        args.output.mkdir(parents=True, exist_ok=False)
        manifest = freeze_preview_inputs(args.output, ROOT/'local-only/station-grasp-integration/active.json',
                                        args.policy_source,args.trial_scope_source)['manifest']
    print(json.dumps(dict(status=manifest['status'], files=len(manifest['output_sha256']),
                         stable_frames=manifest['motion_options']['stable_frames'],
                         live_settings_verified=False, cutover_ready=False, execution_enabled=False)))


if __name__ == '__main__':
    main()
