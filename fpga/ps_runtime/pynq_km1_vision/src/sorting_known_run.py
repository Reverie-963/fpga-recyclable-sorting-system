"""Single-cycle CLI for three qualified known specimens; no unattended batch mode."""
import argparse
import json
import os
from pathlib import Path
import time
import uuid

import cv2

from .sorting_io import CameraYoloe, HTTPYoloe, PLStatistics
from .sorting_known import (GeometryGuardedMotion, KnownController, SCOPE, bound_file,
                            file_hash, known_readiness, validate_known_config, profile_fingerprint)
from .sorting_loop import Journal, need, readiness
from .static_visual_verification import roi_mask
from .sorting_scene import check_motion_layout, verify_setup_reanchor
from .sorting_motion import PROFILES
from .sorting_station_io import StationCamera, StationStatistics


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def preflight(c, base):
    blockers = known_readiness(c, base)
    return dict(status='ready_for_one_supervised_known_item' if not blockers else 'not_ready_for_motion',
                operation_scope=SCOPE, execution_blockers=blockers, single_cycle_only=True,
                production_execution_blockers=readiness(c), physical_motion=False,
                production_quality_claimed=False, unattended_accepted=False,
                operator_ready_acknowledgement_required=True)


def save_config(path, c):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(c, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def capture_empty_reference(c, base, controller, journal, config_path, setup_anchor=None):
    """Operator confirms all boxes empty; images and masks independently check it."""
    task = 'empty-' + uuid.uuid4().hex
    rows, evidence = [], []
    # A camera service restart may be rebound only in this explicit, read-only
    # setup operation after the original fixed marks have passed again.
    controller.c['known']['expected_camera_stream_id'] = None
    for i in range(c['stable_frames']):
        row = controller.acquire()
        need(len(controller.source_objects(row)) == 1, 'place_one_known_source_item_for_setup')
        for profile in c['profiles'].values():
            region = roi_mask(profile['bin_roi'], c['image_size'])
            need(not any((mask & region).any() for mask in row['masks']), 'all_three_bins_must_be_empty')
        if rows:
            controller.unchanged(rows[-1], row)
        evidence.append(journal.capture(task, 'empty_' + str(i), row))
        rows.append(row)
    need(all(time.monotonic()*1000-r['observation']['capture_started_ms'] <= c['max_age_ms'] for r in rows),
         'empty_reference_window_expired')
    image = journal.root / evidence[-1]['image']
    c['empty_reference'] = dict(path=os.path.relpath(image, base), sha256=file_hash(image))
    previous_anchor = c['layout_reference']
    if setup_anchor is not None:
        c['layout_reference'] = dict(c['empty_reference'])
    c['known']['expected_camera_stream_id'] = rows[-1]['observation']['source_stream_id']
    journal.save(task + '.json', dict(operation_scope=SCOPE, physical_motion=False,
        operator_confirmed_all_bins_empty=True, operator_confirmed_neutral_open=True,
        source_label=controller.source_objects(rows[-1])[0][0]['label'], evidence=evidence,
        previous_layout_reference=previous_anchor, setup_anchor_validation=setup_anchor,
        camera_stream_id=c['known']['expected_camera_stream_id']))
    save_config(config_path, c)
    report = preflight(c, base)
    report.update(empty_reference_captured=True, config=str(config_path), setup_record=task + '.json')
    return report


def bind_station_vision(c, base, controller, journal, config_path):
    """Explicit read-only rebind after a verified service migration, not a move."""
    need(c['statistics']['backend'] == 'pl_board_local', 'new_station_backend_required')
    task = 'vision-bind-'+uuid.uuid4().hex
    controller.c['known']['expected_camera_stream_id'] = None
    policy = dict(c['layout'], image_size=c['image_size'])
    setup = controller.source.raw(c['frame_timeout_s'])
    try:
        anchor = check_motion_layout(controller.reference, setup['image'], policy)
    except ValueError:
        anchor = verify_setup_reanchor(controller.reference, setup['image'], policy)
    controller.reference = setup['image'].copy()
    frames, rows = [], []
    for i in range(c['stable_frames']):
        row = controller.acquire()
        if rows:
            controller.unchanged(rows[-1], row)
        saved = journal.capture(task, 'frame_'+str(i), row)
        frames.append(dict(path=os.path.relpath(journal.root/saved['image'], base), sha256=saved['png_sha256']))
        rows.append(row)
    record = dict(kind='known-station-vision-binding-v1', physical_motion=False,
        stream_id=rows[-1]['observation']['source_stream_id'],
        config_sha256=controller.stats.signature, assets=c['statistics']['assets'],
        geometry_profile_sha256={label:profile_fingerprint(c,label) for label in c['profiles']},
        previous_camera_stream_id=c['known'].get('expected_camera_stream_id'),
        fixed_marker_validation=anchor, frames=frames,
        physical_acceptance_claimed=False, geometry_calibration_changed=False)
    journal.save(task+'.json', record)
    c['known']['vision_binding'] = dict(path=os.path.relpath(journal.root/(task+'.json'), base),
                                      sha256=file_hash(journal.root/(task+'.json')))
    c['known']['expected_camera_stream_id'] = record['stream_id']
    c['layout_reference'] = frames[-1]
    save_config(config_path, c)
    report = preflight(c, base)
    report.update(status='station_vision_bound_no_motion', binding_record=task+'.json')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--motion-profile', choices=PROFILES,
                        help='compact: supervised candidate, original geometry qualification only')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--preflight', action='store_true')
    action.add_argument('--preview', action='store_true')
    action.add_argument('--execute', action='store_true')
    action.add_argument('--capture-empty-reference', action='store_true')
    action.add_argument('--bind-vision', action='store_true')
    parser.add_argument('--ready', action='store_true', help='Operator setup: neutral open, known item, empty target, hands clear')
    parser.add_argument('--resolve-task')
    parser.add_argument('--observation')
    args = parser.parse_args()
    path = args.config.resolve()
    c, base = read(path), path.parent
    validate_known_config(c)
    args.motion_profile = args.motion_profile or c['known'].get('motion_profile', 'qualified')
    need(not args.resolve_task or not any((args.execute, args.capture_empty_reference, args.preview, args.bind_vision)),
         'review_cannot_start_motion')
    if args.preflight or not any((args.preview, args.execute, args.capture_empty_reference, args.resolve_task, args.bind_vision)):
        report = preflight(c, base)
        report.update(motion_profile=args.motion_profile,
                      supervised_motion_trial=args.motion_profile == 'compact',
                      candidate_physical_speed_verified=False if args.motion_profile == 'compact' else None)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    with Journal((base / c['journal_dir']).resolve()) as journal:
        if args.resolve_task:
            need(args.observation, 'physical_review_note_required')
            journal.resolve(args.resolve_task, args.observation)
            print(json.dumps(dict(status='operator_resolved_not_automatic_success', physical_motion=False)))
            return 0
        need(journal.pending() is None, 'unresolved_previous_task')
        blockers = known_readiness(c, base, require_empty=not args.capture_empty_reference,
                                   require_vision_binding=not args.bind_vision)
        if args.bind_vision:
            need(not blockers, 'station_binding_inputs_not_ready:'+','.join(blockers))
        if args.execute or args.capture_empty_reference:
            need(args.ready and not blockers, 'known_setup_not_ready:' + ','.join(blockers))
        lease = (base / c['known']['device_lease']).resolve()
        need(not lease.exists(), 'arm_in_use_or_stale_device_lease')
        reference = cv2.imread(str(bound_file(base, c['layout_reference'])))
        empty = cv2.imread(str(bound_file(base, c['empty_reference']))) if c.get('empty_reference') else None
        model = HTTPYoloe(c, None)
        need(model.model_ids['supervised'] == c['model_version'] and
             model.execution_id('supervised') == c['runtime_id'], 'reviewed_model_runtime_changed')
        token_file = c['statistics'].get('token_file')
        if token_file and not os.environ.get(c['statistics']['token_env']):
            os.environ[c['statistics']['token_env']] = (base / token_file).resolve().read_text(encoding='utf-8').strip()
        statistics = StationStatistics(c) if c['statistics']['backend'] == 'pl_board_local' else PLStatistics(c['statistics'])
        source = StationCamera(c, model, statistics) if statistics.backend == 'pl_board_local' else CameraYoloe(c, model)
        stops = [(base / item).resolve() for item in c['known']['stop_files']]
        motion = GeometryGuardedMotion(c, journal, reference, lease, stops)
        controller = KnownController(c, source, statistics, motion, journal, reference, empty, base,
            mode='known_physical' if args.execute else 'preview', ready=args.ready,
            motion_profile=args.motion_profile)
        if args.bind_vision:
            report = bind_station_vision(c, base, controller, journal, path)
        elif args.capture_empty_reference:
            setup = source.raw(c['frame_timeout_s'])
            policy = dict(c['layout'], image_size=c['image_size'])
            try:
                anchor = check_motion_layout(reference, setup['image'], policy)
            except ValueError:
                anchor = verify_setup_reanchor(reference, setup['image'], policy)
            journal.capture('setup-' + uuid.uuid4().hex, 'anchor', setup)
            controller.reference = setup['image'].copy()
            report = capture_empty_reference(c, base, controller, journal, path, setup_anchor=anchor)
        else:
            report = controller.cycle()
        print(json.dumps(dict(status=report['status'], operation_scope=SCOPE,
            task_id=report.get('task_id'), label=report.get('choice', {}).get('label'),
            bin_id=report.get('choice', {}).get('bin_id'), steps=len(report.get('steps', [])),
            elapsed_s=report.get('elapsed_s'), reason=report.get('reason'),
            execution_blockers=report.get('execution_blockers', []),
            report_file=str(journal.root / (report['task_id'] + '.json')) if report.get('task_id') else None,
            empty_reference_captured=report.get('empty_reference_captured', False),
            motion_profile=args.motion_profile,
            motion_optimization=report.get('choice', {}).get('motion_optimization'),
            production_quality_claimed=False, unattended_accepted=False), ensure_ascii=False))
        return 0 if report['status'] in ('preview_only', 'empty_scene', 'completed_visual_evidence',
                                        'ready_for_one_supervised_known_item', 'station_vision_bound_no_motion') else 2


if __name__ == '__main__':
    raise SystemExit(main())
