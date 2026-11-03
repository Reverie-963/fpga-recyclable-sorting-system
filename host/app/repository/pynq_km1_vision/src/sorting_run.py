"""One entry point for preflight, preview, bounded automatic sorting and review."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2

from .sorting_loop import Controller, Journal, acceptance_fingerprint, need, readiness, validate_config
from .sorting_io import CameraYoloe, HTTPYoloe, Motion, PLStatistics, Statistics
from .static_visual_verification import layout_evidence


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def preflight(c, base):
    validate_config(c)
    blockers = readiness(c)
    reference = c.get('layout_reference', {})
    path = (base / reference.get('path', 'MISSING')).resolve()
    if not path.is_file():
        blockers.append('layout_reference.path')
    elif hashlib.sha256(path.read_bytes()).hexdigest() != reference.get('sha256'):
        blockers.append('layout_reference.sha256')
    else:
        try:
            image = cv2.imread(str(path))
            need(image is not None and list(image.shape[:2][::-1]) == c['image_size'], 'wrong_image_size')
            layout_evidence(image, image, dict(c['layout'], image_size=c['image_size']))
        except (ValueError, cv2.error) as exc:
            blockers.append('layout_reference.validation:'+str(exc))
    for label, profile in c['profiles'].items():
        record = profile.get('acceptance_record')
        if record is None:
            continue
        try:
            raw = (base / record['path']).resolve().read_bytes()
            need(hashlib.sha256(raw).hexdigest() == record['sha256'], 'hash_changed')
            accepted = json.loads(raw)
            need(accepted['kind'] == 'sorting-profile-acceptance-v1' and
                 accepted['label'] == label and accepted['station_id'] == c['station_id'] and
                 accepted['automatic_trial_approved'] is True and
                 accepted['configuration_sha256'] == acceptance_fingerprint(c), 'configuration_not_accepted')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            blockers.append(label+'.acceptance_record:'+str(exc))
    return dict(status='ready_for_authorized_trial' if not blockers else 'not_ready_for_motion',
                execution_blockers=blockers, physical_motion=False,
                acceptance_configuration_sha256=acceptance_fingerprint(c),
                automatic_success_claimed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--max-tasks', type=int, default=1)
    parser.add_argument('--max-seconds', type=float, default=300)
    parser.add_argument('--resolve-task')
    parser.add_argument('--observation')
    args = parser.parse_args()
    need(not (args.execute and (args.preflight or args.resolve_task)), 'incompatible_run_modes')
    need(1 <= args.max_tasks <= 100 and 0 < args.max_seconds <= 3600, 'invalid_batch_limits')
    config_path = args.config.resolve()
    c, base = load(config_path), config_path.parent
    result = preflight(c, base)
    if args.preflight:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    with Journal((base / c['journal_dir']).resolve()) as journal:
        if args.resolve_task:
            need(not args.preflight and args.observation, 'review_requires_operator_observation')
            journal.resolve(args.resolve_task, args.observation)
            print(json.dumps(dict(status='operator_resolved', automatic_success_claimed=False)))
            return 0
        need(journal.pending() is None, 'unresolved_previous_task')
        if args.execute:
            need(not result['execution_blockers'], 'execution_blocked:' + ','.join(result['execution_blockers']))
        reference_path = (base / c['layout_reference']['path']).resolve()
        need(reference_path.is_file() and
             hashlib.sha256(reference_path.read_bytes()).hexdigest() == c['layout_reference']['sha256'],
             'calibrated_layout_reference_missing_or_changed')
        reference = cv2.imread(str(reference_path))
        need(reference is not None and list(reference.shape[:2][::-1]) == c['image_size'], 'invalid_layout_reference')
        backend = c.get('inference_backend', 'in_process')
        need(backend in ('in_process', 'http'), 'unknown_inference_backend')
        if backend == 'http':
            detector = HTTPYoloe(c, (base / c['yoloe_config']).resolve())
        else:
            need(c.get('model_family','yoloe') == 'yoloe', 'trained_control_model_requires_explicit_http_adapter')
            from .yoloe_inference import YoloeDetector
            detector = YoloeDetector((base / c['yoloe_config']).resolve(), device=c['device'], half=c['half'])
        mode = c['prompt_mode']
        need(detector.model_ids[mode] == c['model_version'] and
             detector.execution_id(mode) == c['runtime_id'], 'reviewed_yoloe_runtime_changed')
        stats = PLStatistics(c['statistics']) if c['statistics']['backend'] == 'pl_http' else Statistics()
        source = CameraYoloe(c, detector)
        motion = Motion(c['arm'], (base / c['stop_file']).resolve())
        controller = Controller(c, source, stats, motion, journal, reference,
                                mode='physical' if args.execute else 'preview')
        deadline = time.monotonic()+args.max_seconds
        completed = 0
        while time.monotonic() < deadline and completed < args.max_tasks:
            # Reserve enough batch budget for a complete cycle rather than
            # abandoning a known path because the observation budget expired.
            if args.execute and deadline-time.monotonic() < c['max_cycle_s']:
                print(json.dumps(dict(status='batch_budget_exhausted', completed=completed)), flush=True)
                return 0
            outcome = controller.cycle()
            print(json.dumps(outcome, ensure_ascii=False, allow_nan=False), flush=True)
            if outcome['status'] == 'empty_scene':
                if not args.execute:
                    return 0
                time.sleep(.1)
                continue
            if outcome['status'] != 'completed_visual_evidence':
                return 0 if outcome['status'] == 'preview_only' else 2
            completed += 1
        print(json.dumps(dict(status='batch_finished', completed=completed)), flush=True)
        return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps(dict(status='startup_refused', reason=type(exc).__name__+':'+str(exc),
                              automatic_retry=False), ensure_ascii=False), flush=True)
        raise SystemExit(2)
