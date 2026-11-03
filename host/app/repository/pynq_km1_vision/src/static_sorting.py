"""Offline single-object sorting replay. Never imports a hardware backend."""
import argparse
import copy
import json
import math
from pathlib import Path

from .static_mask_contract import decode_mask, geometry, compare_snapshot, verification


class Rejected(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise Rejected(reason)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def text_id(value):
    return isinstance(value, str) and bool(value.strip())


class SortingSimulator:
    """One synchronous task at a time; request IDs remain consumed after failure.

    This is a symbolic path simulator, not robot IK or collision validation.
    A replay session owns deduplication; reuse its recorded requests on restart.
    """
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        self.seen = set()
        self.tasks = set()
        self.last_frame = {}
        self.records = []
        self.busy = False

    def process(self, request, now_ms):
        if not isinstance(request, dict):
            request = {}
        record = {'simulation': True, 'request_id': request.get('request_id'),
                  'task_id': request.get('task_id'), 'camera_id': request.get('camera_id'),
                  'frame_id': request.get('frame_id'), 'model_version': request.get('model_version'),
                  'now_ms': now_ms, 'input': copy.deepcopy(request),
                  'states': ['received'], 'actions': [], 'status': 'rejected', 'reason': None,
                  'automatic_grip_check': False, 'automatic_drop_detection': False}
        try:
            require(not self.busy, 'busy')
            request_id = request.get('request_id')
            require(text_id(request_id), 'invalid_request_id')
            require(request_id not in self.seen, 'duplicate_request')
            self.seen.add(request_id)
            self.busy = True
            task_id = request.get('task_id')
            require(text_id(task_id), 'invalid_identity')
            require(task_id not in self.tasks, 'duplicate_task')
            self.tasks.add(task_id)
            self._run(request, now_ms, record)
        except (ValueError, KeyError, TypeError, IndexError, OverflowError, AttributeError) as exc:
            record['reason'] = str(exc) if isinstance(exc, ValueError) else 'malformed_input'
            record['status'] = 'needs_review' if record['actions'] else 'rejected'
            record['states'].append('stopped')
        finally:
            # Recursive/busy requests must not release the outer task.
            if record['reason'] != 'busy':
                self.busy = False
            self.records.append(copy.deepcopy(record))
        return record

    def _run(self, r, now, out):
        c = self.config
        require(r.get('observation_error') is None, 'observation_unavailable')
        require(r.get('schema_version') == 'static-sim-v1', 'unsupported_schema')
        require(text_id(r.get('task_id')) and text_id(r.get('camera_id')), 'invalid_identity')
        require(r.get('model_version') == c['model_version'], 'model_version_mismatch')
        frame = r.get('frame_id')
        require(type(frame) is int and frame >= 0, 'invalid_frame_id')
        require(frame > self.last_frame.get(r['camera_id'], -1), 'old_frame')
        age = r.get('received_age_ms')
        received = r.get('received_at_ms')
        require(number(now) and number(age) and number(received) and
                age >= 0 and now >= received and
                age + now - received <= c['max_frame_age_ms'], 'stale_frame')
        size = r.get('image_size')
        require(size == c['image_size'], 'image_size_mismatch')
        self.last_frame[r['camera_id']] = frame
        instances = r.get('instances')
        require(isinstance(instances, list), 'invalid_instances')
        require(len(instances) > 0, 'empty_scene')
        require(len(instances) == 1, 'ambiguous_scene')
        obj = instances[0]
        require(text_id(obj.get('instance_id')), 'invalid_instance_id')
        label = obj.get('label')
        require(isinstance(label, str) and label in c['bins'], 'unknown_class')
        confidence = obj.get('confidence')
        require(number(confidence) and c['min_confidence'] <= confidence <= 1, 'low_confidence')
        mask = decode_mask(obj.get('mask_rle'), size)
        foreground = [i for i, value in enumerate(mask) if value]
        require(bool(foreground), 'empty_mask')
        xs = [p % size[0] for p in foreground]
        ys = [p // size[0] for p in foreground]
        box = [min(xs), min(ys), max(xs)+1, max(ys)+1]
        require(obj.get('bbox_xyxy') == box, 'bbox_mask_mismatch')
        # Choose a foreground pixel nearest the centroid (never a hole).
        cx, cy = sum(xs)/len(xs), sum(ys)/len(ys)
        px, py = min(zip(xs, ys), key=lambda p: ((p[0]-cx)**2+(p[1]-cy)**2, p[1], p[0]))
        cal = c.get('calibration')
        require(isinstance(cal, dict) and cal.get('camera_id') == r['camera_id'] and
                text_id(cal.get('version')), 'missing_calibration')
        matrix = cal.get('pixel_to_mm')
        require(isinstance(matrix, list) and len(matrix) == 2 and
                all(isinstance(row, list) and len(row) == 3 and all(number(v) for v in row) for row in matrix), 'invalid_calibration')
        def mapped(x, y):
            return [row[0]*x + row[1]*y + row[2] for row in matrix]
        bounds = c['pick_bounds_mm']
        def inside(point):
            return all(number(v) for v in point) and bounds[0] <= point[0] <= bounds[1] and bounds[2] <= point[1] <= bounds[3]
        require(all(inside(mapped(x, y)) for x, y in zip(xs, ys)), 'out_of_range')
        target = c['bins'][label]
        require(isinstance(target, dict) and text_id(target.get('id')) and
                len(target.get('xy_mm', [])) == 2 and all(number(v) for v in target['xy_mm']), 'invalid_bin')
        require(c.get('path_available') is True, 'no_feasible_path')
        if 'pl_geometry' in r:
            job_id = r.get('geometry_job_id')
            require(type(job_id) is int and 0 <= job_id < 2**32, 'invalid_job_id')
            stats = geometry(mask, size, job_id)
            compare_snapshot(r['pl_geometry'], stats)
            out['geometry_stats'] = stats
        require(not c.get('require_mask_evidence', False) or 'verification_masks' in r,
                'missing_verification_masks')
        evidence = None
        if 'verification_masks' in r:
            require(r['verification_masks'].get('bin_id') == target['id'], 'evidence_bin_mismatch')
            evidence = verification(r, size, mask)
            out['verification_stats'] = evidence
            out['verification_backend'] = 'cpu_reference_with_optional_recorded_pl_check'
            policy = c.get('verification_policy', {})
            fraction = policy.get('min_source_erased_fraction', 0.75)
            gain = policy.get('min_bin_gain_px', 1)
            require(number(fraction) and 0 < fraction <= 1 and
                    type(gain) is int and gain > 0, 'invalid_verification_policy')
            source = evidence['source']
            source_area = source['inter'] + source['erased']
            source_pass = (source_area > 0 and source['erased'] / source_area >= fraction
                           and source['beyond'] == 0)
            bin_pass = evidence['bin']['beyond'] - evidence['bin']['erased'] >= gain
        out['plan'] = {'instance_id': obj['instance_id'], 'label': label, 'pick_pixel': [px, py],
                       'pick_xy_mm': mapped(px, py), 'bin_id': target['id'],
                       'bin_xy_mm': target['xy_mm'], 'calibration_version': cal['version'],
                       'path_kind': 'symbolic_simulation_only'}
        out['states'] += ['validated', 'planned']
        outcomes = r.get('simulated_outcomes', {})
        require('check_grip' not in outcomes, 'unsupported_automatic_grip_check')
        steps = [('pick', 'picked'), ('lift', 'lifted'),
                 ('transfer', 'transferred'), ('release', 'released'),
                 ('retreat', 'retreated'), ('verify_source', 'source_verified'),
                 ('verify_bin', 'bin_verified')]
        for action, state in steps:
            result = outcomes.get(action, 'ok')
            if evidence is not None and result == 'ok':
                if action == 'verify_source' and not source_pass:
                    result = 'source_not_cleared'
                if action == 'verify_bin' and not bin_pass:
                    result = 'bin_not_increased'
            out['actions'].append({'action': action, 'result': result})
            require(result == 'ok', 'execution_unknown' if result == 'unknown' else 'step_failed:' + action)
            out['states'].append(state)
        out['states'].append('completed')
        out['status'] = 'completed'


def replay(document):
    sim = SortingSimulator(document['config'])
    for entry in document['requests']:
        sim.process(entry['request'], entry['now_ms'])
    return sim.records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    records = replay(json.loads(args.replay.read_text(encoding='utf-8')))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'simulation': True, 'tasks': len(records), 'completed': sum(r['status'] == 'completed' for r in records)}))


if __name__ == '__main__':
    main()
