"""Validate recorded PR #33 driver returns and replay sorting, with no devices."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from .static_mask_contract import check, decode_mask, geometry, visual, compare_snapshot
from .static_sorting import SortingSimulator


def encode_mask(pixels, size):
    """Encode an owned flat driver output (e.g. out.tolist()), not a DMA buffer."""
    check(isinstance(pixels, list) and len(pixels) == size[0]*size[1] and
          all(type(v) is int and v in (0, 1) for v in pixels), 'invalid_output_pixels')
    runs, value, count = [], 0, 0
    for pixel in pixels:
        if pixel == value:
            count += 1
        else:
            runs.append(count)
            value, count = pixel, 1
    runs.append(count)
    decode_mask(runs, size)
    return runs


def export_result(stats, output_pixels, size, job_id, **options):
    """Serialize a completed driver's return; performs no hardware call.

    Caller supplies primitive ints (stats are ints in the PR #33 driver) and
    owned output.tolist(). The receiver validates against original inputs.
    """
    decode_mask([size[0]*size[1]], size)
    check(isinstance(stats, dict) and all(type(v) is int for v in stats.values()), 'invalid_stats')
    check(type(job_id) is int and 0 <= job_id < 2**32, 'invalid_job_id')
    return dict(image_size=copy.deepcopy(size), job_id=job_id,
                stats=copy.deepcopy(stats), output_rle=encode_mask(output_pixels, size), **options)


def checked_assets(assets, reference):
    check(isinstance(assets, dict), 'missing_assets')
    for name in ('vision.bit', 'vision.hwh'):
        digest = assets.get(name)
        check(isinstance(digest, str) and len(digest) == 64 and
              all(c in '0123456789abcdef' for c in digest), 'invalid_asset_hash:' + name)
        check(digest == reference['assets'][name], 'asset_mismatch:' + name)


def replay_bundle(bundle, reference):
    check(bundle.get('schema_version') == 'static-driver-capture-v1', 'unsupported_capture_schema')
    check(bundle.get('provenance') in ('synthetic_fixture', 'recorded_driver_return'), 'invalid_provenance')
    check(bundle.get('contract_version') == 'dual-core-v1.1', 'unsupported_contract')
    checked_assets(bundle.get('assets'), reference)
    check(isinstance(bundle.get('requests'), list) and bundle['requests'], 'missing_requests')
    config = copy.deepcopy(bundle['config'])
    config['require_mask_evidence'] = True
    simulator = SortingSimulator(config)
    records = []
    for entry in bundle['requests']:
        r = copy.deepcopy(entry['request'])
        captures = entry['captures']
        check(set(captures) == {'geometry', 'source', 'bin'}, 'incomplete_capture')
        evidence = r['verification_masks']
        size = r['image_size']
        check(len(r['instances']) == 1, 'single_instance_required')
        mask = decode_mask(r['instances'][0]['mask_rle'], size)
        jobs = set()
        for role in ('geometry', 'source', 'bin'):
            capture = captures[role]
            check(capture.get('image_size') == size, 'capture_size_mismatch')
            check(capture.get('role') == role, 'capture_role_mismatch')
            for field in ('request_id', 'task_id', 'camera_id', 'frame_id', 'model_version'):
                check(type(capture.get(field)) is type(r[field]) and capture[field] == r[field],
                      'capture_identity_mismatch:' + field)
            job = capture['job_id']
            check(type(job) is int and 0 <= job < 2**32 and job not in jobs, 'invalid_capture_job')
            jobs.add(job)
            if role == 'geometry':
                check(type(capture.get('erode')) is int and capture['erode'] == 0 and
                      type(capture.get('dilate')) is int and capture['dilate'] == 0,
                      'unsupported_morphology')
                expected = geometry(mask, size, job)
                output = mask
                r['geometry_job_id'] = job
                r['pl_geometry'] = capture['stats']
            else:
                check(type(capture.get('mode')) is int and capture['mode'] == 1, 'unsupported_visual_mode')
                pair = evidence[role]
                check(job == pair['job_id'], 'capture_job_mismatch')
                before = decode_mask(pair['before_rle'], size)
                after = decode_mask(pair['after_rle'], size)
                expected = visual(before, after, size, job)
                output = [int(a != b) for a, b in zip(before, after)]
                pair['pl_stats'] = capture['stats']
            # ctrl may have AP_DONE cleared by earlier reads; status/job/moments
            # and exact output are checked, never infer completion from ctrl.
            compare_snapshot(capture['stats'], expected)
            check(decode_mask(capture['output_rle'], size) == output, 'capture_pixel_mismatch:' + role)
        result = simulator.process(r, entry['now_ms'])
        result['capture_provenance'] = bundle['provenance']
        records.append(result)
    return {'schema_version': 'static-driver-replay-report-v1',
            'simulation': True, 'provenance': bundle['provenance'],
            'assets': bundle['assets'], 'records': records,
            'completed': sum(r['status'] == 'completed' for r in records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--reference-report', type=Path, required=True,
                        help='Reviewed report containing assets vision.bit/vision.hwh hashes')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        raw = args.capture.read_bytes()
        bundle = json.loads(raw)
        reference = json.loads(args.reference_report.read_bytes())
        report = replay_bundle(bundle, reference)
        report['capture_sha256'] = hashlib.sha256(raw).hexdigest()
        report['reference_sha256'] = hashlib.sha256(args.reference_report.read_bytes()).hexdigest()
    except (ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
        parser.exit(2, 'Capture rejected: %s\n' % exc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'simulation': True, 'provenance': report['provenance'],
                      'tasks':len(report['records']), 'completed':report['completed']}))


if __name__ == '__main__':
    main()
