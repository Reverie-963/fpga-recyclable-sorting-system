"""Read-only board-to-PC inference proof; no baseline/region or actuator writes."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

import cv2
import numpy as np

from .board_detector_transport import RelayDetector, token_value


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def inspect_frame(raw, headers, health, config, started_ms, finished_ms):
    """Use the board controller's monotonic clock; no PC wall-clock comparison."""
    require(len(raw) <= 8_500_000 and raw.startswith(b'\x89PNG\r\n\x1a\n'), 'probe_requires_original_png')
    metadata = headers.get('X-Station-Metadata', '')
    require(isinstance(metadata, str) and 0 < len(metadata) <= 16000, 'probe_snapshot_metadata_missing')
    info = json.loads(base64.b64decode(metadata, validate=True))
    for key in ('stream_id', 'assets', 'config_sha256', 'backend', 'baseline_frame_id'):
        require(info[key] == health[key], 'probe_station_identity_changed:'+key)
    require(info['camera_connected'] and not info['faulted'], 'probe_camera_faulted')
    frame = info['frame_id']
    require(type(frame) is int and headers.get('X-Stream-Id') == info['stream_id'] and
            headers.get('X-Frame-Id') == str(frame), 'probe_frame_identity_mismatch')
    prepare, age = info.get('response_prepare_ms', 0), info['frame_age_ms']
    require(type(prepare) in (int, float) and np.isfinite(prepare) and 0 <= prepare <= finished_ms-started_ms+1,
            'probe_invalid_response_timing')
    require(type(age) in (int, float) and np.isfinite(age) and 0 <= age < config['max_age_ms'],
            'probe_frame_expired')
    stamp = started_ms+prepare-age
    require(finished_ms-stamp <= config['max_age_ms'], 'probe_snapshot_deadline_exceeded')
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    require(image is not None and list(image.shape[1::-1]) == config['image_size'], 'probe_image_size_changed')
    bgr = hashlib.sha256(image.tobytes()).hexdigest()
    require(headers.get('X-Decoded-Bgr-Sha256') == bgr, 'probe_snapshot_pixels_changed')
    return image, dict(source_stream_id=info['stream_id'], source_frame_id=frame,
        snapshot_png_sha256=hashlib.sha256(raw).hexdigest(), bgr_sha256=bgr,
        capture_started_ms=stamp, snapshot_age_upper_ms=finished_ms-stamp,
        config_sha256=info['config_sha256'], baseline_frame_id=info['baseline_frame_id'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--relay-token-file', type=Path, required=True)
    p.add_argument('--station-token-file', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--frames', type=int, choices=(1, 3), default=1)
    args = p.parse_args()
    args.output.mkdir(exist_ok=False)
    if hasattr(os, 'nice'):
        os.nice(19)
    cv2.setNumThreads(1)
    c = json.loads(args.config.read_text(encoding='utf-8'))
    station_token = args.station_token_file.read_text().strip()
    relay_token = token_value(args.relay_token_file.read_text().strip())
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    # Fixed, existing board camera endpoint. No generic URL or mutation routes.
    def station(route):
        call = urllib.request.Request('http://127.0.0.1:8783/'+route,
                                      headers={'Authorization':'Bearer '+station_token})
        with opener.open(call, timeout=1.5) as response:
            raw = response.read(8_500_001)
            return raw, response.headers
    report = dict(kind='board-detector-readonly-proof-v1', status='checking', frames=[],
        physical_motion=False, execution_enabled=False, baseline_changed_by_probe=False,
        regions_changed_by_probe=False, source=c['migration_source'])
    exit_code = 1
    try:
        health = json.loads(station('health')[0])
        require(health['camera_connected'] and not health['faulted'] and
                health['backend'] == 'pl_board_local' and health['assets'] == c['statistics']['assets'],
                'probe_board_identity_or_health_mismatch')
        detector = RelayDetector(c, relay_token)
        previous = -1
        for index in range(args.frames):
            started = time.monotonic()*1000
            raw, headers = station('snapshot.png')
            image, identity = inspect_frame(raw, headers, health, c, started, time.monotonic()*1000)
            require(identity['source_frame_id'] > previous, 'probe_frame_not_new')
            previous = identity['source_frame_id']
            (args.output/f'frame-{index}.png').write_bytes(raw)
            before = time.monotonic()*1000
            observation = detector.predict(image, c['prompt_mode'])
            done = time.monotonic()*1000
            row = dict(identity, model_input_sha256=observation['image_sha256'],
                model_version=observation['model_version'], runtime_id=observation['runtime_id'],
                inference_roundtrip_ms=done-before, frame_age_upper_ms=done-identity['capture_started_ms'],
                fresh=done-identity['capture_started_ms'] <= c['max_age_ms'],
                instance_count=len(observation.get('instances', [])))
            report['frames'].append(row)
            (args.output/f'observation-{index}.json').write_text(json.dumps(observation, allow_nan=False))
            require(row['fresh'], 'probe_inference_completed_after_original_frame_deadline')
        final = json.loads(station('health')[0])
        for key in ('stream_id', 'assets', 'config_sha256', 'backend', 'baseline_frame_id'):
            require(final[key] == health[key], 'probe_station_changed_during_readonly_check:'+key)
        report.update(status='passed', all_frames_fresh=True)
        exit_code = 0
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__, reason=str(exc))
    finally:
        (args.output/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(dict(status=report['status'], frames=len(report['frames']), reason=report.get('reason'))))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
