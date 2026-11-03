"""Continuously send the PYNQ camera to protocol-v2 detection; no PL or arm access."""
import argparse
import json
import math
import time
from pathlib import Path
from urllib.error import HTTPError, URLError

from .camera import LatestCamera
from .detect_remote import DetectionClient


def run(cfg, fps=5.0, max_frames=0):
    if not math.isfinite(fps) or not 0 < fps <= 30:
        raise ValueError('fps must be greater than 0 and at most 30')
    if type(max_frames) is not int or max_frames < 0:
        raise ValueError('frames must be 0 (continuous) or a positive integer')
    camera_cfg = cfg['camera']
    w, h = camera_cfg['width'], camera_cfg['height']
    if type(w) is not int or type(h) is not int or not (0 < w <= 1920 and 0 < h <= 1920 and w*h <= 1920*1080):
        raise ValueError('Camera dimensions exceed protocol-v2 bounds')
    inference = cfg['inference']
    for key in ('timeout_s', 'max_result_age_s'):
        value = inference[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('Invalid inference ' + key)
    client = DetectionClient(inference)
    camera = LatestCamera(camera_cfg)
    last = count = 0
    try:
        while not max_frames or count < max_frames:
            started = time.monotonic()
            fid, stamp, frame = camera.read(after=last)
            last = fid
            if frame.shape[:2] != (h, w):
                raise ValueError('Camera dimensions differ from configuration')
            try:
                client.predict(frame, fid, stamp)
            except HTTPError as exc:
                if exc.code < 500:
                    raise  # Bad token, request or endpoint requires operator correction.
                print(json.dumps({'event': 'observation_retry', 'frame_id': fid,
                                  'reason': 'http_' + str(exc.code)}), flush=True)
                time.sleep(1)
            except (URLError, TimeoutError, OSError) as exc:
                # Observation can retry a NEW frame; mechanical controllers keep their fail-stop policy.
                print(json.dumps({'event': 'observation_retry', 'frame_id': fid,
                                  'reason': type(exc).__name__}), flush=True)
                time.sleep(1)
            count += 1
            time.sleep(max(0, 1 / fps - (time.monotonic() - started)))
    finally:
        camera.close()
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--fps', type=float, default=5, help='Maximum send rate, not a measured camera FPS')
    parser.add_argument('--frames', type=int, default=0, help='0 means run until Ctrl+C')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    try:
        run(cfg, args.fps, args.frames)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
