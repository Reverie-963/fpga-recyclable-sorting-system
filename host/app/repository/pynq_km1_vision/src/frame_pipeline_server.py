"""Candidate camera/FPGA service; preserves original station evidence semantics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
from urllib.parse import urlsplit, parse_qs
import cv2
import numpy as np

from .frame_pipeline import PLANES
from .station_vision import StationVision, CPUVision
from .station_vision_server import FrameStore, make_server as station_server


class PipelineFrameStore(FrameStore):
    def publish(self, image, captured):
        with self.lock:
            backend = self.engine.backend
            previous_processing = backend.last_processing
            diagnostic = self.engine.baseline_id is None
            if diagnostic and not self.faulted:
                # Grayscale/edge processing is available before an explicit scene
                # baseline; diagnostic differences never become motion evidence.
                if getattr(self, '_diagnostic_config', None) != self.engine.config_sha256:
                    backend.reset(np.zeros_like(image), self.engine.regions, self.engine.threshold)
                    self._diagnostic_config = self.engine.config_sha256
                if self.sequence % self.engine.window_frames == 0:
                    backend.begin_window()
                backend.process(image)
            else:
                self._diagnostic_config = None
            row = super().publish(image, captured)
            processed = not self.faulted and backend.last_processing is not previous_processing
            row['processed_preview'] = backend.preview_image if processed else None
            row['pipeline'] = dict(backend.last_processing if processed else {},
                available=processed,
                frame_id=row['id'], stream_id=self.stream_id,
                assets=backend.assets, config_sha256=self.engine.config_sha256,
                frame_bgr_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                phases=dict(backend.last_timing or {}) if processed else {},
                change_reference='diagnostic_zero_not_task_evidence' if diagnostic else 'explicit_bound_baseline',
                automatic_motion_allowed=False)
            return row

    def processing(self):
        if self.faulted:
            raise ValueError('frame_pipeline_faulted')
        row = self.latest()
        if not row['pipeline']['available']:
            raise ValueError('frame_pipeline_current_frame_not_processed')
        return dict(row['pipeline'], frame_age_ms=(self.clock()-row['captured'])*1000,
            planes=list(PLANES), kernel_only_measured=False,
            startup_comparison={k: v for k, v in getattr(self, 'pipeline_comparison', {}).items()
                                if k != 'measurements'})

    def health(self):
        data = super().health()
        data['capabilities'] = data['capabilities'] + ['frame_pipeline_v1']
        data['pipeline_contract'] = 'station-frame-pipeline-v1'
        return data


def make_server(host, port, token, store):
    server = station_server(host, port, token, store)
    parent = server.RequestHandlerClass

    class Handler(parent):
        def do_GET(self):
            route = urlsplit(self.path)
            if route.path not in ('/processing', '/processed.png'):
                return super().do_GET()
            if not self.authorized():
                return self.reply(403, dict(error='unauthorized'))
            try:
                if route.path == '/processing':
                    return self.reply(200, store.processing())
                query = parse_qs(route.query)
                plane = query.get('plane', ['smooth'])[0]
                if plane not in PLANES:
                    raise ValueError('unknown_processing_plane')
                frame_id = query.get('frame_id', [None])[0]
                row = store.latest(int(frame_id) if frame_id is not None else None)
                image = row.get('processed_preview')
                if image is None or store.faulted:
                    raise ValueError('no_verified_processing_preview')
                ok, png = cv2.imencode('.png', image[:, :, PLANES.index(plane)])
                if not ok:
                    raise ValueError('cannot_encode_processing_plane')
                return self.reply(200, png.tobytes(), 'image/png', dict(
                    **{'X-Frame-Id': row['id'], 'X-Stream-Id': store.stream_id,
                       'X-Frame-Age-Ms': (store.clock()-row['captured'])*1000,
                       'X-Input-Bgr-Sha256': row['pipeline']['frame_bgr_sha256']}))
            except (ValueError, TypeError, KeyError):
                return self.reply(503, dict(error='fresh_processing_frame_unavailable'))
    server.RequestHandlerClass = Handler
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--load-overlay', action='store_true')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8784)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    if (config.get('backend') != 'pl_board_local' or config.get('synthetic_fixture') is not False or
            config.get('pipeline_contract') != 'station-frame-pipeline-v1'):
        parser.error('Use an explicit reviewed frame-pipeline configuration')
    StationVision(CPUVision(), config['regions'], config['image_size'], config['threshold'],
        config['max_gap_s'], config['window_frames'])
    token = os.environ[config['token_env']]
    if len(token) < 32:
        raise ValueError('station_vision_token_too_short')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'board'))
    from km1_frame_pipeline import StationFramePipeline
    backend = StationFramePipeline((args.config.parent/config['bitfile']).resolve(), config['assets'],
        tuple(config['image_size']), load=args.load_overlay, preview_enabled=config.get('preview_enabled', True))
    engine = StationVision(backend, config['regions'], config['image_size'], config['threshold'],
        config['max_gap_s'], config['window_frames'])
    store, stop = PipelineFrameStore(engine), threading.Event()
    server = make_server(args.host, args.port, token, store)
    worker = None
    try:
        from .frame_pipeline_selftest import startup_selftest
        comparison = startup_selftest(backend, config)
        Path('frame-pipeline-selftest-'+str(time.time_ns())+'.json').write_text(
            json.dumps(comparison, indent=2), encoding='utf-8')
        store.pipeline_comparison = comparison
    except Exception as exc:
        store.faulted = True
        store.error = type(exc).__name__+':'+str(exc)

    def capture():
        cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FPS, 10)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['image_size'][0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['image_size'][1])
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                raise RuntimeError('camera_unavailable')
            while not stop.is_set():
                ok, image = cap.read()
                if not ok:
                    raise RuntimeError('camera_capture_failed')
                store.publish(image, time.monotonic())
        except Exception as exc:
            with store.lock:
                store.faulted = True
                store.error = type(exc).__name__+':'+str(exc)
        finally:
            cap.release()
    if not store.faulted:
        worker = threading.Thread(target=capture, daemon=True)
        worker.start()
    print(json.dumps(dict(status='frame_pipeline_candidate', port=args.port,
        all_cpu_comparisons_passed=not store.faulted, mechanical_commands_sent=False)), flush=True)
    try:
        server.serve_forever()
    finally:
        stop.set()
        if worker is not None:
            worker.join(timeout=3)
        server.server_close()
        if worker is not None and worker.is_alive():
            backend.poisoned = True
        backend.close()


if __name__ == '__main__':
    main()
