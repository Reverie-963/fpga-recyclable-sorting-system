"""PYNQ-only bounded dual-core service; no arm or camera endpoints.

Explicit --load-overlay is required. Merely importing this module never loads
an FPGA image. Copy the existing board/ drivers alongside the project.
"""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
import os
from pathlib import Path
import sys
import threading
import uuid

import numpy as np

from .sorting_io import unpack_mask, digest
from .sorting_loop import need


class DualCoreBackend:
    def __init__(self, bitfile):
        bitfile = Path(bitfile).resolve()
        self.assets = {name: digest(path.read_bytes()) for name, path in
                       [('vision.bit', bitfile), ('vision.hwh', bitfile.with_suffix('.hwh'))]}
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'board'))
        from km1_dma_runtime import DualCoreSession
        from km1_mask_overlay import MaskAccel
        from km1_visual_overlay import VisualAccel
        self.session = DualCoreSession(str(bitfile))
        self.geometry = MaskAccel(session=self.session)
        self.visual = VisualAccel(session=self.session)

    def run(self, role, masks, job):
        masks = [m.astype(np.uint8) for m in masks]
        if role == 'geometry':
            return self.geometry.run(masks[0], erode=0, dilate=0, job_id=job, timeout_sec=3)
        return self.visual.run(*masks, mode=1, job_id=job, timeout_sec=3)


def make_server(host, port, token, backend):
    need(len(token) >= 32, 'statistics_token_too_short')
    lock = threading.Lock()
    session_id = uuid.uuid4().hex
    state = {'faulted': False}

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.request.settimeout(5)

        def log_message(self, *args):
            pass

        def authorized(self):
            return hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + token)

        def send_json(self, code, value):
            raw = json.dumps(value, allow_nan=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if not self.authorized():
                self.send_error(403)
            elif self.path == '/health':
                self.send_json(503 if state['faulted'] else 200,
                    dict(contract='sorting-statistics-v1', session_id=session_id,
                         assets=backend.assets, faulted=state['faulted']))
            else:
                self.send_error(404)

        def do_POST(self):
            if not self.authorized():
                self.send_error(403)
                return
            if self.path != '/run':
                self.send_error(404)
                return
            acquired = False
            entered_driver = False
            try:
                need(self.headers.get('Transfer-Encoding') is None, 'chunked_not_supported')
                length = int(self.headers.get('Content-Length', '0'))
                need(0 < length <= 120000, 'invalid_body_size')
                raw = self.rfile.read(length)
                need(len(raw) == length, 'incomplete_body')
                body = json.loads(raw)
                need(body['session_id'] == session_id, 'wrong_statistics_session')
                need(body['role'] in ('geometry', 'visual'), 'invalid_role')
                need(type(body['job_id']) is int and 0 <= body['job_id'] < 2**32, 'invalid_job_id')
                need(isinstance(body['request_id'], str) and len(body['request_id']) == 32, 'invalid_request_id')
                need(len(body['masks']) == (1 if body['role'] == 'geometry' else 2), 'invalid_mask_count')
                masks = [unpack_mask(m, body['image_size']) for m in body['masks']]
                acquired = lock.acquire(blocking=False)
                need(acquired and not state['faulted'], 'statistics_busy_or_faulted')
                entered_driver = True
                stats, output = backend.run(body['role'], masks, body['job_id'])
                need(stats['status'] == 1 and stats['job_id'] == body['job_id'], 'pl_job_or_status_error')
                need(output.size == masks[0].size and np.isin(output, [0, 1]).all(), 'invalid_pl_pixels')
                self.send_json(200, dict(session_id=session_id, assets=backend.assets,
                    request_id=body['request_id'], stats=stats,
                    output_sha256=digest(output.astype(np.uint8).tobytes())))
            except Exception as exc:
                if entered_driver:
                    state['faulted'] = True  # No automatic recovery/reload behind the controller.
                self.close_connection = True
                self.send_json(503 if entered_driver else 400, dict(error=type(exc).__name__ + ':' + str(exc)))
            finally:
                if acquired:
                    lock.release()

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bitfile', type=Path, required=True)
    parser.add_argument('--load-overlay', action='store_true')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8782)
    parser.add_argument('--token-env', default='KM1_STATS_TOKEN')
    args = parser.parse_args()
    if not args.load_overlay:
        parser.error('Loading the reviewed bit/hwh pair requires --load-overlay')
    token = os.environ[args.token_env]
    need(len(token) >= 32, 'statistics_token_too_short')
    backend = DualCoreBackend(args.bitfile)
    server = make_server(args.host, args.port, token, backend)
    print(json.dumps(dict(status='statistics_only', assets=backend.assets)), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
