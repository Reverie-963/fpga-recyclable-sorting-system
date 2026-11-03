"""Loopback-only observation UI; bounded latest-frame mailbox, no control APIs."""
import base64
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

from .detection import validate_detections


class MonitorState:
    def __init__(self):
        self.condition = threading.Condition()
        self.pending = None
        self.current = None
        self.closed = False
        self.sequence = 0
        self.stream_id = uuid.uuid4().hex
        self.previous_stamp = None
        self.worker = threading.Thread(target=self._encode, daemon=True)
        self.worker.start()

    def publish(self, frame, result, received_at):
        # Copy before returning: the detector/caller may reuse its image buffer.
        item = (frame.copy(), json.loads(json.dumps(result, allow_nan=False)), received_at)
        with self.condition:
            self.pending = item  # Replace pending work instead of building a video queue.
            self.condition.notify()

    def _encode(self):
        while True:
            with self.condition:
                while self.pending is None and not self.closed:
                    self.condition.wait()
                if self.closed:
                    return
                frame, result, stamp = self.pending
                self.pending = None
            try:
                h, w = frame.shape[:2]
                failed = result.get('error') is not None
                detections = [] if failed else validate_detections(result['detections'], w, h)
                if not failed and result.get('image_size') != [w, h]:
                    raise ValueError('Preview source dimensions mismatch')
                ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    raise ValueError('Preview JPEG failed')
                packet = {
                    'status': 'inference_failed' if failed else 'ok',
                    'frame_id': result['frame_id'], 'request_id': result['request_id'],
                    'image_size': [w, h], 'detections': detections,
                    'timing': result.get('timing', {}),
                    'model_id': result.get('model_id', ''),
                    'providers': result.get('providers', []),
                    'image': 'data:image/jpeg;base64,' + base64.b64encode(jpeg).decode('ascii'),
                }
            except Exception:
                # Never retain old boxes as if they described a failed/new frame.
                packet = {'status': 'preview_failed', 'detections': []}
            with self.condition:
                self.sequence += 1
                packet['sequence'] = self.sequence
                delta = stamp - self.previous_stamp if self.previous_stamp is not None else 0
                packet['result_rate_hz'] = 1 / delta if delta > 0 else None
                self.previous_stamp = stamp
                self.current = (packet, stamp)
                self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            if self.current is None:
                return {'status': 'waiting', 'sequence': 0, 'stream_id': self.stream_id}
            packet, stamp = self.current
            return {**packet, 'stream_id': self.stream_id, 'age_s': max(0, time.monotonic() - stamp)}

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.worker.join(timeout=3)


def make_monitor_server(port, state):
    static = Path(__file__).with_name('monitor_assets')
    routes = {'/': ('index.html', 'text/html; charset=utf-8'),
              '/monitor.js': ('monitor.js', 'text/javascript; charset=utf-8'),
              '/monitor.css': ('monitor.css', 'text/css; charset=utf-8')}

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(3)

        def log_message(self, *args):
            pass

        def do_GET(self):
            allowed_hosts = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
            if self.headers.get('Host') not in allowed_hosts:
                self.send_error(403)
                return
            if self.path == '/snapshot':
                body = json.dumps(state.snapshot(), allow_nan=False).encode('utf-8')
                content_type = 'application/json'
            elif self.path in routes:
                name, content_type = routes[self.path]
                body = (static / name).read_bytes()
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass

    return ThreadingHTTPServer(('127.0.0.1', port), Handler)
