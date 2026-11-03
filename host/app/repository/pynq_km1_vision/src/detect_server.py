"""Authenticated bounded single-frame YOLO service, protocol 2."""
import argparse
import base64
import collections
import hmac
import json
import os
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import cv2
import numpy as np
from .detection import YOLODetector


def make_server(address, detector, token, monitor=None):
    if len(token) < 24:
        raise ValueError('Set KM1_TOKEN (at least 24 chars)')
    seen = collections.OrderedDict()
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup(); self.connection.settimeout(3)
        def log_message(self, *args):
            pass
        def reply(self, code, data):
            body = json.dumps(data, allow_nan=False).encode()
            self.send_response(code); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body))); self.end_headers()
            try: self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError): pass
        def authorized(self):
            return hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + token)
        def do_GET(self):
            if not self.authorized(): return self.reply(401, {'error': 'unauthorized'})
            if self.path != '/health': return self.reply(404, {'error': 'not_found'})
            self.reply(200, {'protocol': 2, 'task': 'detect', 'model_id': detector.model_id,
                'runtime_id': detector.runtime_id, 'labels': detector.labels, 'providers': detector.providers})
        def do_POST(self):
            if not self.authorized(): return self.reply(401, {'error': 'unauthorized'})
            if self.path != '/detect': return self.reply(404, {'error': 'not_found'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 8_500_000: raise ValueError('Bad length')
                raw = self.rfile.read(length)
                if len(raw) != length: raise ValueError('Truncated body')
                msg = json.loads(raw); rid = msg['request_id']; fid = msg['frame_id']
                if msg.get('protocol') != 2 or not isinstance(rid, str) or not 8 <= len(rid) <= 120 or type(fid) is not int or fid < 0:
                    raise ValueError('Invalid identity')
                if rid in seen: return self.reply(409, {'error': 'duplicate'})
                data = base64.b64decode(msg['png'], validate=True)
                if len(data) < 24 or data[:8] != b'\x89PNG\r\n\x1a\n': raise ValueError('PNG required')
                w, h = struct.unpack('>II', data[16:24])
                if not (1 <= w <= 1920 and 1 <= h <= 1920 and w*h <= 1920*1080): raise ValueError('Too large')
                frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if frame is None or frame.shape[:2] != (h, w): raise ValueError('Decode mismatch')
                received_at = time.monotonic()
                seen[rid] = None
                while len(seen) > 4096: seen.popitem(last=False)
                try:
                    result = {'protocol': 2, 'request_id': rid, 'frame_id': fid, **detector.predict(frame)}
                except Exception:
                    if monitor is not None:
                        observe(frame, {'request_id': rid, 'frame_id': fid, 'error': 'inference_failed'}, received_at)
                    raise
                self.reply(200, result)
                if monitor is not None:
                    observe(frame, result, received_at)
            except (ValueError, KeyError, TypeError, TimeoutError):
                self.reply(400, {'error': 'invalid_request'})
            except Exception:
                self.reply(500, {'error': 'inference_failed'})
    def observe(frame, result, stamp):
        try:
            monitor.publish(frame, result, stamp)
        except Exception:
            # Preview is optional and cannot change the detection response.
            pass
    return HTTPServer(address, Handler)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', default='config/host_rocm.json')
    p.add_argument('--host', default='127.0.0.1'); p.add_argument('--port', type=int, default=8765)
    p.add_argument('--monitor-port', type=int, help='Enable local observation UI, e.g. 8766 (loopback only)')
    a = p.parse_args(); cfg = json.loads(Path(a.config).read_text(encoding='utf-8-sig'))
    detector = YOLODetector(cfg)
    monitor = monitor_server = monitor_thread = server = None
    try:
        if a.monitor_port is not None:
            from .vision_monitor import MonitorState, make_monitor_server
            monitor = MonitorState()
            monitor_server = make_monitor_server(a.monitor_port, monitor)
        server = make_server((a.host, a.port), detector, os.environ.get('KM1_TOKEN', ''), monitor)
        if monitor_server is not None:
            monitor_thread = threading.Thread(target=monitor_server.serve_forever, daemon=True)
            monitor_thread.start()
            print('Monitor: http://127.0.0.1:' + str(monitor_server.server_port), flush=True)
        print(json.dumps({'address': server.server_address, 'model_id': detector.model_id,
                          'runtime_id': detector.runtime_id, 'providers': detector.providers}), flush=True)
        server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        if server is not None: server.server_close()
        if monitor_server is not None:
            if monitor_thread is not None:
                monitor_server.shutdown()
                monitor_thread.join(timeout=3)
            monitor_server.server_close()
        if monitor is not None: monitor.close()

if __name__ == '__main__': main()
