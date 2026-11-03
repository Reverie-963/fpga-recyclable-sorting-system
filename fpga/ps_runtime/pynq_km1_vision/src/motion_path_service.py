"""Loopback-only math service. Attaches to an already loaded joint overlay.

No programming, camera, serial or actuator API is exposed. One request contains
one complete candidate route; hardware errors latch until operator recovery.
"""
import argparse
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import signal
import time

from .motion_path import checked_segments


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def input_digest(segments):
    return hashlib.sha256(encoded(segments)).hexdigest()


def handler(backend, token, assets):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def setup(self):
            super().setup()
            self.connection.settimeout(5)

        def log_message(self, *args):
            pass  # Never log Authorization or request payloads.

        def reply(self, code, value):
            body = encoded(value)
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Connection', 'close')
            self.end_headers()
            self.wfile.write(body)
            self.close_connection = True

        def authenticated(self):
            return hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer '+token)

        def do_GET(self):
            if not self.authenticated():
                return self.reply(403, {'error': 'unauthorized'})
            if self.path != '/health':
                return self.reply(404, {'error': 'unknown_math_endpoint'})
            return self.reply(200, dict(backend=backend.backend, assets=assets,
                ready=not backend.closed and not backend.faulted, faulted=backend.faulted,
                physical_motion=False))

        def do_POST(self):
            if not self.authenticated():
                return self.reply(403, {'error': 'unauthorized'})
            if self.path != '/evaluate':
                return self.reply(404, {'error': 'unknown_math_endpoint'})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 1 <= size <= 20000 or self.headers.get('Transfer-Encoding'):
                    raise ValueError('invalid_math_request_length')
                raw = self.rfile.read(size)
                if len(raw) != size:
                    raise ValueError('truncated_math_request')
                request = json.loads(raw)
                nonce = request['nonce']
                if not isinstance(nonce, str) or len(nonce) != 32 or any(c not in '0123456789abcdef' for c in nonce):
                    raise ValueError('invalid_math_nonce')
                segments = checked_segments(request['segments'])
                digest = input_digest(segments)
                if request['input_sha256'] != digest or request['assets'] != assets:
                    raise ValueError('math_request_identity_changed')
            except (ValueError, KeyError, TypeError):
                return self.reply(400, {'error': 'invalid_math_request'})
            started = time.perf_counter_ns()
            try:
                result = backend.evaluate(segments)
            except Exception:
                backend.faulted = True
                return self.reply(503, {'error': 'motion_hardware_fault_requires_review'})
            return self.reply(200, dict(result, nonce=nonce, input_sha256=digest, assets=assets,
                service_elapsed_ms=(time.perf_counter_ns()-started)/1e6, physical_motion=False))
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--token-file', type=Path, required=True)
    parser.add_argument('--port', type=int, default=18840)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    root = args.manifest.resolve().parent
    paths = {name: (root/name).resolve() for name in manifest['assets']}
    for name, path in paths.items():
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != manifest['assets'][name]:
            raise RuntimeError('joint_asset_hash_changed:'+name)
    bit = paths['vision_motion.bit']
    hwh = paths['vision_motion.hwh']
    from .motion_path_driver import PLMotionPath, validate_hwh
    from pynq import Overlay
    validate_hwh(hwh)
    overlay = Overlay(str(bit), download=False)
    stats = {path: (path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns) for path in paths.values()}

    def identity():
        if not overlay.is_loaded() or any(
                (p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns) != stamp for p, stamp in stats.items()):
            raise RuntimeError('joint_overlay_identity_changed')

    identity()
    backend = PLMotionPath(overlay.motion_path_0, identity_check=identity)
    token = args.token_file.read_text().strip()
    if len(token) < 32:
        raise RuntimeError('motion_service_token_too_short')
    server = HTTPServer(('127.0.0.1', args.port), handler(backend, token, manifest['assets']))
    print(encoded(dict(ready=True, port=args.port, assets=manifest['assets'])).decode(), flush=True)
    stopping = False
    def stop(number, frame):
        nonlocal stopping
        stopping = True  # Never interrupt an in-flight DMA call.
    for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(number, stop)
    server.timeout = .25
    try:
        while not stopping:
            server.handle_request()
    finally:
        server.server_close()
        try:
            identity()
            backend.close()
        except Exception:
            # Keep CMA and the IP lease alive if an unknown transfer still owns
            # them. Recovery must inspect the device before terminating us.
            print('motion_shutdown_blocked_dma_review_required', flush=True)
            while True:
                time.sleep(1)


if __name__ == '__main__':
    main()
