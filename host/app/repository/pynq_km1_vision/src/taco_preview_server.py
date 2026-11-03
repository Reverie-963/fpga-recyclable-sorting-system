"""Local TACO camera/image preview service; no actuator connections."""
import argparse
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import socket
import time
import threading

sys.path.insert(0, str(Path(__file__).resolve().parent))
from taco_inference import TacoDetector, DEFAULT_CONFIG, annotate


def make_server(detector, port, output, camera_url=None):
    import cv2
    import numpy as np
    allowed_hosts = {'127.0.0.1:' + str(port), 'localhost:' + str(port)}
    allowed_origins = {'http://' + host for host in allowed_hosts}
    output.mkdir(parents=True, exist_ok=True)
    page = Path(__file__).with_name('monitor_assets').joinpath('taco.html').read_bytes()
    state = {'frames': 0, 'last_result': None, 'last_overlay': None}
    inference_lock = threading.Lock()
    live = None
    if camera_url:
        from taco_live_preview import LivePreview
        live = LivePreview(detector, inference_lock, camera_url)

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            self.request.settimeout(15)
            super().setup()

        def log_message(self, fmt, *args):
            pass

        def allowed(self):
            return (self.headers.get('Host') in allowed_hosts and
                    (not self.headers.get('Origin') or self.headers['Origin'] in allowed_origins))

        def reply(self, status, data, kind='application/json; charset=utf-8'):
            if not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=False).encode('utf-8')
            try:
                self.send_response(status)
                self.send_header('Content-Type', kind)
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.end_headers()
                self.wfile.write(data)
            except (ConnectionError, socket.timeout):
                pass

        def do_GET(self):
            if not self.allowed():
                return self.reply(403, {'error': 'local_origin_required'})
            if self.path == '/':
                return self.reply(200, page, 'text/html; charset=utf-8')
            if self.path == '/health':
                camera_state = live.status() if live else dict(camera_connected=False,
                    live_enabled=False, live_available=False, camera_error=None)
                return self.reply(200, dict(model_loaded=True, backend='tensorflow_1.15_cpu',
                    labels=detector.labels[1:], model_sha256=detector.model_id,
                    frames=state['frames'], score_ratio_threshold=detector.cfg['score_ratio_threshold'],
                    sorting_mapping_status='configured',
                    sorting_mapping=detector.cfg['sorting_mapping'],
                    sorting_classes=detector.cfg['sorting_classes'], automatic_motion_allowed=False,
                    quality_gate_passed=False, **camera_state))
            if self.path.startswith('/camera.jpg'):
                packet = live.camera_packet() if live else None
                if packet is None:
                    return self.reply(503, {'error': 'camera_unavailable'})
                return self.reply(200, packet[0], 'image/jpeg')
            if self.path == '/live-result':
                result = live.result_packet() if live else None
                return self.reply(200, result) if result else self.reply(503, {'error': 'waiting_for_live_result'})
            if self.path == '/latest-result' and state['last_result'] is not None:
                return self.reply(200, state['last_result'])
            if self.path.startswith('/overlay.jpg') and state['last_overlay'] is not None:
                return self.reply(200, state['last_overlay'], 'image/jpeg')
            return self.reply(404, {'error': 'not_found'})

        def do_POST(self):
            if not self.allowed():
                return self.reply(403, {'error': 'local_origin_required'})
            if self.path == '/live':
                if live is None:
                    return self.reply(503, {'error': 'camera_not_configured'})
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                    if not 0 < length <= 1000:
                        raise ValueError('Invalid request size')
                    enabled = json.loads(self.rfile.read(length))['enabled']
                    if type(enabled) is not bool:
                        raise ValueError('Expected boolean enabled')
                    live.set_enabled(enabled)
                    return self.reply(200, live.status())
                except (ValueError, TypeError, KeyError, OSError) as exc:
                    return self.reply(400, {'error': str(exc)})
            if self.path != '/detect':
                return self.reply(404, {'error': 'not_found'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 8500000:
                    raise ValueError('Invalid body size')
                payload = json.loads(self.rfile.read(length))
                image = base64.b64decode(payload['image_base64'], validate=True)
                from PIL import Image
                with Image.open(io.BytesIO(image)) as header:
                    w, h = header.size
                    if not 1 <= w <= 1920 or not 1 <= h <= 1080:
                        raise ValueError('Image exceeds 1920x1080')
                frame = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError('Cannot decode image')
            except (ValueError, KeyError, TypeError, OSError) as exc:
                return self.reply(400, {'error': str(exc)})
            try:
                # Browsers may open idle connections. Serve them independently,
                # while keeping the legacy model and evidence writes serialized.
                with inference_lock:
                    result = detector.predict(frame)
                    state['frames'] += 1
                    result.update(frame_id=state['frames'], input_kind='uploaded_image',
                                  image_sha256=__import__('hashlib').sha256(image).hexdigest())
                    ok, encoded = cv2.imencode('.jpg', annotate(frame, result))
                    if not ok:
                        raise RuntimeError('Overlay encode failed')
                    stamp = time.strftime('%Y%m%d-%H%M%S') + '-{:05d}'.format(state['frames'])
                    # Each upload gets an independent evidence directory.
                    folder = output / stamp
                    folder.mkdir(exist_ok=False)
                    (folder / 'input-image').write_bytes(image)
                    (folder / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
                    (folder / 'overlay.jpg').write_bytes(encoded.tobytes())
                    state.update(last_result=result, last_overlay=encoded.tobytes())
                return self.reply(200, result)
            except Exception as exc:
                return self.reply(500, {'error': str(exc)})

    class Server(ThreadingHTTPServer):
        request_queue_size = 64

    server = Server(('127.0.0.1', port), Handler)
    server.live_preview = live
    if live:
        live.start()
    bound_port = server.server_address[1]
    allowed_hosts.clear()
    allowed_hosts.update({'127.0.0.1:' + str(bound_port), 'localhost:' + str(bound_port)})
    allowed_origins.clear()
    allowed_origins.update('http://' + host for host in allowed_hosts)
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--port', type=int, default=8777)
    parser.add_argument('--camera-url', help='Local camera-only bridge URL, e.g. http://127.0.0.1:8780/frame.jpg')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_CONFIG.parents[2] / 'local-only/taco-deployment/preview')
    args = parser.parse_args()
    detector = TacoDetector(args.config)
    server = make_server(detector, args.port, args.output_dir, args.camera_url)
    print('TACO original model ready: http://127.0.0.1:{}'.format(args.port), flush=True)
    try:
        server.serve_forever()
    finally:
        if server.live_preview:
            server.live_preview.close()
        server.server_close()


if __name__ == '__main__':
    main()
