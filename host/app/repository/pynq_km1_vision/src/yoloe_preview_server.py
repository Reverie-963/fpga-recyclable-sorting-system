"""Shared local image/live camera monitor and legacy YOLOE launcher. No actuators."""
import argparse
from contextlib import nullcontext
import base64
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import time
from urllib.parse import urlsplit, parse_qs
import uuid
import threading
import socket

try:
    from .yoloe_inference import YoloeDetector, DEFAULT_CONFIG, ROOT, annotate
except ImportError:
    from yoloe_inference import YoloeDetector, DEFAULT_CONFIG, ROOT, annotate


def make_server(detector, port, output, examples_dir=None, camera_url=None):
    import cv2
    import numpy as np
    from PIL import Image
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    page = Path(__file__).with_name('monitor_assets').joinpath('yoloe.html').read_bytes()
    examples = {}
    if examples_dir and (Path(examples_dir) / 'index.json').is_file():
        for item in json.loads((Path(examples_dir) / 'index.json').read_text(encoding='utf-8')):
            path = (Path(examples_dir) / item['file']).resolve()
            if not path.is_relative_to(Path(examples_dir).resolve()) or not path.is_file():
                raise ValueError('Example outside sample directory')
            examples[item['id']] = dict(title=item['title'], path=path)
    state = {'frames': 0, 'last_result': None, 'last_overlay': None}
    inference_lock = threading.Lock()
    try:
        from .monitor_control import ControlTelemetry
    except ImportError:
        from monitor_control import ControlTelemetry
    control_telemetry = ControlTelemetry()
    live = None
    if camera_url:
        try:
            from .yoloe_live_preview import LivePreview
        except ImportError:
            from yoloe_live_preview import LivePreview
        live = LivePreview(detector, inference_lock, camera_url)
    server = None

    class Handler(BaseHTTPRequestHandler):
        timeout = 20

        def log_message(self, fmt, *args):
            pass

        def allowed(self):
            port_actual = server.server_port
            hosts = {'127.0.0.1:' + str(port_actual), 'localhost:' + str(port_actual)}
            origins = {'http://' + host for host in hosts}
            return (self.headers.get('Host') in hosts and
                    (not self.headers.get('Origin') or self.headers['Origin'] in origins))

        def reply(self, status, data, kind='application/json; charset=utf-8', headers=None):
            if not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=False).encode('utf-8')
            try:
                self.send_response(status)
                self.send_header('Content-Type', kind)
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                for key, value in (headers or {}).items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(data)
            except (ConnectionError, TimeoutError):
                pass

        def do_GET(self):
            if not self.allowed():
                return self.reply(403, {'error': 'local_origin_required'})
            path = urlsplit(self.path).path
            if path == '/':
                return self.reply(200, page, 'text/html; charset=utf-8')
            if path == '/health':
                camera = live.status() if live else dict(live_available=False, camera_connected=False,
                                                         live_enabled=False)
                return self.reply(200, dict(detector.health(), frames=state['frames'], **camera))
            if path == '/benchmark':
                try:
                    from .monitor_benchmark import benchmark_summary
                except ImportError:
                    from monitor_benchmark import benchmark_summary
                return self.reply(200, benchmark_summary(detector.health()))
            if path == '/control-status':
                task=parse_qs(urlsplit(self.path).query).get('task_id',[None])[0]
                return self.reply(200, control_telemetry.status(detector.health(),task_id=task))
            if path == '/camera.jpg':
                packet = live.camera_packet() if live else None
                return (self.reply(200, packet[0], 'image/jpeg',
                    {'X-Frame-Id': str(packet[1]), 'X-Frame-Age-Ms': str(round(packet[2], 2))})
                    if packet else self.reply(503, {'error': 'camera_unavailable'}))
            if path == '/live-result':
                result = live.result_packet() if live else None
                return self.reply(200, result) if result else self.reply(503, {'error': 'waiting_for_fresh_result'})
            if path == '/examples':
                return self.reply(200, [dict(id=k, title=v['title'], url='/examples/' + k)
                                        for k, v in examples.items()])
            if path.startswith('/examples/') and path[len('/examples/'):] in examples:
                return self.reply(200, examples[path[len('/examples/'):]]['path'].read_bytes(), 'image/jpeg')
            if path == '/latest-result' and state['last_result'] is not None:
                return self.reply(200, state['last_result'])
            if path == '/overlay.jpg' and state['last_overlay'] is not None:
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
                        raise ValueError('Invalid live control size')
                    payload = json.loads(self.rfile.read(length))
                    if not isinstance(payload, dict):
                        raise ValueError('Expected live control object')
                    live.configure(payload['enabled'], payload['prompt_mode'])
                    return self.reply(200, live.status())
                except (KeyError, ValueError, TypeError, OSError) as exc:
                    return self.reply(400, {'error': str(exc)})
            if self.path != '/detect':
                return self.reply(404, {'error': 'not_found'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 8500000:
                    raise ValueError('Invalid body size')
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict) or not isinstance(payload.get('image_base64'), str):
                    raise ValueError('image_base64 must be a string')
                control_only=payload.get('control_only',False)
                if type(control_only) is not bool:raise ValueError('control_only must be boolean')
                mode = payload.get('prompt_mode')
                if mode is not None and mode not in getattr(detector, 'models', ('text', 'visual')):
                    raise ValueError('Requested inference mode is not available')
                image = base64.b64decode(payload['image_base64'], validate=True)
                with Image.open(io.BytesIO(image)) as header:
                    w, h = header.size
                    if not 1 <= w <= 1920 or not 1 <= h <= 1080:
                        raise ValueError('Image exceeds 1920x1080')
                    if getattr(header, 'n_frames', 1) != 1:
                        raise ValueError('Upload a single still image')
                frame = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError('Cannot decode image')
            except (ValueError, KeyError, TypeError, OSError) as exc:
                return self.reply(400, {'error': str(exc)})
            try:
                with (live.control_request() if control_only and live else nullcontext()), inference_lock:
                    predict=getattr(detector,'predict_control',detector.predict) if control_only else detector.predict
                    result = predict(frame, prompt_mode=mode)
                    state['frames'] += 1
                    result.update(frame_id=state['frames'], input_kind='uploaded_image',
                                  image_sha256=hashlib.sha256(image).hexdigest())
                    if control_only:
                        result.pop('overlay_base64',None)
                        result['control_only']=True
                        return self.reply(200,result)
                    ok, overlay = cv2.imencode('.jpg', annotate(frame, result))
                    if not ok:
                        raise RuntimeError('Cannot encode overlay')
                    folder = output / (time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:12])
                    folder.mkdir(exist_ok=False)
                    (folder / 'input-image').write_bytes(image)
                    (folder / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
                    (folder / 'overlay.jpg').write_bytes(overlay.tobytes())
                    state.update(last_result=result, last_overlay=overlay.tobytes())
                return self.reply(200, result)
            except Exception as exc:
                return self.reply(500, {'error': str(exc)})

    class PreviewServer(ThreadingHTTPServer):
        allow_reuse_address = False

        def server_bind(self):
            if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            super().server_bind()

    server = PreviewServer(('127.0.0.1', port), Handler)
    server.live = live
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--port', type=int, default=8778)
    parser.add_argument('--prompt-mode', choices=['text', 'visual'])
    parser.add_argument('--device', choices=['cpu', 'cuda:0'], default='cpu')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--camera-url', help='Existing local camera-only bridge /frame.jpg URL')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'local-only/yoloe-deployment/preview')
    parser.add_argument('--examples-dir', type=Path, default=ROOT / 'local-only/yoloe-deployment/examples')
    args = parser.parse_args()
    detector = YoloeDetector(args.config, device=args.device, half=args.half)
    if args.prompt_mode:
        if args.prompt_mode not in detector.models:
            parser.error('Requested prompt mode has not been prepared')
        detector.default_prompt_mode = args.prompt_mode
    # Complete GPU warmup before publishing a ready monitor.
    if args.device != 'cpu':
        import numpy as np
        for mode in detector.models:
            detector.predict(np.zeros((480, 640, 3), np.uint8), prompt_mode=mode)
    server = make_server(detector, args.port, args.output_dir, args.examples_dir, args.camera_url)
    if server.live:
        server.live.start()
    print('YOLOE three-material preview ready: http://127.0.0.1:' + str(server.server_port), flush=True)
    try:
        server.serve_forever()
    finally:
        if server.live:
            server.live.close()
        server.server_close()


if __name__ == '__main__':
    main()
