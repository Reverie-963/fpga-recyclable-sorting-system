"""Independent official YOLOE playground: prompt-free, text and boxed references."""
import argparse
import base64
from collections import Counter, deque
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import threading
import time
from urllib.parse import urlsplit
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'local-only/yoloe-official/assets'
ORIGINAL = ROOT / 'local-only/yoloe-deployment/assets'


def parse_classes(value):
    if isinstance(value, str):
        value = re.split(r'[,，;；\n]+', value)
    if (not isinstance(value, list) or not 1 <= len(value) <= 80 or
            any(not isinstance(x, str) or not 1 <= len(x.strip()) <= 100 for x in value)):
        raise ValueError('请输入 1–80 个类别，每个类别最多 100 字符，用逗号或换行分隔')
    return list(dict.fromkeys(x.strip() for x in value))


def decode_image(value):
    import cv2
    import numpy as np
    from PIL import Image
    if not isinstance(value, str) or len(value) > 16000000:
        raise ValueError('图片格式无效或文件过大')
    data = base64.b64decode(value, validate=True)
    with Image.open(io.BytesIO(data)) as image:
        w, h = image.size
        if not 1 <= w <= 2560 or not 1 <= h <= 2560 or w*h > 4500000 or getattr(image, 'n_frames', 1) != 1:
            raise ValueError('请选择单张图片，最长边不超过 2560 像素、总像素不超过 450 万')
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError('无法读取图片')
    return frame


def validate_box(box, shape):
    if (not isinstance(box, list) or len(box) != 4 or
            any(type(x) not in (int, float) or not math.isfinite(x) for x in box)):
        raise ValueError('请在参考图上拖动框选目标')
    x0, y0, x1, y1 = box
    h, w = shape[:2]
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h and x1-x0 >= 3 and y1-y0 >= 3):
        raise ValueError('参考框超出图片或尺寸过小')
    return [float(x) for x in box]


def jpeg64(frame):
    import cv2
    ok, data = cv2.imencode('.jpg', frame)
    if not ok:
        raise RuntimeError('无法生成结果图片')
    return base64.b64encode(data).decode('ascii')


class OfficialDetector:
    def __init__(self, device):
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['OMP_NUM_THREADS'] = '4'
        os.environ['MKL_NUM_THREADS'] = '4'
        os.environ['YOLO_AUTOINSTALL'] = 'false'
        os.environ['YOLO_CONFIG_DIR'] = str(ASSETS.parent / 'settings')
        (ASSETS.parent / 'settings/Ultralytics').mkdir(parents=True, exist_ok=True)
        import torch
        import ultralytics
        from ultralytics import YOLOE
        torch.set_num_threads(4)
        self.torch, self.device = torch, device
        self.half = device != 'cpu'
        if device != 'cpu':
            if not torch.cuda.is_available() or not torch.version.hip:
                raise RuntimeError('请求 ROCm GPU，但当前环境没有可用 ROCm 设备')
            torch.backends.cudnn.enabled = False
        self.execution = dict(device=device, device_name=torch.cuda.get_device_name(0) if device != 'cpu' else 'CPU',
                              precision='fp16' if self.half else 'fp32', torch=torch.__version__,
                              ultralytics=ultralytics.__version__, hip=torch.version.hip)
        self.hashes = {}
        for path in (ASSETS / 'yoloe-11m-seg-pf.pt', ORIGINAL / 'yoloe-11m-seg.pt'):
            self.hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if self.hashes['yoloe-11m-seg-pf.pt'] != '0e31ac208a31b591c20c69f6e525110cc9e96a7dbc1dc2e441ae46597205bbfd':
            raise ValueError('官方无提示模型哈希不匹配')
        if self.hashes['yoloe-11m-seg.pt'] != '785199b7cc75a5f4041dec15782a4b8a4105e5a34d7ffb287a0a10781724c5bd':
            raise ValueError('官方提示模型哈希不匹配')
        self.models = dict(free=YOLOE(str(ASSETS / 'yoloe-11m-seg-pf.pt')).to(device),
                           prompted=YOLOE(str(ORIGINAL / 'yoloe-11m-seg.pt')).to(device))
        head = self.models['free'].model.model[-1]
        if not hasattr(head, 'lrpc'):
            raise ValueError('下载的权重缺少无提示识别头')
        self.vocabulary = list(self.models['free'].names.values())
        if len(self.vocabulary) < 1000:
            raise ValueError('无提示权重缺少内置词表')
        self.embeddings = {}
        self.labels = {}
        self.reference = None
        self.defaults = ['person', 'bottle', 'cup', 'box', 'can', 'chair']
        print('Loaded official PF vocabulary:', len(self.vocabulary), flush=True)

    def prepare(self, payload):
        mode = payload.get('mode', 'free')
        if mode not in ('free', 'text', 'visual'):
            raise ValueError('未知识别模式')
        confidence = payload.get('confidence', .25)
        if type(confidence) not in (float, int) or not math.isfinite(confidence) or not .01 <= confidence <= .95:
            raise ValueError('置信度阈值须在 0.01–0.95 之间')
        names = self.labels.get(mode, [])
        model = self.models['prompted']
        if mode == 'text':
            names = parse_classes(payload.get('classes', self.labels.get('text', self.defaults)))
            if names != self.labels.get('text'):
                import numpy as np
                fingerprint = hashlib.sha256(json.dumps(names,ensure_ascii=False).encode('utf-8')).hexdigest()
                cache = ASSETS.parent / 'text-cache' / (fingerprint+'.npy')
                if not cache.is_file():
                    print('Encoding text in dedicated CPU runtime:', names, flush=True)
                    interpreter = ROOT / 'local-only/yoloe-deployment/venv/Scripts/python.exe'
                    worker = Path(__file__).with_name('yoloe_official_text.py')
                    run = subprocess.run([str(interpreter),str(worker),str(cache)],
                        input=json.dumps(names), text=True, encoding='utf-8', capture_output=True,
                        timeout=120, cwd=ORIGINAL, env=dict(os.environ,PYTHONUTF8='1'))
                    if run.returncode:
                        raise RuntimeError('文字编码失败：'+run.stderr[-500:])
                values = np.load(cache,allow_pickle=False)
                if values.shape != (1,len(names),512) or not np.isfinite(values).all():
                    raise ValueError('文字嵌入缓存无效')
                embedding = self.torch.from_numpy(values).to(self.device)
                self.embeddings['text'], self.labels['text'] = embedding, names
        if mode == 'visual':
            if payload.get('reference_base64'):
                import numpy as np
                from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
                frame = decode_image(payload['reference_base64'])
                box = validate_box(payload.get('box'), frame.shape)
                model.model.float()
                model.model.model[-1].nc = 1
                model.model.names = {0: 'object0'}
                predictor = YOLOEVPSegPredictor(overrides=dict(task='segment', mode='predict', save=False,
                    verbose=False, device=self.device, half=False, imgsz=640, batch=1), _callbacks=model.callbacks)
                predictor.set_prompts(dict(bboxes=np.array([box], dtype=np.float32), cls=np.array([0])))
                predictor.setup_model(model=model.model, verbose=False)
                with self.torch.inference_mode():
                    embedding = predictor.get_vpe(frame).detach()
                self.embeddings['visual'], self.labels['visual'] = embedding, ['object0']
                self.reference = dict(box=box, image_size=[frame.shape[1], frame.shape[0]],
                                      image_sha256=hashlib.sha256(frame.tobytes()).hexdigest())
                model.predictor = None
            if 'visual' not in self.embeddings:
                raise ValueError('请上传参考图并框选目标，再应用设置')
            names = ['object0']
        return dict(mode=mode, confidence=float(confidence), classes=names,
                    vocabulary_count=len(self.vocabulary) if mode == 'free' else len(names))

    def predict(self, frame, settings):
        started = time.perf_counter()
        mode = settings['mode']
        model = self.models['free' if mode == 'free' else 'prompted']
        if mode != 'free':
            embedding = self.embeddings[mode].to(self.device)
            model.set_classes(self.labels[mode], embedding)
        raw = model.predict(frame, device=self.device, half=self.half, imgsz=640,
            conf=settings['confidence'], iou=.5, max_det=100, retina_masks=True, agnostic_nms=True,
            verbose=False, save=False)[0]
        if self.device != 'cpu':
            self.torch.cuda.synchronize()
        instances = []
        masks = None
        if settings.get('include_masks') and len(raw.boxes):
            if raw.masks is None or len(raw.masks.data) != len(raw.boxes):
                raise ValueError('实例分割掩膜缺失')
            masks = raw.masks.data.detach().cpu().numpy() > .5
            if masks.shape[1:] != frame.shape[:2]:
                raise ValueError('掩膜与原始图像尺寸不一致')
        for i, (box, score, cls) in enumerate(zip(raw.boxes.xyxy.tolist(), raw.boxes.conf.tolist(), raw.boxes.cls.tolist())):
            obj = dict(class_id=int(cls), label=raw.names[int(cls)], confidence=score, bbox_xyxy=box)
            if masks is not None:
                import numpy as np
                try:
                    from .yoloe_inference import encode_mask
                except ImportError:
                    from yoloe_inference import encode_mask
                ys, xs = np.nonzero(masks[i])
                if not len(xs):
                    continue
                obj.update(mask_rle=encode_mask(masks[i]), area_px=len(xs), detection_bbox_xyxy=box,
                    bbox_xyxy=[int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1])
            instances.append(obj)
        return dict(schema_version='yoloe-official-observation-v1', mode=mode,
            instances=instances, counts=dict(Counter(x['label'] for x in instances)),
            image_size=[frame.shape[1], frame.shape[0]], confidence_threshold=settings['confidence'],
            source_bgr_sha256=hashlib.sha256(frame.tobytes()).hexdigest(),
            inference_ms=(time.perf_counter()-started)*1000, execution=self.execution,
            model_name='yoloe-11m-seg-pf.pt' if mode == 'free' else 'yoloe-11m-seg.pt',
            model_sha256=self.hashes['yoloe-11m-seg-pf.pt' if mode == 'free' else 'yoloe-11m-seg.pt'],
            vocabulary_count=settings['vocabulary_count'], classes=settings['classes'],
            reference=self.reference if mode == 'visual' else None,
            overlay_base64=(jpeg64(raw[raw.boxes.conf >= settings.get('display_confidence', settings['confidence'])].plot())
                            if settings.get('return_overlay',True) else None),
            automatic_motion_allowed=False,
            generated_at=time.time())


class Playground:
    freshness = 5

    def __init__(self, detector, camera_url):
        url = urlsplit(camera_url)
        if url.scheme != 'http' or url.hostname not in ('127.0.0.1', 'localhost') or url.path != '/frame.jpg' or url.query or url.fragment or url.username or url.password:
            raise ValueError('只允许复用本机 /frame.jpg 相机桥接')
        self.detector, self.camera_url = detector, camera_url
        self.lock, self.compute = threading.RLock(), threading.Lock()
        self.settings = detector.prepare(dict(mode='free'))
        self.generation = 0
        self.enabled = True
        self.frame = self.result = None
        self.camera_error = self.inference_error = None
        self.done = threading.Event()
        self.completed = deque(maxlen=30)
        self.control_pending = 0

    def fresh(self, packet):
        return packet is not None and time.monotonic()-packet['received']+packet['upstream_age_ms']/1000 < self.freshness

    def configure(self, payload):
        enabled = payload.get('enabled', True)
        if type(enabled) is not bool:
            raise ValueError('enabled 须为布尔值')
        with self.compute:
            settings = self.detector.prepare(payload)
            with self.lock:
                self.settings = settings
                self.enabled = enabled
                self.generation += 1
                self.result = None
                self.completed.clear()
                self.inference_error = None
        return self.health()

    def health(self):
        with self.lock:
            times = list(self.completed)
            fps = (len(times)-1)/(times[-1]-times[0]) if len(times)>1 and time.monotonic()-times[-1]<5 else 0
            return dict(ready=True, settings=dict(self.settings), generation=self.generation,
                camera_connected=self.fresh(self.frame), camera_error=self.camera_error,
                enabled=self.enabled, inference_error=self.inference_error, recognition_fps=fps,
                execution=self.detector.execution, model_hashes=self.detector.hashes,
                vocabulary_count=len(self.detector.vocabulary), visual_ready='visual' in self.detector.embeddings,
                freshness_limit_ms=self.freshness*1000, automatic_motion_allowed=False)

    def camera(self):
        with self.lock:
            return self.frame if self.fresh(self.frame) else None

    def latest(self):
        with self.lock:
            if not self.enabled or not self.fresh(self.result) or not self.fresh(self.frame):
                return None
            age = (time.monotonic()-self.result['received'])*1000+self.result['upstream_age_ms']
            return dict(self.result['data'], result_age_ms=age)

    def capture_loop(self):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while not self.done.is_set():
            try:
                with opener.open(self.camera_url, timeout=2) as response:
                    data = response.read(10000001)
                    identity = (response.headers.get('X-Stream-Id'), response.headers.get('X-Frame-Id'))
                    age = float(response.headers['X-Frame-Age-Ms'])
                if not all(identity) or not math.isfinite(age) or not 0 <= age < 2000:
                    raise ValueError('相机帧标识或帧龄无效')
                with self.lock:
                    previous = self.frame
                if previous is None or previous['identity'] != identity:
                    frame = decode_image(base64.b64encode(data).decode('ascii'))
                    with self.lock:
                        if previous and previous['identity'][0] != identity[0]:
                            self.result = None
                            self.generation += 1
                        self.frame = dict(bgr=frame, jpeg=data, identity=identity, received=time.monotonic(), upstream_age_ms=age)
                self.camera_error = None
            except Exception as exc:
                with self.lock:
                    self.frame = self.result = None
                    self.camera_error = str(exc)[:160]
            self.done.wait(.15)

    def inference_loop(self):
        previous = None
        while not self.done.is_set():
            try:
                with self.compute:
                    with self.lock:
                        frame, generation, settings, enabled = self.frame, self.generation, dict(self.settings), self.enabled
                    identity = (frame['identity'], generation) if frame else None
                    if enabled and not self.control_pending and self.fresh(frame) and identity != previous:
                        data = self.detector.predict(frame['bgr'], settings)
                        data.update(source='camera', generation=generation, source_stream_id=frame['identity'][0], source_frame_id=frame['identity'][1])
                        with self.lock:
                            if generation == self.generation and self.enabled and self.fresh(frame) and self.fresh(self.frame) and frame['identity'][0] == self.frame['identity'][0]:
                                self.result = dict(data=data, received=frame['received'], upstream_age_ms=frame['upstream_age_ms'])
                                self.inference_error = None
                                self.completed.append(time.monotonic())
                        previous = identity
            except Exception as exc:
                with self.lock:
                    self.result = None
                    self.inference_error = type(exc).__name__ + ': ' + str(exc)[:220]
                self.done.wait(1)
            self.done.wait(.15)

    def start(self):
        for target in (self.capture_loop, self.inference_loop):
            threading.Thread(target=target, daemon=True).start()


def make_server(playground, port, page_name='yoloe_official.html'):
    page = Path(__file__).with_name('monitor_assets').joinpath(page_name).read_bytes()

    class Handler(BaseHTTPRequestHandler):
        timeout = 120

        def log_message(self, *args):
            pass

        def allowed(self):
            hosts = {'127.0.0.1:'+str(self.server.server_port), 'localhost:'+str(self.server.server_port)}
            return self.headers.get('Host') in hosts and (not self.headers.get('Origin') or self.headers['Origin'] in {'http://'+x for x in hosts})

        def reply(self, code, value, kind='application/json; charset=utf-8'):
            data = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode('utf-8')
            try:
                self.send_response(code)
                for k,v in {'Content-Type':kind,'Content-Length':str(len(data)), 'Cache-Control':'no-store', 'X-Content-Type-Options':'nosniff'}.items():
                    self.send_header(k,v)
                self.end_headers()
                self.wfile.write(data)
            except (ConnectionError, TimeoutError):
                pass

        def do_GET(self):
            if not self.allowed():
                return self.reply(403, dict(error='仅允许本机同源访问'))
            path = urlsplit(self.path).path
            if path == '/':
                return self.reply(200,page,'text/html; charset=utf-8')
            if path == '/health':
                return self.reply(200,playground.health())
            if path == '/backgrounds' and hasattr(playground, 'backgrounds'):
                return self.reply(200,playground.backgrounds())
            if path.startswith('/background/') and path.endswith('.jpg') and hasattr(playground, 'background_records'):
                try:
                    return self.reply(200,playground.background_records.image(path[len('/background/'):-4]),'image/jpeg')
                except (ValueError, OSError, KeyError):
                    return self.reply(404,dict(error='背景记录不存在或校验失败'))
            if path == '/policy' and hasattr(playground.detector, 'policy'):
                return self.reply(200,playground.detector.policy)
            if path == '/vocabulary':
                return self.reply(200,playground.detector.vocabulary)
            if path == '/example.jpg':
                return self.reply(200,(ASSETS/'bus.jpg').read_bytes(),'image/jpeg')
            if path == '/camera.jpg':
                frame = playground.camera()
                return self.reply(200,frame['jpeg'],'image/jpeg') if frame else self.reply(503,dict(error='相机暂不可用'))
            if path == '/live-result':
                result = playground.latest()
                return self.reply(200,result) if result else self.reply(503,dict(error='等待新的识别结果'))
            return self.reply(404,dict(error='not_found'))

        def do_POST(self):
            if not self.allowed():
                return self.reply(403,dict(error='仅允许本机同源访问'))
            if self.path not in ('/configure','/detect','/plan','/backgrounds'):
                return self.reply(404,dict(error='not_found'))
            try:
                length = int(self.headers.get('Content-Length','0'))
                if not 0 < length <= 18000000:
                    raise ValueError('请求过大或为空')
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload,dict):
                    raise ValueError('请求须为 JSON 对象')
                if self.path == '/configure':
                    return self.reply(200,playground.configure(payload))
                if self.path == '/backgrounds':
                    if not hasattr(playground, 'record_background'):
                        return self.reply(404,dict(error='not_found'))
                    return self.reply(200,playground.record_background(payload))
                if self.path == '/plan':
                    if not hasattr(playground, 'plan'):
                        return self.reply(404,dict(error='not_found'))
                    return self.reply(200,playground.plan(payload))
                if payload.get('source') == 'camera':
                    packet = playground.camera()
                    if packet is None:
                        raise ValueError('相机暂不可用，请上传图片')
                    frame = packet['bgr']
                else:
                    frame = decode_image(payload.get('image_base64'))
                control_only=payload.get('control_only',False)
                if type(control_only) is not bool:
                    raise ValueError('control_only 须为布尔值')
                with playground.lock:
                    if control_only:playground.control_pending+=1
                try:
                    with playground.compute:
                        with playground.lock:
                            settings, generation = dict(playground.settings), playground.generation
                        if control_only:settings['return_overlay']=False
                        result = playground.detector.predict(frame,settings)
                        result.update(source=payload.get('source','upload'), generation=generation)
                finally:
                    with playground.lock:
                        if control_only:playground.control_pending-=1
                return self.reply(200,result)
            except (ValueError, TypeError, KeyError, OSError) as exc:
                return self.reply(400,dict(error=str(exc)))
            except Exception as exc:
                return self.reply(500,dict(error=type(exc).__name__+': '+str(exc)))

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        allow_reuse_address = False

        def server_bind(self):
            if hasattr(socket,'SO_EXCLUSIVEADDRUSE'):
                self.socket.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
            super().server_bind()

    return Server(('127.0.0.1',port),Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8811)
    parser.add_argument('--device', choices=['cpu','cuda:0'], default='cuda:0')
    parser.add_argument('--camera-url',default='http://127.0.0.1:8780/frame.jpg')
    args = parser.parse_args()
    # The installed official text encoder resolves its cached file from the working directory.
    os.chdir(ORIGINAL)
    detector = OfficialDetector(args.device)
    playground = Playground(detector,args.camera_url)
    server = make_server(playground,args.port)
    playground.start()
    print('Official YOLOE playground ready: http://127.0.0.1:'+str(server.server_port),flush=True)
    try:
        server.serve_forever()
    finally:
        playground.done.set()
        server.server_close()


if __name__ == '__main__':
    main()
