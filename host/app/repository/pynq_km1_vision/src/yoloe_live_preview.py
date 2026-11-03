from contextlib import contextmanager
"""Latest-frame YOLOE monitor. Reuses a local camera bridge; never controls hardware."""
import base64
from collections import deque
import hashlib
import math
import threading
import time
from urllib.parse import urlsplit
import urllib.request


class LivePreview:
    def __init__(self, detector, inference_lock, camera_url, clock=time.monotonic):
        url = urlsplit(camera_url)
        if (url.scheme != 'http' or url.hostname not in ('127.0.0.1', 'localhost') or
            url.username or url.password or url.path != '/frame.jpg' or url.query or url.fragment):
            raise ValueError('Use a local camera-only /frame.jpg source')
        self.detector, self.inference_lock, self.camera_url = detector, inference_lock, camera_url
        self.clock = clock
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.frame = self.result = None
        self.sequence = self.generation = 0
        self.mode = detector.default_prompt_mode
        self.enabled = True
        self.control_pending=0
        self.error = '等待相机'
        self.inference_error = None
        self.previous = None
        self.completed = deque(maxlen=60)
        self.threads = []

    def start(self):
        for target in (self._camera, self._infer):
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            self.threads.append(thread)

    def close(self):
        self.done.set()
        for thread in self.threads:
            thread.join(timeout=3)

    def fresh(self, packet, now=None):
        return (packet is not None and
                (self.clock() if now is None else now) - packet['received'] + packet['upstream_age_ms']/1000 < 2)

    def configure(self, enabled, mode):
        if type(enabled) is not bool or mode not in self.detector.models:
            raise ValueError('Invalid live control or unprepared prompt mode')
        with self.lock:
            if self.enabled != enabled or self.mode != mode:
                self.generation += 1
                self.result = None
                self.completed.clear()
            self.enabled, self.mode = enabled, mode

    def disconnected(self, error):
        with self.lock:
            self.frame = self.result = None
            self.generation += 1
            self.completed.clear()
            self.error = error

    def accept_frame(self, jpeg, stream_id, source_id, upstream_age_ms):
        import cv2
        import numpy as np
        if (not isinstance(stream_id, str) or not stream_id or not isinstance(source_id, str) or
            not source_id or not math.isfinite(upstream_age_ms) or not 0 <= upstream_age_ms < 2000 or
            not 0 < len(jpeg) <= 8500000):
            raise ValueError('Invalid camera identity, size, or age')
        identity = (stream_id, source_id)
        with self.lock:
            if self.frame is not None and self.frame['identity'] == identity:
                return  # Polling the same frame must not refresh its age.
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.shape[0] > 1080 or frame.shape[1] > 1920:
            raise ValueError('Invalid camera JPEG')
        with self.lock:
            if self.frame is None or self.frame['identity'][0] != stream_id:
                self.generation += 1
                self.result = None
            self.sequence += 1
            self.frame = dict(jpeg=jpeg, bgr=frame, received=self.clock(),
                upstream_age_ms=upstream_age_ms, identity=identity, sequence=self.sequence)
            self.error = None

    def status(self):
        with self.lock:
            connected = self.fresh(self.frame)
            times = list(self.completed)
            fps = ((len(times)-1)/(times[-1]-times[0]) if len(times) > 1 and
                   times[-1] > times[0] and self.clock()-times[-1] < 2 and self.enabled and connected else 0.)
            return dict(live_available=True, camera_connected=connected,
                live_enabled=self.enabled, live_prompt_mode=self.mode,
                camera_frame_id=self.frame['sequence'] if connected else None,
                camera_error=self.error if not connected else None,
                inference_error=self.inference_error, recognition_fps=fps,
                freshness_limit_ms=2000, automatic_motion_allowed=False)

    def camera_packet(self):
        with self.lock:
            if not self.fresh(self.frame):
                return None
            age = (self.clock()-self.frame['received'])*1000 + self.frame['upstream_age_ms']
            return self.frame['jpeg'], self.frame['sequence'], max(0., age)

    def result_packet(self):
        with self.lock:
            if not self.enabled or not self.fresh(self.frame) or self.result is None:
                return None
            row = dict(self.result)
            age = (self.clock()-row['camera_received_monotonic_ms']/1000)*1000 + row['upstream_age_ms']
            if age >= 2000:
                return None
            row['result_age_ms'] = max(0., age)
            return row

    @contextmanager
    def control_request(self):
        with self.lock:self.control_pending+=1
        try:yield
        finally:
            with self.lock:self.control_pending-=1

    def infer_once(self):
        import cv2
        try:
            from .yoloe_inference import annotate
        except ImportError:
            from yoloe_inference import annotate
        with self.inference_lock:
            with self.lock:
                packet, mode, generation = self.frame, self.mode, self.generation
                if self.control_pending or not self.enabled or not self.fresh(packet):
                    return False
                identity = (packet['identity'], mode, generation)
                if identity == self.previous:
                    return False
            result = self.detector.predict(packet['bgr'], prompt_mode=mode)
            ok, overlay = cv2.imencode('.jpg', annotate(packet['bgr'], result))
            if not ok:
                raise RuntimeError('Cannot encode live overlay')
        result.update(input_kind='live_camera', frame_id=packet['sequence'],
            source_stream_id=packet['identity'][0], source_frame_id=packet['identity'][1],
            camera_received_monotonic_ms=packet['received']*1000,
            upstream_age_ms=packet['upstream_age_ms'],
            image_sha256=hashlib.sha256(packet['jpeg']).hexdigest(),
            overlay_base64=base64.b64encode(overlay.tobytes()).decode('ascii'))
        with self.lock:
            self.previous = identity
            if (not self.enabled or generation != self.generation or not self.fresh(packet) or
                not self.fresh(self.frame) or self.frame['identity'][0] != packet['identity'][0]):
                return False
            self.result = result
            self.inference_error = None
            self.completed.append(self.clock())
        return True

    def _camera(self):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while not self.done.is_set():
            try:
                with opener.open(self.camera_url, timeout=1.5) as reply:
                    stream, frame = reply.headers.get('X-Stream-Id'), reply.headers.get('X-Frame-Id')
                    age = float(reply.headers['X-Frame-Age-Ms'])
                    jpeg = reply.read(8500001)
                self.accept_frame(jpeg, stream, frame, age)
            except Exception as exc:
                self.disconnected('相机连接中断：' + type(exc).__name__)
            self.done.wait(.08)

    def _infer(self):
        while not self.done.is_set():
            try:
                if not self.infer_once():
                    self.done.wait(.03)
            except Exception as exc:
                with self.lock:
                    self.result = None
                    self.inference_error = type(exc).__name__ + ': ' + str(exc)[:180]
                self.done.wait(.5)
