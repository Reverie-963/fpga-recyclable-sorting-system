"""Bounded latest-frame camera/inference workers; stale frames never become live."""
import base64
import hashlib
import threading
import time
import urllib.parse
import urllib.request


class LivePreview:
    def __init__(self, detector, inference_lock, camera_url):
        parts = urllib.parse.urlsplit(camera_url)
        if parts.scheme != 'http' or parts.hostname not in ('127.0.0.1', 'localhost'):
            raise ValueError('Camera source must be the local camera-only bridge')
        self.detector, self.inference_lock, self.camera_url = detector, inference_lock, camera_url
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.frame = None
        self.result = None
        self.sequence = 0
        self.enabled = True
        self.error = '等待相机'
        self.inference_error = None
        self.threads = []

    def start(self):
        for target in (self._camera, self._infer):
            thread = threading.Thread(target=target, daemon=True)
            thread.start(); self.threads.append(thread)

    def close(self):
        self.done.set()
        for thread in self.threads:
            thread.join(timeout=4)

    def set_enabled(self, enabled):
        with self.lock:
            self.enabled = enabled

    def status(self):
        with self.lock:
            connected = self.frame is not None and time.monotonic() - self.frame['received'] < 2
            return dict(camera_connected=connected, camera_error=self.error,
                        live_enabled=self.enabled, live_available=True,
                        camera_frame_id=self.frame['sequence'] if connected else None,
                        inference_error=self.inference_error)

    def camera_packet(self):
        with self.lock:
            if self.frame is None or time.monotonic() - self.frame['received'] >= 2:
                return None
            return self.frame['jpeg'], self.frame['sequence']

    def result_packet(self):
        with self.lock:
            if (self.frame is None or time.monotonic() - self.frame['received'] >= 2 or
                    self.result is None):
                return None
            row = dict(self.result)
            row['result_age_ms'] = (time.monotonic() - row['camera_received_monotonic_ms']/1000) * 1000
            return row

    def _camera(self):
        import cv2
        import numpy as np
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        previous = None
        while not self.done.is_set():
            try:
                with opener.open(self.camera_url, timeout=3) as reply:
                    identity = (reply.headers['X-Stream-Id'], reply.headers['X-Frame-Id'])
                    age = float(reply.headers['X-Frame-Age-Ms'])
                    jpeg = reply.read(8500001)
                if not all(identity) or not 0 <= age < 2000 or len(jpeg) > 8500000:
                    raise ValueError('Invalid camera frame metadata')
                if identity != previous:
                    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                    if frame is None or frame.shape[0] > 1080 or frame.shape[1] > 1920:
                        raise ValueError('Invalid camera frame')
                    with self.lock:
                        self.sequence += 1
                        self.frame = dict(jpeg=jpeg, frame=frame, received=time.monotonic(),
                                          sequence=self.sequence, source_frame_id=identity[1])
                        self.error = None
                    previous = identity
            except Exception as exc:
                with self.lock:
                    self.frame = None
                    self.result = None
                    self.error = '相机连接中断：' + type(exc).__name__
            self.done.wait(.15)

    def _infer(self):
        import cv2
        from taco_inference import annotate
        previous = -1
        while not self.done.is_set():
            with self.lock:
                packet = self.frame
                enabled = self.enabled
            if (not enabled or packet is None or packet['sequence'] == previous or
                    time.monotonic() - packet['received'] >= 2):
                self.done.wait(.1); continue
            try:
                with self.inference_lock:
                    row = self.detector.predict(packet['frame'])
                    ok, overlay = cv2.imencode('.jpg', annotate(packet['frame'], row))
                if not ok:
                    raise RuntimeError('Cannot encode live overlay')
                row.update(input_kind='live_camera', frame_id=packet['sequence'],
                           source_frame_id=packet['source_frame_id'],
                           camera_received_monotonic_ms=packet['received']*1000,
                           image_sha256=hashlib.sha256(packet['jpeg']).hexdigest(),
                           overlay_base64=base64.b64encode(overlay.tobytes()).decode('ascii'))
                with self.lock:
                    if self.enabled and self.frame is not None:
                        self.result = row
                        self.inference_error = None
                previous = packet['sequence']
            except Exception as exc:
                with self.lock:
                    self.result = None
                    self.inference_error = type(exc).__name__
                self.done.wait(1)
