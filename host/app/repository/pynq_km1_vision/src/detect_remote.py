"""Full-frame v2 client; never compare clocks from different machines."""
import base64
import json
import os
import time
import uuid
from urllib.request import Request, build_opener, ProxyHandler
from urllib.parse import urlparse
import cv2
from .detection import validate_detections

class DetectionClient:
    def __init__(self, cfg):
        self.cfg = cfg; self.session = uuid.uuid4().hex; self.model_id = None; self.runtime_id = None
        self.token = os.environ.get(cfg.get('token_env', 'KM1_TOKEN'), '')
        if len(self.token) < 24: raise ValueError('Missing KM1_TOKEN')
        url = urlparse(cfg['url'])
        if url.scheme not in ('http', 'https') or not url.hostname or url.username: raise ValueError('Invalid URL')
        self.opener = build_opener(ProxyHandler({}))
    def predict(self, frame, fid, captured_at):
        ok, image = cv2.imencode('.png', frame)
        if not ok: raise ValueError('PNG encode failed')
        rid = self.session + ':' + str(fid) + ':' + uuid.uuid4().hex[:8]
        body = json.dumps({'protocol': 2, 'request_id': rid, 'frame_id': fid,
                           'png': base64.b64encode(image).decode()}).encode()
        if len(body) > 8_500_000: raise ValueError('Request too large')
        request = Request(self.cfg['url'].rstrip('/') + '/detect', body,
            {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.token})
        t = time.monotonic()
        with self.opener.open(request, timeout=self.cfg['timeout_s']) as response: raw = response.read(131073)
        if len(raw) > 131072: raise ValueError('Response too large')
        result = json.loads(raw); now = time.monotonic()
        if now - captured_at > self.cfg['max_result_age_s']: raise TimeoutError('Stale detection')
        if result.get('protocol') != 2 or result.get('frame_id') != fid or result.get('request_id') != rid:
            raise ValueError('Mismatched detection response')
        for field in ('model_id', 'runtime_id'):
            value = result.get(field)
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value): raise ValueError('Invalid fingerprint')
            expected = self.cfg.get('expected_' + field) or getattr(self, field)
            if expected and expected != value: raise ValueError('Changed ' + field)
            setattr(self, field, value)
        h, w = frame.shape[:2]
        if result.get('image_size') != [w, h]: raise ValueError('Source dimensions mismatch')
        result['detections'] = validate_detections(result.get('detections'), w, h)
        result.update(rpc_ms=(now-t)*1000, request_bytes=len(body))
        return result
