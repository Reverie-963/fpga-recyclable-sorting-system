"""Local fake upstream only: no GPU, camera, board, or serial is accessed."""
import base64
import copy
import hashlib
import json
import threading
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

import cv2
import numpy as np

from src.board_detector_transport import (CONTRACT, RelayDetector, RelayTransport,
                                          encoded, make_server)


class RelayTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.health = dict(model_loaded=True, model_family='yolov8s',
            model_versions=dict(supervised='frozen-model'), runtime_id='frozen-runtime',
            execution=dict(device='cuda:0', precision='fp16'))
        self.mutate = lambda v: v
        self.server = make_server(0, 't'*40, upstream_request=self.upstream)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close)
        self.url = 'http://127.0.0.1:'+str(self.server.server_port)
        self.config = dict(detector_url=self.url, frame_timeout_s=2, model_family='yolov8s',
            prompt_mode='supervised', device='cuda:0', half=True, model_version='frozen-model',
            runtime_id='frozen-runtime')

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)

    def upstream(self, route, data):
        self.calls.append((route, data))
        if route == 'health':
            return copy.deepcopy(self.health)
        raw = base64.b64decode(data['image_base64'])
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        return self.mutate(dict(image_sha256=hashlib.sha256(raw).hexdigest(),
            model_version='frozen-model', runtime_id='frozen-runtime',
            prompt_mode='supervised', image_size=list(image.shape[1::-1]), instances=[]))

    def test_exact_png_and_original_detector_guards_preserved(self):
        detector = RelayDetector(self.config, 't'*40)
        image = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
        result = detector.predict(image, 'supervised')
        sent = self.calls[-1][1]
        raw = base64.b64decode(sent['image_base64'])
        self.assertTrue(np.array_equal(cv2.imdecode(np.frombuffer(raw, np.uint8), 1), image))
        self.assertEqual(result['image_sha256'], hashlib.sha256(raw).hexdigest())
        self.assertEqual([r for r, _ in self.calls], ['health', 'detect'])

    def test_wrong_token_does_not_call_upstream(self):
        with self.assertRaises(urllib.error.HTTPError) as error:
            RelayTransport(self.url, 'w'*40, 2).request('health')
        self.assertEqual(error.exception.code, 403)
        self.assertEqual(self.calls, [])

    def test_changed_model_runtime_pixels_prompt_or_size_rejected_once(self):
        detector = RelayDetector(self.config, 't'*40)
        for key, value in [('model_version','new-model'), ('runtime_id','new-runtime'),
                           ('image_sha256','0'*64), ('prompt_mode','text'), ('image_size',[1, 1])]:
            with self.subTest(key=key):
                self.mutate = lambda v, k=key, x=value: dict(v, **{k:x})
                before = len(self.calls)
                with self.assertRaises(ValueError):
                    detector.predict(np.zeros((4, 5, 3), np.uint8), 'supervised')
                self.assertEqual(len(self.calls), before+1)

    def test_changed_health_model_family_rejected(self):
        self.health['model_family'] = 'yolo11s'
        with self.assertRaisesRegex(ValueError, 'deployed_model_family_mismatch'):
            RelayDetector(self.config, 't'*40)

    def test_unknown_or_motion_routes_rejected_before_upstream(self):
        relay = RelayTransport(self.url, 't'*40, 2)
        for route in ('execute', 'stop', 'baseline', 'regions', 'frame.jpg', '../health'):
            with self.subTest(route=route), self.assertRaises(ValueError):
                relay.request(route)
        self.assertEqual(self.calls, [])

    def test_truncated_or_non_png_input_rejected_before_upstream(self):
        for raw in (b'', b'\xff\xd8jpeg', b'not a png'):
            with self.assertRaises(ValueError):
                RelayTransport(self.url, 't'*40, 2).request('detect', dict(
                    image_base64=base64.b64encode(raw).decode(), prompt_mode='supervised'))
        self.assertEqual(self.calls, [])

    def test_reordered_or_corrupt_response_rejected(self):
        relay = RelayTransport(self.url, 't'*40, 2)
        class Response:
            def __init__(self, value): self.value = value
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, size): return encoded(self.value)
        for changes in ({'request_id':'0'*32}, {'route':'detect'}, {'payload_sha256':'0'*64}):
            def reply(call, timeout, changes=changes):
                sent = json.loads(call.data)
                payload = dict(ok=True)
                result = dict(contract=CONTRACT, request_id=sent['request_id'], route='health',
                              payload=payload, payload_sha256=hashlib.sha256(encoded(payload)).hexdigest())
                return Response(dict(result, **changes))
            with self.subTest(changes=changes), patch.object(relay.opener, 'open', side_effect=reply):
                with self.assertRaisesRegex(ValueError, 'detector_exchange_identity_mismatch'):
                    relay.request('health')

    def test_cross_host_time_not_used_and_late_reply_rejected(self):
        relay = RelayTransport(self.url, 't'*40, .1, clock=iter([1000, 1001]).__next__)
        with self.assertRaisesRegex(TimeoutError, 'detector_relay_deadline_exceeded'):
            relay.request('health')

    def test_remote_host_or_invalid_timeout_cannot_be_selected(self):
        for url in ('http://192.168.2.99:8778', self.url+'/detect', 'https://localhost:8778'):
            with self.assertRaises(ValueError): RelayTransport(url, 't'*40, 2)
        for timeout in (0, True, 31, float('nan')):
            with self.assertRaises(ValueError): RelayTransport(self.url, 't'*40, timeout)

    def test_busy_relay_rejects_without_queueing(self):
        entered, release = threading.Event(), threading.Event()
        # The callback itself is replaceable without changing the listener.
        before = self.upstream
        errors = []
        results = []
        def blocked(route, data):
            entered.set()
            if not release.wait(2): raise TimeoutError('test_release_timeout')
            return before(route, data)
        # make_server captured the original bound callback, whose method reads
        # self.health. Hold an explicit server using the blocking callback.
        server = make_server(0, 't'*40, upstream_request=blocked)
        serving = threading.Thread(target=server.serve_forever, daemon=True)
        serving.start()
        relay = RelayTransport('http://127.0.0.1:'+str(server.server_port), 't'*40, 2)
        def first():
            try: results.append(relay.request('health'))
            except Exception as exc: errors.append(exc)
        worker = threading.Thread(target=first)
        try:
            worker.start()
            self.assertTrue(entered.wait(1))
            with self.assertRaises(urllib.error.HTTPError) as error:
                relay.request('health')
            self.assertEqual(error.exception.code, 409)
        finally:
            release.set()
            worker.join(3)
            server.shutdown()
            server.server_close()
            serving.join(2)
        self.assertFalse(errors)
        self.assertEqual(len(results), 1)
        self.assertEqual(len(self.calls), 1)

    def test_upstream_failure_is_one_attempt(self):
        self.mutate = lambda v: (_ for _ in ()).throw(OSError('offline upstream failure'))
        detector = RelayDetector(self.config, 't'*40)
        with self.assertRaises(urllib.error.HTTPError) as error:
            detector.predict(np.zeros((4, 5, 3), np.uint8), 'supervised')
        self.assertEqual(error.exception.code, 502)
        self.assertEqual([route for route, _ in self.calls], ['health', 'detect'])

    def test_server_validates_routes_even_when_client_checks_are_bypassed(self):
        invalid = dict(contract=CONTRACT, request_id='a'*32, route='execute', payload=None)
        request = urllib.request.Request(self.url+'/exchange', data=encoded(invalid), headers={
            'Authorization':'Bearer '+'t'*40, 'Content-Type':'application/json'})
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=2)
        self.assertEqual(error.exception.code, 400)
        self.assertEqual(self.calls, [])

    def test_explicit_three_frame_concurrency_is_bounded_without_queue(self):
        release,entered=threading.Event(),threading.Event()
        lock=threading.Lock();count=[0]
        def blocked(route,data):
            with lock:
                count[0]+=1
                if count[0]==3:entered.set()
            if not release.wait(2):raise TimeoutError('test timeout')
            return dict(ready=True)
        server=make_server(0,'t'*40,upstream_request=blocked,max_inflight=3)
        serving=threading.Thread(target=server.serve_forever,daemon=True);serving.start()
        relay=RelayTransport('http://127.0.0.1:'+str(server.server_port),'t'*40,2)
        results=[];errors=[]
        def request():
            try:results.append(relay.request('health'))
            except Exception as exc:errors.append(exc)
        workers=[threading.Thread(target=request) for _ in range(3)]
        try:
            for worker in workers:worker.start()
            self.assertTrue(entered.wait(1))
            with self.assertRaises(urllib.error.HTTPError) as error:relay.request('health')
            self.assertEqual(error.exception.code,409)
        finally:
            release.set()
            for worker in workers:worker.join(3)
            server.shutdown();server.server_close();serving.join(2)
        self.assertFalse(errors);self.assertEqual(len(results),3);self.assertEqual(count[0],3)


if __name__ == '__main__':
    unittest.main()
