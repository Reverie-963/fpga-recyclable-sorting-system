import base64
import json
import os
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np

from src.camera_observe import run
from src.detect_remote import DetectionClient
from src.detect_server import make_server
from src.vision_monitor import MonitorState, make_monitor_server

TOKEN = 'observation_test_01234567890123456789'


class Detector:
    model_id = 'a' * 64
    runtime_id = 'b' * 64
    providers = ['simulation']
    labels = ['cap_deformed']
    fail = False

    def predict(self, frame):
        if self.fail:
            raise RuntimeError('Simulated inference error')
        return {'model_id': self.model_id, 'runtime_id': self.runtime_id,
                'providers': self.providers, 'image_size': [frame.shape[1], frame.shape[0]],
                'detections': [{'label': 'cap_deformed', 'confidence': .9, 'xyxy': [10, 12, 40, 44]}],
                'timing': {'total_ms': 5}}


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.state = MonitorState()
        self.addCleanup(self.state.close)
        self.frame = np.full((80, 120, 3), (20, 60, 180), np.uint8)
        self.detector = Detector()

    def wait_snapshot(self, sequence=1):
        deadline = time.monotonic() + 3
        with self.state.condition:
            while self.state.sequence < sequence:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.fail('Monitor encoding did not finish')
                self.state.condition.wait(remaining)
        return self.state.snapshot()

    def result(self, fid=1):
        return {**self.detector.predict(self.frame), 'frame_id': fid, 'request_id': 'test-request-' + str(fid)}

    def start_server(self, server):
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        def close():
            server.shutdown()
            server.server_close()
            thread.join(3)
        self.addCleanup(close)
        return 'http://127.0.0.1:' + str(server.server_port)

    def test_frame_result_pair_copy_and_age(self):
        result = self.result(23)
        self.state.publish(self.frame, result, time.monotonic() - 4)
        self.frame[:] = 0
        result['detections'].clear()
        packet = self.wait_snapshot()
        self.assertEqual(packet['frame_id'], 23)
        self.assertEqual(len(packet['detections']), 1)
        self.assertGreaterEqual(packet['age_s'], 4)
        jpeg = base64.b64decode(packet['image'].split(',')[1])
        decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape, (80, 120, 3))
        self.assertGreater(float(decoded.mean()), 50)

    def test_empty_detection_clears_previous_boxes(self):
        self.state.publish(self.frame, self.result(), time.monotonic())
        self.wait_snapshot()
        empty = self.result(2)
        empty['detections'] = []
        self.state.publish(self.frame, empty, time.monotonic())
        packet = self.wait_snapshot(2)
        self.assertEqual(packet['status'], 'ok')
        self.assertEqual(packet['detections'], [])
        self.assertEqual(packet['frame_id'], 2)

    def test_error_and_dimension_mismatch_clear_boxes(self):
        self.state.publish(self.frame, {'frame_id': 2, 'request_id': 'failure', 'error': 'inference_failed'}, time.monotonic())
        packet = self.wait_snapshot()
        self.assertEqual(packet['status'], 'inference_failed')
        self.assertEqual(packet['detections'], [])
        self.assertIn('image', packet)
        wrong = self.result()
        wrong['image_size'] = [1, 2]
        self.state.publish(self.frame, wrong, time.monotonic())
        packet = self.wait_snapshot(2)
        self.assertEqual(packet['status'], 'preview_failed')
        self.assertNotIn('image', packet)

    def test_waiting_local_ui_host_restriction_and_no_control_endpoint(self):
        ui = make_monitor_server(0, self.state)
        self.assertEqual(ui.server_address[0], '127.0.0.1')
        url = self.start_server(ui)
        with urlopen(url + '/snapshot') as response:
            self.assertEqual(json.load(response)['status'], 'waiting')
            self.assertEqual(response.headers['Cache-Control'], 'no-store')
            self.assertIsNone(response.headers.get('Access-Control-Allow-Origin'))
        for path in ('/', '/monitor.js', '/monitor.css'):
            with urlopen(url + path) as response:
                self.assertEqual(response.status, 200)
        for path in ('/execute', '/../config.json'):
            with self.assertRaises(HTTPError) as err:
                urlopen(url + path)
            self.assertEqual(err.exception.code, 404)
        with self.assertRaises(HTTPError) as err:
            urlopen(Request(url + '/snapshot', headers={'Host': 'untrusted.example'}))
        self.assertEqual(err.exception.code, 403)

    def test_detection_integration_auth_duplicate_and_error(self):
        url = self.start_server(make_server(('127.0.0.1', 0), self.detector, TOKEN, self.state))
        cfg = {'url': url, 'timeout_s': 1, 'max_result_age_s': 3}
        with patch.dict(os.environ, KM1_TOKEN=TOKEN):
            client = DetectionClient(cfg)
        result = client.predict(self.frame, 15, time.monotonic())
        packet = self.wait_snapshot()
        self.assertEqual(packet['frame_id'], 15)
        self.assertEqual(packet['detections'], result['detections'])
        encoded = base64.b64encode(cv2.imencode('.png', self.frame)[1]).decode()
        body = json.dumps({'protocol': 2, 'request_id': packet['request_id'], 'frame_id': 15, 'png': encoded}).encode()
        with self.assertRaises(HTTPError) as err:
            urlopen(Request(url + '/detect', body, {'Authorization': 'Bearer ' + TOKEN}))
        self.assertEqual(err.exception.code, 409)
        with self.assertRaises(HTTPError) as err:
            urlopen(Request(url + '/detect', body))
        self.assertEqual(err.exception.code, 401)
        self.assertEqual(self.state.snapshot()['sequence'], 1)
        self.detector.fail = True
        with self.assertRaises(HTTPError) as err:
            client.predict(self.frame, 16, time.monotonic())
        self.assertEqual(err.exception.code, 500)
        packet = self.wait_snapshot(2)
        self.assertEqual(packet['status'], 'inference_failed')
        self.assertEqual(packet['frame_id'], 16)
        self.assertEqual(packet['detections'], [])

    def test_preview_failure_does_not_change_detection_response(self):
        class BrokenPreview:
            def publish(self, *args):
                raise RuntimeError('Preview unavailable')
        url = self.start_server(make_server(('127.0.0.1', 0), self.detector, TOKEN, BrokenPreview()))
        with patch.dict(os.environ, KM1_TOKEN=TOKEN):
            client = DetectionClient({'url': url, 'timeout_s': 1, 'max_result_age_s': 3})
        self.assertEqual(len(client.predict(self.frame, 1, time.monotonic())['detections']), 1)


class ObserveTests(unittest.TestCase):
    cfg = {'camera': {'index': 0, 'width': 120, 'height': 80},
           'inference': {'url': 'http://127.0.0.1:8765', 'timeout_s': 1, 'max_result_age_s': 2}}

    def test_limited_sender_retries_new_frames_and_closes_camera(self):
        with patch('src.camera_observe.LatestCamera') as camera_type, patch('src.camera_observe.DetectionClient') as client_type, patch('src.camera_observe.time.sleep'):
            camera = camera_type.return_value
            frame = np.zeros((80, 120, 3), np.uint8)
            camera.read.side_effect = [(1, 1, frame), (3, 2, frame)]
            client_type.return_value.predict.side_effect = [URLError('offline'), {}]
            self.assertEqual(run(self.cfg, max_frames=2), 2)
            self.assertEqual(camera.read.call_args_list[1].kwargs, {'after': 1})
            camera.close.assert_called_once()

    def test_identity_error_stops_and_closes_camera(self):
        with patch('src.camera_observe.LatestCamera') as camera_type, patch('src.camera_observe.DetectionClient') as client_type:
            camera_type.return_value.read.return_value = (1, 1, np.zeros((80, 120, 3), np.uint8))
            client_type.return_value.predict.side_effect = ValueError('Changed model_id')
            with self.assertRaisesRegex(ValueError, 'Changed model_id'):
                run(self.cfg, max_frames=1)
            camera_type.return_value.close.assert_called_once()

    def test_bad_token_stops_instead_of_retrying(self):
        with patch('src.camera_observe.LatestCamera') as camera_type, patch('src.camera_observe.DetectionClient') as client_type:
            camera_type.return_value.read.return_value = (1, 1, np.zeros((80, 120, 3), np.uint8))
            client_type.return_value.predict.side_effect = HTTPError('http://localhost/detect', 401, 'unauthorized', {}, None)
            with self.assertRaises(HTTPError):
                run(self.cfg, max_frames=1)
            camera_type.return_value.close.assert_called_once()

    def test_invalid_rate_rejected_before_camera_access(self):
        with patch('src.camera_observe.LatestCamera') as camera_type:
            for fps in (0, -1, 31, float('nan')):
                with self.assertRaises(ValueError):
                    run(self.cfg, fps=fps)
            camera_type.assert_not_called()


if __name__ == '__main__':
    unittest.main()
