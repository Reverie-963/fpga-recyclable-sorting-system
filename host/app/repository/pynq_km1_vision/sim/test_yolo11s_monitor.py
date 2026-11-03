"""Model-switch contracts: legacy IDs, fixed modes and live HTTP integration."""
import base64
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request
import cv2
import numpy as np
from sim.test_yoloe_inference import raw_result
from src.yolo11s_inference import convert_materials, Yolo11sDetector
from src.yoloe_preview_server import make_server


class MaterialSwitchTests(unittest.TestCase):
    def test_legacy_classes_map_by_material_not_numeric_position(self):
        items = convert_materials(raw_result([0, 1, 2], np.ones((3, 3, 5))), (3, 5))
        self.assertEqual([i['label'] for i in items], ['metal', 'plastic', 'paper'])
        self.assertEqual([i['class_id'] for i in items], [1, 0, 2])
        self.assertEqual([i['display_name'] for i in items], ['金属', '塑料', '纸质'])
        self.assertTrue(all(i['sorting_bin'] is None for i in items))

    def test_no_implicit_cpu_or_other_accelerator(self):
        for device in ('cpu', 'npu', 'cuda:1'):
            with self.assertRaises(ValueError):
                Yolo11sDetector(device=device)

    def test_previous_prompt_modes_rejected_before_inference(self):
        detector = Yolo11sDetector.__new__(Yolo11sDetector)
        for mode in ('visual', 'text'):
            with self.assertRaises(ValueError):
                detector.predict(np.zeros((3, 5, 3), np.uint8), mode)

    def test_old_physical_loop_rejects_replaced_service_at_startup(self):
        from src.sorting_io import HTTPYoloe
        health = dict(model_loaded=True, model_family='yolo11s', model_versions={'supervised':'hash'})
        with patch.object(HTTPYoloe, 'request', return_value=health):
            with self.assertRaisesRegex(ValueError, 'detector_service_is_not_yoloe'):
                HTTPYoloe(dict(detector_url='http://127.0.0.1:8778'), 'unused')

    def test_http_and_live_share_fixed_supervised_mode(self):
        calls = []

        def predict(frame, prompt_mode=None):
            calls.append(prompt_mode)
            return dict(instances=[], image_size=[5, 3], prompt_mode='supervised',
                        inference_ms=1, automatic_motion_allowed=False)

        detector = SimpleNamespace(models={'supervised': None}, default_prompt_mode='supervised',
            predict=predict, health=lambda:dict(model_name='YOLO11s-seg', model_loaded=True))
        with tempfile.TemporaryDirectory() as folder:
            server = make_server(detector, 0, Path(folder), camera_url='http://127.0.0.1:8780/frame.jpg')
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = 'http://127.0.0.1:' + str(server.server_port)
            try:
                with urllib.request.urlopen(base+'/benchmark', timeout=3) as reply:
                    self.assertFalse(json.load(reply)['available'])
                ok, png = cv2.imencode('.png', np.zeros((3, 5, 3), np.uint8))
                self.assertTrue(ok)
                payload = dict(image_base64=base64.b64encode(png).decode(), prompt_mode='supervised')
                def post(path, data):
                    return urllib.request.urlopen(urllib.request.Request(base+path,
                        data=json.dumps(data).encode(), headers={'Content-Type':'application/json'}), timeout=3)
                with post('/detect', payload) as reply:
                    self.assertFalse(json.load(reply)['automatic_motion_allowed'])
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    post('/detect', dict(payload, prompt_mode='visual'))
                self.assertEqual(raised.exception.code, 400)
                raised.exception.close()
                with post('/live', dict(enabled=True, prompt_mode='supervised')) as reply:
                    self.assertEqual(json.load(reply)['live_prompt_mode'], 'supervised')
                server.live.accept_frame(png.tobytes(), 'test', '1', 0)
                self.assertTrue(server.live.infer_once())
                self.assertEqual(server.live.result_packet()['prompt_mode'], 'supervised')
                self.assertEqual(calls, ['supervised', 'supervised'])
            finally:
                server.shutdown()
                thread.join()
                server.server_close()


if __name__ == '__main__':
    unittest.main()
