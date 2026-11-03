"""Protect material mapping, original-image mask coordinates and preview-only HTTP behavior."""
import base64
import io
import json
from pathlib import Path
import tempfile
import threading
import socket
from types import SimpleNamespace
import unittest
import urllib.error
import urllib.request
import numpy as np
try:
    from PIL import Image
except ImportError:
    Image = None
from src.static_mask_contract import decode_mask
from src.yoloe_inference import convert_instances, read_config, YoloeDetector
from src.yoloe_preview_server import make_server


class ArrayTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def __len__(self):
        return len(self.value)


class FakeBoxes:
    def __init__(self, classes):
        self.cls = np.array(classes)
        self.conf = np.full(len(classes), .8)
        self.xyxy = np.tile([1, 0, 3, 2], (len(classes), 1))

    def __len__(self):
        return len(self.cls)


def raw_result(classes, masks):
    return SimpleNamespace(boxes=FakeBoxes(classes),
                           masks=SimpleNamespace(data=ArrayTensor(masks)))


class YoloeContractTests(unittest.TestCase):
    def test_unsupported_device_and_cpu_half_are_not_silently_accepted(self):
        for device, half in [('npu', False), ('cuda:3', False), ('cpu', True)]:
            with self.assertRaises(ValueError):
                YoloeDetector(device=device, half=half)

    def test_concrete_prompts_map_to_exactly_three_materials(self):
        cfg = read_config()
        masks = np.ones((len(cfg['prompts']), 2, 3), np.uint8)
        result = convert_instances(raw_result(list(range(len(masks))), masks), cfg, (2, 3))
        self.assertEqual([x['label'] for x in result],
                         ['plastic', 'plastic', 'plastic', 'metal', 'paper', 'paper', 'paper'])
        self.assertEqual([x['class_id'] for x in result], [0, 0, 0, 1, 2, 2, 2])
        self.assertEqual({x['display_name'] for x in result}, {'塑料', '金属', '纸质'})
        self.assertTrue(all(x['sorting_bin'] is None for x in result))

    def test_non_square_mask_preserves_position_area_and_rle(self):
        mask = np.array([[0, 1, 1, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]], np.uint8)
        item = convert_instances(raw_result([4], [mask]), read_config(), (3, 5))[0]
        self.assertEqual(item['bbox_xyxy'], [1, 0, 3, 2])
        self.assertEqual(item['area_px'], 3)
        self.assertEqual(decode_mask(item['mask_rle'], [5, 3]), mask.ravel().tolist())

    def test_letterboxed_mask_is_rejected_instead_of_stretched(self):
        with self.assertRaisesRegex(ValueError, 'original image'):
            convert_instances(raw_result([0], np.ones((1, 5, 5))), read_config(), (3, 5))

    def test_unknown_prompt_is_rejected_instead_of_relabelled(self):
        for class_id in (-1, 7, .5):
            with self.assertRaisesRegex(ValueError, 'class ID'):
                convert_instances(raw_result([class_id], np.ones((1, 2, 3))), read_config(), (2, 3))

    def test_boxes_without_masks_are_rejected(self):
        raw = raw_result([0], np.ones((1, 2, 3)))
        raw.masks = None
        with self.assertRaisesRegex(ValueError, 'masks are missing'):
            convert_instances(raw, read_config(), (2, 3))


@unittest.skipIf(Image is None, 'YOLOE preview optional Pillow dependency is not installed')
class PreviewHttpTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.calls = 0
        cfg = read_config()

        def predict(frame, prompt_mode=None):
            self.calls += 1
            return dict(image_size=[frame.shape[1], frame.shape[0]], instances=[],
                        automatic_motion_allowed=False, quality_gate_passed=False)

        detector = SimpleNamespace(predict=predict, health=lambda: dict(model_loaded=True,
            categories=cfg['categories'], camera_connected=False, automatic_motion_allowed=False))
        self.server = make_server(detector, 0, Path(self.temp.name))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = 'http://127.0.0.1:' + str(self.server.server_port)

    def tearDown(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()
        self.temp.cleanup()

    def post(self, payload, headers=None):
        request = urllib.request.Request(self.base + '/detect',
            data=json.dumps(payload).encode(), headers=headers or {'Content-Type': 'application/json'})
        return urllib.request.urlopen(request, timeout=5)

    def test_invalid_payloads_do_not_reach_detector(self):
        for payload in ([], {'image_base64': 17}, {'image_base64': 'not base64'}, {}):
            with self.assertRaises(urllib.error.HTTPError) as raised:
                self.post(payload)
            self.assertEqual(raised.exception.code, 400)
            raised.exception.close()
        self.assertEqual(self.calls, 0)

    def test_foreign_origin_is_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as raised:
            self.post({}, {'Origin': 'https://example.com'})
        self.assertEqual(raised.exception.code, 403)
        raised.exception.close()
        self.assertEqual(self.calls, 0)

    def test_idle_browser_connections_do_not_block_health(self):
        idle = [socket.create_connection(('127.0.0.1', self.server.server_port), timeout=2)
                for _ in range(4)]
        try:
            with urllib.request.urlopen(self.base + '/health', timeout=2) as response:
                self.assertTrue(json.load(response)['model_loaded'])
        finally:
            for connection in idle:
                connection.close()

    def test_second_server_cannot_reuse_the_active_port(self):
        with self.assertRaises(OSError):
            make_server(SimpleNamespace(), self.server.server_port, Path(self.temp.name))

    def test_oversize_image_rejected_before_inference(self):
        stream = io.BytesIO()
        Image.new('RGB', (1921, 1)).save(stream, format='PNG')
        with self.assertRaises(urllib.error.HTTPError) as raised:
            self.post({'image_base64': base64.b64encode(stream.getvalue()).decode()})
        self.assertEqual(raised.exception.code, 400)
        raised.exception.close()
        self.assertEqual(self.calls, 0)

    def test_success_persists_separate_evidence_and_keeps_motion_disabled(self):
        stream = io.BytesIO()
        Image.new('RGB', (5, 3)).save(stream, format='PNG')
        for _ in range(2):
            with self.post({'image_base64': base64.b64encode(stream.getvalue()).decode()}) as response:
                result = json.load(response)
            self.assertFalse(result['automatic_motion_allowed'])
            self.assertFalse(result['quality_gate_passed'])
        self.assertEqual(result['image_size'], [5, 3])
        self.assertEqual(result['frame_id'], 2)
        self.assertEqual(len(list(Path(self.temp.name).glob('*/result.json'))), 2)


if __name__ == '__main__':
    unittest.main()
