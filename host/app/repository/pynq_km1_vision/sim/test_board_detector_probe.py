import base64
import hashlib
import json
import unittest

import cv2
import numpy as np

from src.board_detector_probe import inspect_frame


class ProbeFrameTests(unittest.TestCase):
    def setUp(self):
        self.image = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
        self.raw = cv2.imencode('.png', self.image)[1].tobytes()
        self.health = dict(stream_id='pinned-stream', assets={'bit':'a'*64}, config_sha256='b'*64,
                           backend='pl_board_local', baseline_frame_id=None)
        self.info = dict(self.health, frame_id=123, camera_connected=True, faulted=False,
                         frame_age_ms=50, response_prepare_ms=10)
        self.config = dict(image_size=[5, 4], max_age_ms=2000)

    def headers(self):
        return {'X-Station-Metadata':base64.b64encode(json.dumps(self.info).encode()).decode(),
                'X-Stream-Id':'pinned-stream', 'X-Frame-Id':'123',
                'X-Decoded-Bgr-Sha256':hashlib.sha256(self.image.tobytes()).hexdigest()}

    def test_same_pixels_and_conservative_local_capture_estimate(self):
        image, identity = inspect_frame(self.raw, self.headers(), self.health, self.config, 1000, 1100)
        self.assertTrue(np.array_equal(image, self.image))
        self.assertEqual(identity['capture_started_ms'], 960)
        self.assertEqual(identity['snapshot_age_upper_ms'], 140)

    def test_station_change_or_fault_rejected(self):
        for key, value in [('stream_id','new'), ('config_sha256','new'), ('baseline_frame_id',124),
                           ('assets',{}), ('backend','cpu_reference'), ('faulted',True)]:
            with self.subTest(key=key):
                original = self.info.copy()
                self.info[key] = value
                with self.assertRaises(ValueError):
                    inspect_frame(self.raw, self.headers(), self.health, self.config, 1000, 1100)
                self.info = original

    def test_expired_frame_and_fake_timing_rejected(self):
        for key, value in [('frame_age_ms',2100), ('frame_age_ms',float('nan')),
                           ('response_prepare_ms',1000), ('frame_id',True)]:
            with self.subTest(key=key):
                original = self.info.copy()
                self.info[key] = value
                with self.assertRaises(ValueError):
                    inspect_frame(self.raw, self.headers(), self.health, self.config, 1000, 1100)
                self.info = original
        with self.assertRaisesRegex(ValueError, 'probe_snapshot_deadline_exceeded'):
            inspect_frame(self.raw, self.headers(), self.health, self.config, 1000, 3100)

    def test_pixels_header_identity_and_size_cannot_be_substituted(self):
        for key, value in [('X-Decoded-Bgr-Sha256','0'*64), ('X-Frame-Id','124'), ('X-Stream-Id','new')]:
            headers = dict(self.headers(), **{key:value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                inspect_frame(self.raw, headers, self.health, self.config, 1000, 1100)
        with self.assertRaises(ValueError):
            inspect_frame(self.raw, self.headers(), self.health, dict(self.config,image_size=[1, 1]), 1000, 1100)


if __name__ == '__main__':
    unittest.main()
