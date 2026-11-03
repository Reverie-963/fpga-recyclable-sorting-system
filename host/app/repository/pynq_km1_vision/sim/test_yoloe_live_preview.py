"""Live freshness, latest-frame and mode-change regressions; no devices required."""
import threading
from types import SimpleNamespace
import unittest
import cv2
import numpy as np
from src.yoloe_live_preview import LivePreview


class LiveTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.
        self.calls = []
        self.callback = None

        def predict(frame, prompt_mode=None):
            self.calls.append(prompt_mode)
            if self.callback:
                self.callback()
            return dict(instances=[], image_size=[4, 3], prompt_mode=prompt_mode,
                        inference_ms=10, automatic_motion_allowed=False)

        self.detector = SimpleNamespace(predict=predict, default_prompt_mode='visual',
                                        models={'visual': None, 'text': None})
        self.live = LivePreview(self.detector, threading.Lock(),
                                'http://127.0.0.1:8780/frame.jpg', clock=lambda: self.now)
        ok, jpeg = cv2.imencode('.jpg', np.zeros((3, 4, 3), np.uint8))
        self.assertTrue(ok)
        self.jpeg = jpeg.tobytes()

    def accept(self, frame_id='1', stream='a', age=0):
        self.live.accept_frame(self.jpeg, stream, frame_id, age)

    def test_control_priority_pauses_only_background_inference_and_releases_on_error(self):
        self.accept()
        with self.assertRaises(ValueError):
            with self.live.control_request():
                self.assertFalse(self.live.infer_once())
                self.assertIsNotNone(self.live.camera_packet())
                raise ValueError('control failed')
        self.assertEqual(self.live.control_pending,0)
        self.assertTrue(self.live.infer_once())

    def test_duplicate_poll_does_not_make_old_image_fresh(self):
        self.accept()
        self.now += 2.1
        self.accept()
        self.assertFalse(self.live.status()['camera_connected'])
        self.assertIsNone(self.live.camera_packet())
        self.assertFalse(self.live.infer_once())

    def test_upstream_age_is_part_of_freshness_limit(self):
        self.accept(age=1800)
        self.now += .3
        self.assertIsNone(self.live.camera_packet())
        self.assertIsNone(self.live.result_packet())

    def test_only_latest_frame_is_inferred_and_no_repeated_inference(self):
        for frame in ('1', '2', '3'):
            self.accept(frame)
        self.assertTrue(self.live.infer_once())
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.live.result_packet()['source_frame_id'], '3')
        self.assertFalse(self.live.infer_once())

    def test_old_result_expires_even_when_camera_keeps_updating(self):
        self.accept()
        self.assertTrue(self.live.infer_once())
        self.now += 2.1
        self.accept('2')
        self.assertTrue(self.live.status()['camera_connected'])
        self.assertIsNone(self.live.result_packet())

    def test_pause_during_inference_discards_inflight_result(self):
        self.accept()
        self.callback = lambda: self.live.configure(False, 'visual')
        self.assertFalse(self.live.infer_once())
        self.assertIsNone(self.live.result_packet())
        self.assertIsNotNone(self.live.camera_packet())

    def test_mode_change_during_inference_discards_old_mode(self):
        self.accept()
        self.callback = lambda: self.live.configure(True, 'text')
        self.assertFalse(self.live.infer_once())
        self.callback = None
        self.assertTrue(self.live.infer_once())
        self.assertEqual(self.live.result_packet()['prompt_mode'], 'text')

    def test_disconnect_and_stream_restart_discard_inflight_result(self):
        self.accept()

        def restart():
            self.live.disconnected('test_disconnect')
            self.accept('1', stream='b')

        self.callback = restart
        self.assertFalse(self.live.infer_once())
        self.assertIsNone(self.live.result_packet())
        self.callback = None
        self.assertTrue(self.live.infer_once())
        self.assertEqual(self.live.result_packet()['source_stream_id'], 'b')

    def test_invalid_controls_and_camera_urls_rejected(self):
        for enabled, mode in ((1, 'visual'), (True, 'npu')):
            with self.assertRaises(ValueError):
                self.live.configure(enabled, mode)
        for url in ('http://example.com/frame.jpg', 'https://localhost/frame.jpg',
                    'http://user:password@localhost/frame.jpg', 'http://localhost/frame.jpg?url=other'):
            with self.assertRaises(ValueError):
                LivePreview(self.detector, threading.Lock(), url)

    def test_invalid_age_does_not_create_camera_frame(self):
        for age in (-1, float('nan'), 2000):
            with self.assertRaises(ValueError):
                self.accept(age=age)
        self.assertIsNone(self.live.frame)


if __name__ == '__main__':
    unittest.main()
