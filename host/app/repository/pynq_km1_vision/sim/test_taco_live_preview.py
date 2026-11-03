"""Freshness and bounded-frame contracts, without a camera or legacy runtime."""
import threading
import time
import unittest
from src.taco_live_preview import LivePreview


class LivePreviewTests(unittest.TestCase):
    def make_live(self):
        return LivePreview(None, threading.Lock(), 'http://127.0.0.1:8780/frame.jpg')

    def test_disconnected_and_stale_frames_are_not_served(self):
        live = self.make_live()
        self.assertFalse(live.status()['camera_connected'])
        self.assertIsNone(live.camera_packet())
        live.frame = dict(jpeg=b'old', sequence=1, received=time.monotonic()-3)
        live.result = dict(camera_received_monotonic_ms=(time.monotonic()-4)*1000)
        self.assertFalse(live.status()['camera_connected'])
        self.assertIsNone(live.camera_packet())
        self.assertIsNone(live.result_packet())

    def test_latest_frame_and_result_keep_separate_identity_and_age(self):
        live = self.make_live()
        live.frame = dict(jpeg=b'new', sequence=42, received=time.monotonic())
        live.result = dict(frame_id=21, camera_received_monotonic_ms=(time.monotonic()-5)*1000)
        self.assertEqual(live.camera_packet(), (b'new', 42))
        result = live.result_packet()
        self.assertEqual(result['frame_id'], 21)
        self.assertGreaterEqual(result['result_age_ms'], 5000)
        live.set_enabled(False)
        self.assertFalse(live.status()['live_enabled'])
        self.assertTrue(live.status()['camera_connected'])

    def test_camera_url_is_loopback_only(self):
        for url in ('https://127.0.0.1/frame.jpg', 'http://example.com/frame.jpg'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                LivePreview(None, threading.Lock(), url)


if __name__ == '__main__':
    unittest.main()
