"""Candidate image previews must remain separate from task evidence."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.request
import urllib.error
import cv2
import numpy as np

from src.frame_pipeline import process_frame
from src.frame_pipeline_server import PipelineFrameStore, make_server
from src.frame_pipeline_selftest import startup_selftest
from src.station_vision import StationVision
from sim.test_frame_pipeline import regions


class Backend:
    backend = 'pl_board_local'
    pipeline_contract = 'station-frame-pipeline-v1'
    assets = {'station_vision.bit': 'a'*64, 'station_vision.hwh': 'b'*64}
    preview_enabled = True
    preview_image = last_processing = last_timing = None
    def reset(self, baseline, rects, threshold):
        self.base, self.rects, self.threshold = baseline.copy(), rects, threshold
        self.history = None
    def begin_window(self): self.history = None
    def read_history_words(self):
        bits = np.pad(self.history.ravel(), (0, (-self.history.size) % 32))
        return np.packbits(bits, bitorder='little').view(np.uint32)
    def process(self, frame):
        out = process_frame(self.base, frame, self.rects, self.threshold, self.history)
        self.history = out['history']
        self.preview_image = out['preview']
        self.last_processing = dict(contract=self.pipeline_contract, regions=out['processing'],
            gray_histogram_16=out['gray_histogram_16'], automatic_motion_allowed=False)
        self.last_timing = dict(total_ms=1., kernel_only_measured=False)
        return out['regions']


class PipelineServiceTests(unittest.TestCase):
    def store(self):
        self.clock = 1.
        backend = Backend()
        engine = StationVision(backend, regions(17, 11), [17, 11], 45)
        return PipelineFrameStore(engine, clock=lambda: self.clock)

    def test_diagnostic_images_do_not_bind_scene_evidence(self):
        store = self.store()
        store.publish(np.zeros((11, 17, 3), np.uint8), self.clock)
        info = store.processing()
        self.assertTrue(info['available'])
        self.assertFalse(info['automatic_motion_allowed'])
        self.assertEqual(info['change_reference'], 'diagnostic_zero_not_task_evidence')
        self.assertIsNone(store.engine.baseline_id)
        self.assertFalse(store.vision()['available'])
        self.assertIn('frame_pipeline_v1', store.health()['capabilities'])

    def test_explicit_baseline_keeps_original_evidence_and_returns_correct_planes(self):
        store = self.store()
        base = np.zeros((11, 17, 3), np.uint8)
        store.publish(base, self.clock)
        store.bind(dict(stream_id=store.stream_id, config_sha256=store.engine.config_sha256, frame_id=1))
        frame = base.copy()
        frame[2:9, 3:14] = 255
        self.clock += .1
        row = store.publish(frame, self.clock)
        expected = process_frame(base, frame, regions(17, 11), 45)
        self.assertEqual(row['evidence']['regions'], expected['regions'])
        np.testing.assert_array_equal(row['processed_preview'], expected['preview'])
        self.assertEqual(store.processing()['change_reference'], 'explicit_bound_baseline')
        self.assertTrue(store.vision()['available'])

    def test_preview_http_requires_authorization_and_exact_fresh_frame(self):
        store = self.store()
        image = np.zeros((11, 17, 3), np.uint8)
        image[2:7, 3:9] = 255
        store.publish(image, self.clock)
        server = make_server('127.0.0.1', 0, 'x'*32, store)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = 'http://127.0.0.1:'+str(server.server_port)
        def get(path, auth=True):
            return urllib.request.urlopen(urllib.request.Request(base+path,
                headers={'Authorization': 'Bearer '+'x'*32} if auth else {}), timeout=3)
        try:
            with get('/processing') as reply:
                self.assertEqual(json.load(reply)['frame_id'], 1)
            with get('/processed.png?plane=gray&frame_id=1') as reply:
                png = reply.read()
                self.assertEqual(reply.headers['X-Frame-Id'], '1')
            decoded = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
            np.testing.assert_array_equal(decoded, store.latest()['processed_preview'][:, :, 0])
            for path, auth, code in (('/processing', False, 403), ('/execute', True, 404),
                    ('/processed.png?plane=missing', True, 503)):
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    get(path, auth)
                self.assertEqual(raised.exception.code, code)
                raised.exception.close()
            store.faulted = True
            with self.assertRaises(urllib.error.HTTPError) as raised:
                get('/processing')
            self.assertEqual(raised.exception.code, 503)
            raised.exception.close()
            store.faulted = False
            self.clock += 3
            with self.assertRaises(urllib.error.HTTPError) as raised:
                get('/processed.png?plane=gray&frame_id=1')
            self.assertEqual(raised.exception.code, 503)
            raised.exception.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

    def test_capture_gap_never_attaches_previous_preview_to_new_frame(self):
        store = self.store()
        base = np.zeros((11, 17, 3), np.uint8)
        store.publish(base, self.clock)
        store.bind(dict(stream_id=store.stream_id, config_sha256=store.engine.config_sha256, frame_id=1))
        self.clock += 1
        row = store.publish(np.full_like(base, 255), self.clock)
        self.assertIsNone(store.engine.baseline_id)
        self.assertIsNone(row['processed_preview'])
        self.assertFalse(row['pipeline']['available'])
        self.assertEqual(row['pipeline']['phases'], {})
        with self.assertRaisesRegex(ValueError, 'current_frame_not_processed'):
            store.processing()
        self.clock += .1
        store.publish(base, self.clock)
        self.assertTrue(store.processing()['available'])
        self.assertEqual(store.processing()['change_reference'], 'diagnostic_zero_not_task_evidence')

    def test_full_workload_selftest_refuses_any_mismatched_summary(self):
        config = dict(image_size=[17, 11], regions=regions(17, 11), threshold=45)
        backend = Backend()
        report = startup_selftest(backend, config)
        self.assertTrue(report['all_equal'])
        self.assertEqual(report['samples'], 12)
        self.assertFalse(report['whole_system_speedup_measured'])
        original = backend.process
        def wrong(frame):
            result = original(frame)
            backend.last_processing['gray_histogram_16'][0] += 1
            return result
        backend.process = wrong
        with self.assertRaisesRegex(RuntimeError, 'summary_mismatch'):
            startup_selftest(backend, config)

    def test_selftest_rejects_wrong_history_tail_even_if_summaries_match(self):
        backend = Backend()
        original = backend.read_history_words
        def wrong():
            words = original()
            words[-1] ^= np.uint32(1 << 31)
            return words
        backend.read_history_words = wrong
        with self.assertRaisesRegex(RuntimeError, 'packed_history_mismatch'):
            startup_selftest(backend, dict(image_size=[17, 11], regions=regions(17, 11), threshold=45))
