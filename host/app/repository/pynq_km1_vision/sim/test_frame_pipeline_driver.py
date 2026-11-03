"""MMIO transaction, history and exact CPU comparison without PYNQ hardware."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

from src.frame_pipeline import process_frame, result_words
from sim.test_frame_pipeline import regions

spec = importlib.util.spec_from_file_location('candidate_frame_driver',
    Path(__file__).parents[1]/'board/km1_frame_pipeline.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class Buffer(np.ndarray):
    def __new__(cls, size):
        value = np.zeros(size, np.uint32).view(cls)
        value.physical_address = id(value) & 0xffffffff
        value.freed = False
        return value
    def flush(self): pass
    def invalidate(self): pass
    def freebuffer(self): self.freed = True


class DriverTests(unittest.TestCase):
    def backend(self):
        b = driver.StationFramePipeline.__new__(driver.StationFramePipeline)
        b.size = (17, 11)
        n = 17*11
        b.buffers = [Buffer(s) for s in (n, n, (n+31)//32, (n+31)//32, n, 24, 88)]
        b.base, b.frame, b.prior, b.next, b.preview, b.rois, b.results = b.buffers
        b.job, b.poisoned, b.reset_history = 0, False, True
        b.preview_enabled = True
        b.preview_image = b.last_timing = b.last_processing = None
        b.overlay = SimpleNamespace(is_loaded=lambda: True)
        b.lease = SimpleNamespace(close=lambda: None)
        values = {}
        b.ip = SimpleNamespace(write=lambda key, value: values.__setitem__(key, value),
                               read=lambda key: self.read(b, values, key))
        return b

    def read(self, b, values, key):
        if key == 0:
            if values.get(0) == 1:
                base = b.base.view(np.uint8).reshape(11, 17, 4)[:, :, :3]
                frame = b.frame.view(np.uint8).reshape(11, 17, 4)[:, :, :3]
                history = np.unpackbits(b.prior.view(np.uint8), bitorder='little')[:187].reshape(11, 17).astype(bool)
                result = process_frame(base, frame, regions(17, 11), values[0x90],
                    None if values[0x98] else history)
                b.results[:] = result_words(result)
                packed = np.packbits(np.pad(result['history'].ravel(), (0, len(b.next)*32-187)), bitorder='little')
                b.next[:] = packed.view(np.uint32)
                if values[0xa0]:
                    b.preview[:] = result['preview'].reshape(-1, 4).copy().view(np.uint32).ravel()
                values[0] = 2
            return values.get(0, 0)
        return 1 if key == 0xb8 else b.job if key == 0xb0 else 0

    def test_mmio_history_ping_pong_and_original_stats(self):
        b = self.backend()
        image = np.zeros((11, 17, 3), np.uint8)
        b.reset(image, regions(17, 11), 45)
        history = None
        for col in (3, 4, 5):
            frame = image.copy()
            frame[2:9, col:col+6] = 255
            expected = process_frame(image, frame, regions(17, 11), 45, history)
            prior = b.prior
            self.assertEqual(b.process(frame), expected['regions'])
            self.assertIsNot(b.prior, prior)
            np.testing.assert_array_equal(b.preview_image, expected['preview'])
            self.assertEqual(b.last_processing['gray_histogram_16'], expected['gray_histogram_16'])
            self.assertFalse(b.last_processing['automatic_motion_allowed'])
            self.assertFalse(b.last_timing['kernel_only_measured'])
            self.assertGreater(b.last_timing['total_ms'], 0)
            history = expected['history']
            packed = np.packbits(np.pad(history.ravel(), (0, (-history.size) % 32)), bitorder='little')
            np.testing.assert_array_equal(b.read_history_words(), packed.view(np.uint32))
        b.begin_window()
        self.assertTrue(b.reset_history)

    def test_preview_can_be_disabled_without_changing_raw_statistics(self):
        b = self.backend()
        image = np.zeros((11, 17, 3), np.uint8)
        b.reset(image, regions(17, 11), 0)
        b.preview_enabled = False
        b.process(image)
        self.assertIsNone(b.preview_image)
        self.assertFalse(b.last_processing['preview_enabled'])

    def test_bad_completion_keeps_buffers_and_lease_quarantined(self):
        b = self.backend()
        image = np.zeros((11, 17, 3), np.uint8)
        b.reset(image, regions(17, 11), 0)
        b.ip.read = lambda key: 2 if key == 0 else 0
        with self.assertRaisesRegex(RuntimeError, 'job_or_status'):
            b.process(image)
        buffers = b.buffers[:]
        b.close()
        self.assertTrue(all(not x.freed for x in buffers))
        self.assertEqual(b.buffers, [])
        self.assertTrue(b.poisoned)

    def test_invalid_input_cannot_start_or_swap_history(self):
        b = self.backend()
        prior = b.prior
        with self.assertRaises(ValueError):
            b.process(np.zeros((3, 4, 3), np.uint8))
        self.assertIs(b.prior, prior)
        self.assertFalse(b.poisoned)
