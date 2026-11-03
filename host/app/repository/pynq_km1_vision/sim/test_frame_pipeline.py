"""The candidate pipeline must preserve all original RGB evidence exactly."""
import unittest
import cv2
import numpy as np

from src.frame_pipeline import process_frame, result_words, unpack_results
from src.station_vision import oracle


def regions(w, h):
    return {name: [0, 0, w, h] for name in
            ('source', 'bin_plastic', 'bin_metal', 'bin_paper', 'static_a', 'static_b')}


class FramePipelineTests(unittest.TestCase):
    def test_original_rgb_stats_and_raw_persistence_are_unchanged(self):
        rng = np.random.default_rng(18)
        base = rng.integers(0, 256, (19, 35, 3), dtype=np.uint8)
        history = None
        for _ in range(4):
            frame = rng.integers(0, 256, base.shape, dtype=np.uint8)
            expected, hist = oracle(base, frame, regions(35, 19), 45, history)
            out = process_frame(base, frame, regions(35, 19), 45, history)
            self.assertEqual(out['regions'], expected)
            np.testing.assert_array_equal(out['history'], hist)
            history = out['history']
            words = result_words(out)
            self.assertEqual(len(words), 88)
            decoded = unpack_results(words)
            for key in ('regions', 'processing', 'gray_histogram_16'):
                self.assertEqual(decoded[key], out[key])
            self.assertFalse(out['automatic_motion_allowed'])

    def test_four_planes_and_zero_border_mean_are_exact(self):
        image = np.full((5, 7, 3), 255, dtype=np.uint8)
        out = process_frame(np.zeros_like(image), image, regions(7, 5))
        preview = out['preview']
        self.assertEqual(preview.shape, (5, 7, 4))
        self.assertEqual(preview.dtype, np.uint8)
        self.assertEqual(int(preview[0, 0, 0]), 255)
        self.assertEqual(int(preview[0, 0, 1]), (4*255+4)//9)
        self.assertEqual(int(preview[2, 3, 1]), 255)
        self.assertEqual(sum(out['gray_histogram_16']), 35)

    def test_single_pixel_changes_are_removed_only_in_display_branch(self):
        base = np.zeros((9, 9, 3), np.uint8)
        frame = base.copy()
        frame[4, 4] = 255
        out = process_frame(base, frame, regions(9, 9))
        self.assertEqual(out['regions']['source']['changed_px'], 1)
        self.assertEqual(out['regions']['source']['persistent_px'], 1)
        self.assertEqual(out['processing']['source']['clean_changed_px'], 0)

    def test_tiny_images_and_last_row_last_column_are_retained(self):
        for w, h in ((1, 1), (1, 7), (7, 1), (2, 2), (640, 480)):
            base = np.zeros((h, w, 3), np.uint8)
            frame = base.copy()
            frame[-1, -1] = 255
            out = process_frame(base, frame, regions(w, h))
            self.assertEqual(out['regions']['source']['bbox_inclusive_px'], [w-1, h-1, w-1, h-1])
            self.assertEqual(int(out['preview'][-1, -1, 0]), 255)
            self.assertEqual(sum(out['gray_histogram_16']), w*h)

    def test_invalid_sizes_history_and_threshold_are_rejected(self):
        frame = np.zeros((8, 10, 3), np.uint8)
        for history in (np.zeros((7, 10), bool), np.zeros((8, 10), np.uint8)):
            with self.assertRaises(ValueError):
                process_frame(frame, frame, regions(10, 8), history=history)
        with self.assertRaises(ValueError):
            process_frame(frame, frame, regions(10, 8), 766)
        with self.assertRaises(ValueError):
            unpack_results(np.zeros(48, np.uint32))

    def test_preview_disabled_preserves_all_statistics(self):
        image = np.full((9, 11, 3), 123, np.uint8)
        base = np.zeros_like(image)
        enabled = process_frame(base, image, regions(11, 9))
        disabled = process_frame(base, image, regions(11, 9), preview_enabled=False)
        self.assertIsNone(disabled['preview'])
        for key in ('regions', 'processing', 'gray_histogram_16'):
            self.assertEqual(enabled[key], disabled[key])
        np.testing.assert_array_equal(enabled['history'], disabled['history'])
