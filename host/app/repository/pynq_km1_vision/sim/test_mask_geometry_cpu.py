import unittest
import numpy as np
from src.mask_geometry_cpu import image_stats_u8
from src.sorting_io import Statistics, image_stats
from src.static_visual_verification import decode_mask
from src.static_mask_contract import decode_mask as integer_decode
from src.sorting_loop import mask_of


class BinaryMomentsTests(unittest.TestCase):
    def test_rle_decode_and_exclusive_bbox_keep_full_image_contract(self):
        for runs, size in (([0, 6], [3, 2]), ([2, 0, 0, 3, 1], [3, 2]),
                           ([0, 1, 4, 1], [3, 2]), ([307200], [640, 480])):
            actual = decode_mask(runs, size)
            self.assertEqual(actual.dtype, np.uint8)
            self.assertEqual(actual.tolist(), integer_decode(runs, size))
        obj = dict(mask_rle=[0, 1, 4, 1], bbox_xyxy=[0, 0, 3, 2])
        mask = mask_of(obj, [3, 2])
        self.assertEqual(mask.dtype, np.bool_)
        self.assertEqual(mask.tolist(), [[True, False, False], [False, False, True]])
        with self.assertRaisesRegex(ValueError, 'empty_instance_mask'):
            mask_of(dict(mask_rle=[6], bbox_xyxy=[0, 0, 0, 0]), [3, 2])
        with self.assertRaisesRegex(ValueError, 'bbox_mismatch'):
            mask_of(dict(obj, bbox_xyxy=[0, 0, 2, 1]), [3, 2])

    def test_rle_validation_rejects_bool_negative_overflow_and_invalid_size(self):
        for runs in ([True, 5], [-1, 7], [2**80], [5], [], (6,)):
            with self.assertRaisesRegex(ValueError, 'invalid_mask'):
                decode_mask(runs, [3, 2])
        with self.assertRaisesRegex(ValueError, 'invalid_image_size'):
            decode_mask([3], [True, 3])

    def test_all_fields_match_independent_integer_reference(self):
        rng = np.random.default_rng(2041)
        for h, w in ((1, 1), (7, 17), (91, 123), (480, 640)):
            for density in (0, .01, .5, 1):
                mask = rng.random((h, w)) < density
                self.assertEqual(image_stats_u8(mask, 73), image_stats(mask, 73))

    def test_single_pixel_diagonal_and_noncontiguous_views(self):
        mask = np.zeros((480, 640), bool)
        mask[479, 639] = True
        self.assertEqual(image_stats_u8(mask, 1), image_stats(mask, 1))
        rng = np.random.default_rng(6)
        source = rng.random((480, 640)) > .2
        for view in (source[:, ::2], source[::-1, ::-1], source.T[:480, :480]):
            self.assertEqual(image_stats_u8(view, 4), image_stats(view, 4))

    def test_full_frame_closed_form_and_empty_inclusive_bounds(self):
        h, w = 480, 640
        result = image_stats_u8(np.ones((h, w), bool), 8)
        self.assertEqual(result['m20'], h*(w-1)*w*(2*w-1)//6)
        self.assertEqual(result['m02'], w*(h-1)*h*(2*h-1)//6)
        self.assertEqual(result['m11'], (w*(w-1)//2)*(h*(h-1)//2))
        self.assertEqual((result['max_col'], result['max_row']), (639, 479))
        result = image_stats_u8(np.zeros((h, w), bool), 9)
        self.assertEqual([result[k] for k in ('min_col', 'min_row', 'max_col', 'max_row')], [0]*4)

    def test_dispatch_keeps_job_identity_and_legacy_fallback(self):
        stats = Statistics()
        for mask in (np.ones((4, 5), bool), np.ones((3, 3), np.uint8)*255,
                     np.ones((481, 4), bool), np.zeros((0, 3), bool)):
            before = mask.copy()
            result = stats.geometry(mask)
            self.assertEqual(result, image_stats(mask, result['job_id']))
            np.testing.assert_array_equal(mask, before)
        self.assertEqual(stats.job, 4)

    def test_direct_accelerator_rejects_incompatible_contract(self):
        for mask in (np.zeros((481, 2), bool), np.ones((2, 2), np.uint8), np.zeros((2, 2, 2), bool)):
            with self.assertRaisesRegex(ValueError, 'bounded_bool'):
                image_stats_u8(mask, 1)
