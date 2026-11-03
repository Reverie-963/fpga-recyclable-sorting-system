import unittest
import cv2
import numpy as np
from src.sorting_scene import check_motion_layout


class MotionSceneTests(unittest.TestCase):
    def setUp(self):
        self.reference=np.full((80,120,3),100,np.uint8)
        self.policy=dict(image_size=[120,80],static_rois=[[5,5,30,30],[85,45,115,75]],
                         search_margin_px=5,max_layout_shift_px=1,min_patch_std=8,min_patch_corners=4,
                         min_patch_correlation=.98,min_patch_peak_margin=.01,min_motion_patch_dynamic_range=80)
        rng=np.random.default_rng(41)
        for x0,y0,x1,y1 in self.policy['static_rois']:
            self.reference[y0:y1,x0:x1]=rng.integers(30,191,(y1-y0,x1-x0,3),dtype=np.uint8)

    def test_same_frame_patch_cache_preserves_results_and_invalidates_on_changes(self):
        from unittest.mock import patch
        cache={};expected=check_motion_layout(self.reference,self.reference,self.policy)
        with patch('src.sorting_scene.cv2.goodFeaturesToTrack',wraps=cv2.goodFeaturesToTrack) as features:
            one=check_motion_layout(self.reference,self.reference,self.policy,_frame_cache=cache)
            two=check_motion_layout(self.reference,self.reference,self.policy,_frame_cache=cache)
            self.assertEqual(one,expected);self.assertEqual(two,expected);self.assertEqual(features.call_count,2)
        two['markers'][0]['correlation']=-1
        self.assertEqual(check_motion_layout(self.reference,self.reference,self.policy,_frame_cache=cache),expected)
        with self.assertRaises(ValueError):
            check_motion_layout(self.reference,np.roll(self.reference,3,axis=1),self.policy,_frame_cache=cache)
        stricter=dict(self.policy,min_patch_correlation=1.01)
        with self.assertRaises(ValueError):
            check_motion_layout(self.reference,self.reference,stricter,_frame_cache=cache)

    def test_brightness_change_does_not_invent_camera_movement(self):
        current=np.clip(self.reference.astype(float)*1.2+30,0,255).astype(np.uint8)
        report=check_motion_layout(self.reference,current,self.policy)
        self.assertTrue(all(r['shift_px']==0 for r in report['markers']))
        self.assertTrue(all(r['rgb_mean_difference_diagnostic']>12 for r in report['markers']))

    def test_scoped_cache_preserves_results_and_rejects_pixel_mutation(self):
        from src.sorting_scene import MotionLayoutCache
        current=self.reference.copy();expected=check_motion_layout(self.reference,current,self.policy)
        cache=MotionLayoutCache(self.reference,current)
        with cache:
            self.assertEqual(check_motion_layout(self.reference,current,self.policy,_frame_cache=cache),expected)
            self.assertEqual(check_motion_layout(self.reference,current,self.policy,_frame_cache=cache),expected)
        with self.assertRaisesRegex(ValueError,'cache_frame_mismatch'):
            check_motion_layout(self.reference,current,self.policy,_frame_cache=cache)
        with self.assertRaisesRegex(ValueError,'pixels_changed_during_check'):
            with cache:
                check_motion_layout(self.reference,current,self.policy,_frame_cache=cache)
                current[40,60]=255
        self.assertFalse(cache.active)

    def test_scoped_cache_cannot_be_used_for_a_different_array(self):
        from src.sorting_scene import MotionLayoutCache
        current=self.reference.copy()
        with MotionLayoutCache(self.reference,current) as cache:
            with self.assertRaisesRegex(ValueError,'cache_frame_mismatch'):
                check_motion_layout(self.reference,current.copy(),self.policy,_frame_cache=cache)

    def test_three_pixel_camera_movement_still_rejects(self):
        with self.assertRaisesRegex(ValueError,'moved_occluded_or_ambiguous'):
            check_motion_layout(self.reference,np.roll(self.reference,3,axis=1),self.policy)

    def test_blackout_or_occluded_mark_rejects(self):
        for current in [np.zeros_like(self.reference),self.reference.copy()]:
            current[5:30,5:30]=0
            with self.assertRaises(ValueError):check_motion_layout(self.reference,current,self.policy)

    def test_dark_low_dynamic_range_rejects(self):
        with self.assertRaises(ValueError):
            check_motion_layout(self.reference,(self.reference*.1).astype(np.uint8),self.policy)

    def test_body_change_outside_fixed_markers_is_allowed(self):
        current=self.reference.copy();current[32:75,35:75]=240
        self.assertEqual(len(check_motion_layout(self.reference,current,self.policy)['markers']),2)

    def test_changed_marker_pattern_rejects(self):
        current=self.reference.copy();current[5:30,5:30]=np.rot90(current[5:30,5:30])
        with self.assertRaises(ValueError):check_motion_layout(self.reference,current,self.policy)


if __name__=='__main__':unittest.main()
