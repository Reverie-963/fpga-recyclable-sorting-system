import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from src.static_visual_verification import review_checkpoint, review_recording


def rle(mask):
    runs, bit, length = [], 0, 0
    for value in mask.ravel():
        if value != bit:
            runs.append(length)
            bit, length = int(value), 0
        length += 1
    return runs + [length]


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.p = dict(task_id='one', camera_id='camera', model_version='frozen-240',
                      calibration_id='fixed', bin_id='right-paper', label='cardboard',
                      image_size=[80, 60], source_roi=[0, 20, 20, 40],
                      grip_roi=[25, 20, 45, 40], bin_roi=[55, 20, 75, 40],
                      static_rois=[[2, 2, 18, 18], [60, 42, 78, 58]],
                      reference_frame_id=0, reference_capture_started_ms=500,
                      stable_frames=3, search_margin_px=5, max_layout_shift_px=1,
                      min_patch_std=8, min_patch_corners=4, min_patch_peak_margin=.01,
                      min_patch_correlation=.9, max_patch_mean_difference=12,
                      min_frame_gap_ms=100, max_age_ms=2000, min_target_area_px=8,
                      max_target_area_px=200, min_bin_gain_px=8, min_confidence=.8,
                      min_source_erased_fraction=.9, min_target_roi_fraction=.95,
                      min_persistent_fraction=.7, max_center_jitter_px=2,
                      rgb_difference_threshold=90)
        self.reference = np.full((60, 80, 3), 100, np.uint8)
        rng = np.random.default_rng(41)
        for x0, y0, x1, y1 in self.p['static_rois']:
            self.reference[y0:y1, x0:x1] = rng.integers(0, 256, (y1-y0, x1-x0, 3), dtype=np.uint8)
        self.source = np.zeros((60, 80), np.uint8)
        self.source[24:30, 4:10] = 1

    def rows(self, stage='placed'):
        mask = np.zeros_like(self.source)
        x = 60 if stage == 'placed' else 30
        mask[24:30, x:x+6] = 1
        rows = []
        for i in range(3):
            image = self.reference.copy()
            image[mask.astype(bool)] = 230
            observation = {k: self.p[k] for k in ('task_id', 'camera_id', 'model_version', 'calibration_id', 'bin_id')}
            observation.update(frame_id=i+1, capture_started_ms=1000+i*200,
                               image_size=[80, 60], stage=stage,
                               instances=[dict(label='cardboard', confidence=.9, mask_rle=rle(mask))])
            rows.append(dict(image=image, observation=observation))
        return rows

    def review(self, rows=None, stage='placed', now=1500):
        return review_checkpoint(self.reference, rle(self.source),
                                 self.rows(stage) if rows is None else rows, self.p, stage, now)

    def assert_review(self, rows, reason, stage='placed'):
        result = self.review(rows, stage)
        self.assertEqual(result['status'], 'needs_review', result)
        self.assertIn(reason, result['reasons'])
        self.assertFalse(result['automatic_motion_allowed'])

    def test_all_stages_are_visual_evidence_only(self):
        for stage in ('lift', 'hold', 'placed'):
            result = self.review(stage=stage)
            self.assertEqual(result['status'], 'visual_evidence_pass', result)
            for flag in ('physical_success_verified', 'independently_validated', 'automatic_motion_allowed', 'automatic_retry'):
                self.assertFalse(result[flag])

    def test_source_rgb_difference_is_diagnostic_not_success_requirement(self):
        result = self.review()
        self.assertEqual(result['status'], 'visual_evidence_pass')
        self.assertTrue(all(f['source_rgb_changed_fraction'] == 0 for f in result['frames']))

    def test_box_change_without_semantic_target_is_not_success(self):
        rows = self.rows()
        for row in rows:
            row['observation']['instances'] = []
        self.assert_review(rows, 'target_missing_or_ambiguous_at_checkpoint')

    def test_source_object_still_present_or_replaced_blocks(self):
        for label in ('cardboard', 'plastic'):
            rows = self.rows()
            rows[1]['observation']['instances'].append(dict(label=label, confidence=.3, mask_rle=rle(self.source)))
            self.assert_review(rows, 'source_not_clear')

    def test_drop_or_occlusion_during_hold_blocks(self):
        rows = self.rows('hold')
        rows[1]['observation']['instances'] = []
        self.assert_review(rows, 'target_missing_or_ambiguous_at_checkpoint', 'hold')

    def test_wrong_material_and_low_confidence_block(self):
        rows = self.rows()
        rows[1]['observation']['instances'][0]['label'] = 'plastic'
        self.assert_review(rows, 'target_missing_or_ambiguous_at_checkpoint')
        rows = self.rows()
        rows[1]['observation']['instances'][0]['confidence'] = .79
        self.assert_review(rows, 'uncertain_checkpoint_target')

    def test_ambiguous_bin_blocks(self):
        rows = self.rows()
        rows[1]['observation']['instances'] *= 2
        self.assert_review(rows, 'target_missing_or_ambiguous_at_checkpoint')

    def test_wrong_bin_region_blocks(self):
        rows = self.rows('hold')
        for row in rows:
            row['observation']['stage'] = 'placed'
        self.assert_review(rows, 'target_missing_or_ambiguous_at_checkpoint')

    def test_existing_bin_object_without_new_appearance_blocks(self):
        rows = self.rows()
        for row in rows:
            row['image'] = self.reference.copy()
        self.assert_review(rows, 'no_persistent_new_bin_target')

    def test_transient_shadow_or_change_blocks(self):
        rows = self.rows()
        rows[1]['image'] = self.reference.copy()
        self.assert_review(rows, 'no_persistent_new_bin_target')

    def test_camera_shift_blocks(self):
        rows = self.rows()
        rows[1]['image'] = np.roll(rows[1]['image'], 3, axis=1)
        self.assert_review(rows, 'static_layout_changed_or_occluded')

    def test_static_patch_occlusion_blocks(self):
        rows = self.rows()
        rows[1]['image'][2:18, 2:18] = 0
        self.assert_review(rows, 'static_layout_changed_or_occluded')

    def test_changed_arm_appearance_outside_static_patches_is_allowed(self):
        rows = self.rows()
        for row in rows:
            row['image'][45:59, 25:45] = 240
        self.assertEqual(self.review(rows)['status'], 'visual_evidence_pass')

    def test_textureless_reference_cannot_claim_camera_stability(self):
        self.reference[2:18, 2:18] = 100
        self.assert_review(self.rows(), 'static_patch_lacks_texture')

    def test_straight_edge_is_not_a_two_dimensional_camera_reference(self):
        self.reference[2:18, 2:10] = 0
        self.reference[2:18, 10:18] = 255
        self.assert_review(self.rows(), 'static_patch_lacks_corners')

    def test_repeating_pattern_cannot_hide_ambiguous_motion(self):
        rng = np.random.default_rng(21)
        tile = rng.integers(0, 256, (4, 4, 3), dtype=np.uint8)
        self.reference[:20, :24] = np.tile(tile, (5, 6, 1))
        self.assert_review(self.rows(), 'static_patch_ambiguous_match')

    def test_target_slipping_blocks(self):
        rows = self.rows('hold')
        mask = np.zeros_like(self.source)
        mask[24:30, 35:41] = 1
        rows[-1]['observation']['instances'][0]['mask_rle'] = rle(mask)
        self.assert_review(rows, 'target_unstable_or_slipping', 'hold')

    def test_identity_mismatches_block(self):
        for key in ('task_id', 'camera_id', 'model_version', 'calibration_id', 'bin_id'):
            rows = self.rows()
            rows[1]['observation'][key] = 'wrong'
            self.assert_review(rows, 'frame_identity_mismatch:' + key)

    def test_frame_reuse_wrong_stage_and_stale_timestamps_block(self):
        rows = self.rows()
        rows[1]['observation']['frame_id'] = 1
        self.assert_review(rows, 'repeated_or_unordered_frame')
        rows = self.rows()
        rows[1]['observation']['stage'] = 'lift'
        self.assert_review(rows, 'frame_stage_or_size_mismatch')
        self.assertEqual(self.review(now=5000)['reasons'], ['stale_or_future_frame'])

    def test_reference_is_older_than_checkpoint(self):
        self.p['reference_frame_id'] = 5
        self.assert_review(self.rows(), 'repeated_or_unordered_frame')
        self.p['reference_frame_id'] = 0
        self.p['reference_capture_started_ms'] = 1200
        self.assert_review(self.rows(), 'frame_precedes_reference')

    def test_burst_frames_do_not_count_as_hold_interval(self):
        rows = self.rows('hold')
        rows[1]['observation']['capture_started_ms'] = 1001
        self.assert_review(rows, 'insufficient_checkpoint_time_span', 'hold')

    def test_missing_frames_do_not_default_to_pass(self):
        self.assert_review(self.rows()[:2], 'wrong_checkpoint_frame_count')

    def test_overlapping_or_local_static_patches_block(self):
        self.p['static_rois'][0] = [0, 20, 15, 35]
        self.assert_review(self.rows(), 'static_patch_overlaps_action_or_patch')
        self.p['static_rois'] = [[2, 2, 8, 8], [9, 2, 15, 8]]
        self.assert_review(self.rows(), 'static_patches_too_close')

    def test_nonfinite_policy_and_malformed_masks_block(self):
        self.p['min_confidence'] = float('nan')
        self.assert_review(self.rows(), 'invalid_threshold:min_confidence')
        self.p['min_confidence'] = .8
        rows = self.rows()
        rows[1]['observation']['instances'][0]['mask_rle'] = [1]
        self.assert_review(rows, 'invalid_mask')

    def test_recording_checks_image_hash_and_mask_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            def save(name, image):
                path = base / name
                cv2.imwrite(str(path), image)
                return dict(path=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            document = dict(schema_version='static-visual-checkpoint-v1',
                            reference_image=save('reference.png', self.reference),
                            before_mask_rle=rle(self.source), policy=self.p,
                            stage='placed', recorded_review_ms=1500, frames=[])
            before = {k: self.p[k] for k in ('task_id', 'camera_id', 'model_version', 'calibration_id', 'bin_id', 'image_size')}
            before.update(stage='before', frame_id=0, capture_started_ms=500,
                          image_sha256=document['reference_image']['sha256'],
                          instances=[dict(label='cardboard', confidence=.9, mask_rle=rle(self.source))])
            document['before_observation'] = before
            for i, row in enumerate(self.rows()):
                entry = save(f'{i}.png', row['image'])
                observation = copy.deepcopy(row['observation'])
                observation['image_sha256'] = entry['sha256']
                document['frames'].append(dict(image=entry, observation=observation))
            result = review_recording(json.loads(json.dumps(document)), base)
            self.assertEqual(result['status'], 'visual_evidence_pass', result)
            document['before_observation']['model_version'] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'reference_identity_mismatch:model_version'):
                review_recording(document, base)
            document['before_observation']['model_version'] = self.p['model_version']
            document['frames'][0]['observation']['image_sha256'] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'mask_image_binding_mismatch'):
                review_recording(document, base)
            document['frames'][0]['observation']['image_sha256'] = document['frames'][0]['image']['sha256']
            (base / '0.png').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'image_hash_mismatch'):
                review_recording(document, base)


if __name__ == '__main__':
    unittest.main()
