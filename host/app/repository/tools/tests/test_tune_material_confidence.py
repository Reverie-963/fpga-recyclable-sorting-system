"""Check cached confidence curves against full mask matching."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import tune_material_confidence as tuning


class ConfidenceContracts(unittest.TestCase):
    def test_prefix_counts_equal_original_matching_at_all_thresholds(self):
        left=np.zeros((6,8),bool);left[:,:4]=True
        right=~left
        truth=[dict(class_id=0,mask=left),dict(class_id=1,mask=right)]
        predictions=[dict(class_id=2,confidence=.95,mask=left),
            dict(class_id=0,confidence=.8,mask=left),
            dict(class_id=0,confidence=.7,mask=left),
            dict(class_id=1,confidence=.5,mask=right),
            dict(class_id=1,confidence=.2,mask=left),
            dict(class_id=2,confidence=.1,mask=right)]
        prefix=tuning.prepare_prefix(truth,predictions)
        for threshold in tuning.GRID:
            filtered=[p for p in predictions if tuning.keep(p['confidence'],threshold)]
            self.assertEqual(tuning.counts_at(prefix,threshold),tuning.c.match_instances(truth,filtered,.5))

    def test_strict_fp32_boundary_matches_native_comparison(self):
        boundary=float(np.float32(.3))
        self.assertFalse(tuning.keep(boundary,.3))
        self.assertTrue(tuning.keep(float(np.nextafter(np.float32(.3),np.float32(1))),.3))

    def test_empty_false_positive_is_removed_but_empty_image_retained(self):
        predictions=[dict(class_id=1,confidence=.2,mask=np.ones((2,2),bool))]
        prefix=tuning.prepare_prefix([],predictions)
        self.assertEqual(tuning.counts_at(prefix,.1)['empty_false_positive_images'],1)
        counts=tuning.counts_at(prefix,.3)
        self.assertEqual(counts['empty_false_positive_images'],0)
        self.assertEqual(counts['empty_images'],1)
        self.assertEqual(counts['images'],1)

    def test_wrong_contour_never_becomes_true_positive_at_lower_confidence(self):
        mask=np.ones((4,4),bool);partial=np.zeros_like(mask);partial[0]=True
        truth=[dict(class_id=0,mask=mask)]
        prefix=tuning.prepare_prefix(truth,[dict(class_id=0,confidence=.9,mask=partial)])
        counts=tuning.counts_at(prefix,.1)
        self.assertEqual(counts['tp'],[0,0,0])
        self.assertEqual(counts['fp'],[1,0,0])
        self.assertEqual(counts['fn'],[1,0,0])


if __name__=='__main__':
    unittest.main()
