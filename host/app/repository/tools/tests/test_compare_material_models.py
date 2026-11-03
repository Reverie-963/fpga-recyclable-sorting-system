"""Contracts for material-ID mapping and stored retrospective evidence."""
import sys
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compare_material_models as comparison


class Tensor:
    def __init__(self, array):
        self.array = np.asarray(array)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array

    def tolist(self):
        return self.array.tolist()

    def __getitem__(self, item):
        return Tensor(self.array[item])

    def __len__(self):
        return len(self.array)


class Boxes:
    cls = Tensor([7, 2])
    conf = Tensor([.8, .7])
    xyxy = Tensor([[0, 0, 2, 2], [2, 1, 4, 3]])

    def __len__(self):
        return 2


class ComparisonContracts(unittest.TestCase):
    def test_extension_rejects_incompatible_baseline_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / 'baseline.json'
            out = Path(directory) / 'output'
            value = dict(policy=dict(comparison.POLICY, conf=.8), dataset_sha256=comparison.DATA_HASH,
                         immutable_inputs_and_old_test_receipts_unchanged=True)
            base.write_text(json.dumps(value), encoding='utf-8')
            arguments = ['compare', '--output', str(out), '--acknowledge-used-test',
                         '--only-yolo11n', '--baseline-comparison', str(base)]
            with patch.object(sys, 'argv', arguments), self.assertRaisesRegex(ValueError, 'policy or dataset'):
                comparison.main()
            self.assertFalse(out.exists())
            value['policy'] = comparison.POLICY
            value['immutable_inputs_and_old_test_receipts_unchanged'] = False
            base.write_text(json.dumps(value), encoding='utf-8')
            with patch.object(sys, 'argv', arguments), self.assertRaisesRegex(ValueError, 'integrity audit'):
                comparison.main()
            self.assertFalse(out.exists())

    def test_prompt_ids_map_to_materials_not_application_numeric_ids(self):
        raw = SimpleNamespace(boxes=Boxes(), masks=SimpleNamespace(data=Tensor(np.ones((2, 3, 4)))))
        predictions = comparison.predictions_for(raw, (3, 4), {7: 1, 2: 0})
        self.assertEqual([p['class_id'] for p in predictions], [1, 0])
        self.assertEqual([p['source_class_id'] for p in predictions], [7, 2])
        self.assertEqual([p['mask'].shape for p in predictions], [(3, 4), (3, 4)])
        with self.assertRaisesRegex(ValueError, 'Unmapped'):
            comparison.predictions_for(raw, (3, 4), {7: 1})

    def test_letterboxed_masks_are_rejected_not_stretched(self):
        raw = SimpleNamespace(boxes=Boxes(), masks=SimpleNamespace(data=Tensor(np.ones((2, 4, 4)))))
        with self.assertRaisesRegex(ValueError, 'original image'):
            comparison.predictions_for(raw, (3, 4), {7: 1, 2: 0})

    def test_saved_masks_round_trip_including_empty_and_full(self):
        for mask in [np.zeros((3, 4), bool), np.ones((3, 4), bool), np.eye(4, dtype=bool),
                     np.array([[1, 0, 1], [0, 1, 0]], dtype=bool)]:
            self.assertTrue(np.array_equal(mask, comparison.decode_mask(comparison.encode_mask(mask), mask.shape)))
        with self.assertRaisesRegex(ValueError, 'Invalid'):
            comparison.decode_mask([1, 2], (3, 4))

    def test_wrong_material_and_duplicate_are_not_true_positives(self):
        mask = np.ones((4, 5), bool)
        truth = [dict(class_id=0, mask=mask)]
        predictions = [dict(class_id=1, mask=mask, confidence=.9),
                       dict(class_id=0, mask=mask, confidence=.8),
                       dict(class_id=0, mask=mask, confidence=.7)]
        counts = comparison.match_instances(truth, predictions, .5)
        self.assertEqual(counts['tp'], [1, 0, 0])
        self.assertEqual(counts['fp'], [1, 1, 0])
        self.assertEqual(counts['fn'], [0, 0, 0])
        self.assertEqual([e['kind'] for e in comparison.error_details(truth, predictions)],
                         ['wrong_material', 'duplicate'])

    def test_bad_contour_is_false_positive_and_missed_truth(self):
        mask = np.ones((4, 5), bool)
        partial = np.zeros_like(mask)
        partial[0] = True
        truth = [dict(class_id=2, mask=mask)]
        predictions = [dict(class_id=2, mask=partial, confidence=.99)]
        counts = comparison.match_instances(truth, predictions, .5)
        self.assertEqual(counts['tp'], [0, 0, 0])
        self.assertEqual(counts['fp'], [0, 0, 1])
        self.assertEqual(counts['fn'], [0, 0, 1])
        errors = comparison.error_details(truth, predictions)
        self.assertEqual(errors[0]['kind'], 'mask_iou_below_0.5')
        self.assertEqual(errors[1]['type'], 'fn')


if __name__ == '__main__':
    unittest.main()
