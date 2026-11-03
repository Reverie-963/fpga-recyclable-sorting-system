"""Check mask transport and original class IDs without legacy TensorFlow."""
import json
from pathlib import Path
import unittest
import numpy as np
from src.taco_inference import encode_mask, validate_sorting_mapping
from src.static_mask_contract import decode_mask


class TacoMaskTests(unittest.TestCase):
    def test_transport_roundtrip_foreground_at_origin(self):
        mask = np.array([[1, 1, 0], [0, 1, 0]], dtype=np.uint8)
        runs = encode_mask(mask)
        self.assertEqual(runs[0], 0)
        self.assertEqual(decode_mask(runs, [3, 2]), mask.ravel().tolist())

    def test_transport_all_background_and_all_foreground(self):
        for value in (0, 1):
            mask = np.full((3, 5), value, np.uint8)
            self.assertEqual(decode_mask(encode_mask(mask), [5, 3]), mask.ravel().tolist())

    def test_original_class_order_and_user_grouping(self):
        path = Path(__file__).resolve().parents[1] / 'config/taco_original.json'
        cfg = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(cfg['labels'], ['Bottle', 'Bottle cap', 'Can', 'Cigarette',
            'Cup', 'Lid', 'Other', 'Plastic bag + wrapper', 'Pop tab', 'Straw'])
        expected = {
            'Bottle': 'bottle', 'Bottle cap': 'bottle', 'Cup': 'bottle',
            'Straw': 'bottle', 'Lid': 'bottle', 'Can': 'can', 'Pop tab': 'can',
            'Other': 'other', 'Cigarette': 'other', 'Plastic bag + wrapper': 'other'}
        self.assertEqual(cfg['sorting_mapping'], expected)
        self.assertEqual(cfg['sorting_classes'], {'bottle': '瓶子类', 'can': '罐子类', 'other': '其他类'})
        validate_sorting_mapping(cfg)
        self.assertIs(cfg['physical_motion_enabled'], False)

    def test_reject_incomplete_or_unknown_mapping(self):
        path = Path(__file__).resolve().parents[1] / 'config/taco_original.json'
        for kind in ('missing', 'unknown_label', 'unknown_bin'):
            cfg = json.loads(path.read_text(encoding='utf-8'))
            if kind == 'missing':
                del cfg['sorting_mapping']['Lid']
            elif kind == 'unknown_label':
                cfg['sorting_mapping']['Unrecognized'] = 'other'
            else:
                cfg['sorting_mapping']['Can'] = 'metal'
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_sorting_mapping(cfg)


if __name__ == '__main__':
    unittest.main()
