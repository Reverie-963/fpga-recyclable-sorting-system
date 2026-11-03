import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from src.train_segment import check_dataset


class SegmentDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = []
        for split, value in [('train', 30), ('val', 80)]:
            (self.root / 'images' / split).mkdir(parents=True)
            (self.root / 'labels' / split).mkdir(parents=True)
            cv2.imwrite(str(self.root / 'images' / split / 'scene.png'), np.full((32, 32, 3), value, np.uint8))
            (self.root / 'labels' / split / 'scene.txt').write_text(
                '\n'.join(str(c) + ' 0.1 0.1 0.4 0.1 0.4 0.4 0.1 0.4' for c in range(3)))
            self.rows.append({'split': split, 'image': f'images/{split}/scene.png',
                              'session_id': split + '-session',
                              'object_ids': [split + '-' + str(c) for c in range(3)]})
        self.save()

    def save(self):
        (self.root / 'split_manifest.json').write_text(json.dumps({'images': self.rows}))

    def test_valid_grouped_polygon_data_and_digest(self):
        report = check_dataset(self.root)
        self.assertEqual(report['images'], {'train': 1, 'val': 1})
        self.assertEqual(report['instances']['val'], {0: 1, 1: 1, 2: 1})
        before = report['dataset_sha256']
        path = self.root / 'labels/train/scene.txt'
        path.write_text(path.read_text().replace('0.1', '0.2'))
        self.assertNotEqual(check_dataset(self.root)['dataset_sha256'], before)

    def test_material_schema_explicit_and_order_checked(self):
        manifest = {'images': self.rows, 'class_names': ['can', 'plastic', 'cardboard']}
        path = self.root / 'split_manifest.json'
        path.write_text(json.dumps(manifest))
        self.assertEqual(check_dataset(self.root)['labels'], manifest['class_names'])
        manifest['class_names'] = ['plastic', 'can', 'cardboard']
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'class_names'):
            check_dataset(self.root)

    def test_reject_physical_sample_and_session_leakage(self):
        self.rows[1]['object_ids'][0] = self.rows[0]['object_ids'][0]
        self.save()
        with self.assertRaisesRegex(ValueError, 'leakage: object'):
            check_dataset(self.root)
        self.rows[1]['object_ids'][0] = 'val-0'
        self.rows[1]['session_id'] = self.rows[0]['session_id']
        self.save()
        with self.assertRaisesRegex(ValueError, 'leakage: session'):
            check_dataset(self.root)

    def test_reject_duplicate_image_across_splits(self):
        (self.root / 'images/val/scene.png').write_bytes((self.root / 'images/train/scene.png').read_bytes())
        with self.assertRaisesRegex(ValueError, 'leakage: image_hash'):
            check_dataset(self.root)

    def test_reject_box_labels_nonfinite_and_degenerate_polygons(self):
        for text in ['0 0.5 0.5 0.2 0.2', '0 nan 0.1 0.2 0.2 0.3 0.3',
                     '0 0.1 0.1 0.2 0.2 0.3 0.3', '0 0 0 2 0 1 1']:
            (self.root / 'labels/train/scene.txt').write_text(text)
            with self.assertRaises(ValueError):
                check_dataset(self.root)

    def test_reject_missing_class_or_missing_label(self):
        path = self.root / 'labels/val/scene.txt'
        path.write_text('0 0.1 0.1 0.4 0.1 0.4 0.4')
        with self.assertRaisesRegex(ValueError, 'all three'):
            check_dataset(self.root)
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            check_dataset(self.root)

    def test_reject_unlisted_image_and_no_data(self):
        cv2.imwrite(str(self.root / 'images/train/extra.png'), np.zeros((32, 32, 3), np.uint8))
        with self.assertRaisesRegex(ValueError, 'Manifest'):
            check_dataset(self.root)
        self.rows = []
        self.save()
        with self.assertRaisesRegex(ValueError, 'No annotated images'):
            check_dataset(self.root)


if __name__ == '__main__':
    unittest.main()
