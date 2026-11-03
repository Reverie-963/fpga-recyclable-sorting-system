import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.prepare_recyclables_public import build, candidates, polygon_line, sha256


class PublicDataTests(unittest.TestCase):
    def test_reject_whole_scene_with_unlabeled_or_unsupported_target_mask(self):
        image = {'id': 1, 'width': 100, 'height': 100}
        ann = {'id': 2, 'image_id': 1, 'category_id': 0,
               'segmentation': [[10, 10, 60, 10, 60, 60, 10, 60]]}
        data = {'images': [image], 'annotations': [ann],
                'categories': [{'id': 0, 'name': 'Drink can'}, {'id': 1, 'name': 'Unlabeled litter'}]}
        self.assertEqual(len(candidates(data)[0]), 1)
        data['annotations'].append(dict(ann, id=3, category_id=1))
        self.assertEqual(len(candidates(data)[0]), 0)
        data['annotations'].pop()
        ann['segmentation'].append([70, 70, 80, 70, 80, 80])
        self.assertEqual(len(candidates(data)[0]), 0)

    def test_normalization_and_reject_box_or_tiny_mask(self):
        image = {'width': 100, 'height': 200}
        ann = {'category_id': 4, 'segmentation': [[10, 20, 60, 20, 60, 120]]}
        values = [float(v) for v in polygon_line(ann, image, {4: 1}).split()]
        self.assertEqual(values, [1, .1, .1, .6, .1, .6, .6])
        for segments in ([[10, 20, 60, 20]], [[1, 1, 1.001, 1, 1, 1.001]], {'counts': []}):
            ann['segmentation'] = segments
            with self.assertRaises(ValueError):
                polygon_line(ann, image, {4: 1})

    def test_build_balanced_train_only_preserves_holdout_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, cache, output = root / 'base', root / 'cache', root / 'output'
            rows = []
            rng = np.random.default_rng(4)
            for split in ('train', 'val'):
                (base / 'images' / split).mkdir(parents=True)
                (base / 'labels' / split).mkdir(parents=True)
                cv2.imwrite(str(base / 'images' / split / 'local.png'), rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))
                (base / 'labels' / split / 'local.txt').write_text('\n'.join(
                    f'{c} .1 .1 .6 .1 .6 .6' for c in range(3)))
                rows.append({'image': f'images/{split}/local.png', 'split': split,
                             'session_id': split, 'object_ids': [split]})
            (base / 'split_manifest.json').write_text(json.dumps(
                {'class_names': ['can', 'plastic', 'cardboard'], 'images': rows}))
            cache.mkdir()
            data = {'images': [], 'annotations': [], 'categories': [
                {'id': c, 'name': name} for c, name in enumerate(
                    ['Drink can', 'Clear plastic bottle', 'Drink carton'])]}
            for i in range(12):
                data['images'].append({'id': i, 'width': 64, 'height': 64,
                    'file_name': f'batch_1/{i}.jpg',
                    'flickr_url': f'https://farm1.staticflickr.com/test/{i}.jpg'})
                data['annotations'].append({'id': i, 'image_id': i, 'category_id': i % 3,
                    'segmentation': [[5, 5, 50, 5, 50, 50]]})
                if i < 3:
                    raw = (base / 'images/val/local.png').read_bytes()
                else:
                    raw = cv2.imencode('.png', rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))[1].tobytes()
                (cache / f'taco-{i:04d}.image').write_bytes(raw)
            (cache / 'taco-release-1.0-annotations.json').write_text(json.dumps(data))
            with patch('src.prepare_recyclables_public.fetch', side_effect=AssertionError('No network in test')):
                report = build(base, output, cache, per_class=1, workers=1)
            self.assertEqual(report['public_images'], 3)
            self.assertEqual(report['balance_buckets'], {0: 1, 1: 1, 2: 1})
            self.assertGreater(report['skipped'].get('exact_or_near_duplicate', 0), 0)
            for prefix, suffix in [('images', '.png'), ('labels', '.txt')]:
                rel = Path(prefix, 'val', 'local' + suffix)
                self.assertEqual(sha256((base / rel).read_bytes()), sha256((output / rel).read_bytes()))
            manifest = json.loads((output / 'split_manifest.json').read_text(encoding='utf-8'))
            self.assertTrue(all(r['split'] == 'train' for r in manifest['images'] if r.get('source')))
            with self.assertRaisesRegex(ValueError, 'exists'):
                build(base, output, cache, per_class=1)


if __name__ == '__main__':
    unittest.main()
