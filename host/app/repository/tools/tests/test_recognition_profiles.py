"""Check profile identity and the known-item support bank's boundaries."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'pynq_km1_vision'))
from src.recognition_profiles import configuration_hash
from src.inventory_reference import ReferenceBank, descriptor, distance


class ProfileTests(unittest.TestCase):
    def test_same_weight_has_different_identity_for_threshold_and_references(self):
        base = dict(id='same', sha256='weight', classes=['can'], mapping={'can':'metal'}, confidence=.1)
        hashes = {configuration_hash(base), configuration_hash(dict(base, confidence=.3)),
                  configuration_hash(dict(base, reference_bank_sha256='support-bank'))}
        self.assertEqual(len(hashes), 3)
        for threshold in (0, float('nan'), True, 1):
            with self.assertRaises(ValueError): configuration_hash(dict(base, confidence=threshold))

    def test_descriptor_uses_foreground_and_assistance_excludes_query(self):
        image = np.zeros((30, 40, 3), np.uint8)
        mask = np.zeros((30, 40), bool); mask[5:15, 5:15] = True
        image[mask] = [0, 0, 220]
        changed = image.copy(); changed[~mask] = [100, 255, 50]
        features = descriptor(image, mask)
        self.assertAlmostEqual(distance(features, descriptor(changed, mask)), 0)
        row = dict(image='support.jpg', image_bgr_sha256='support-bgr', object_id='can-1', material='metal',
                   features={k:v.tolist() for k,v in features.items()})
        bank = dict(schema='known-inventory-reference-v1', image_size=[40,30],
                    source_polygon=[[0,0],[20,0],[20,20],[0,20]], references=[row])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'bank.json'; path.write_text(json.dumps(bank))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            reference = ReferenceBank(path, digest)
            self.assertEqual(reference.assist(image, mask, 'different-query')['reference_material'], 'metal')
            self.assertFalse(reference.assist(image, mask, 'support-bgr')['reference_assisted'])
            outside = np.zeros_like(mask); outside[22:28,25:35] = True
            self.assertFalse(reference.assist(image, outside, 'query')['reference_assisted'])
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'hash_changed'): ReferenceBank(path, digest)


if __name__ == '__main__': unittest.main()
