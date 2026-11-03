"""Known-item appearance assistance; original neural labels stay intact."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np


def descriptor(image, mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != image.shape[:2] or not mask.any():
        raise ValueError('reference_mask_empty_or_wrong_size')
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], mask.astype(np.uint8), [24, 8], [0, 180, 0, 256]).reshape(-1)
    hist = np.sqrt(hist/max(float(hist.sum()), 1.))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[mask].mean(axis=0)/255.
    y, x = np.nonzero(mask)
    _, (w, h), _ = cv2.minAreaRect(np.column_stack([x, y]).astype(np.float32))
    ratio = np.log(max(w, h, 1.)/max(min(w, h), 1.))/np.log(12.)
    fill = float(mask.sum()/max((x.max()-x.min()+1)*(y.max()-y.min()+1), 1))
    return dict(hist=hist, lab=lab, shape=np.array([ratio, fill]))


def distance(a, b):
    return (.6*float(np.linalg.norm(a['hist']-b['hist'])/np.sqrt(2.))
        + .25*float(np.linalg.norm(a['lab']-b['lab']))
        + .15*float(np.linalg.norm(a['shape']-b['shape'])))


class ReferenceBank:
    def __init__(self, path, expected_sha256):
        raw = Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError('reference_bank_hash_changed')
        data = json.loads(raw)
        if data['schema'] != 'known-inventory-reference-v1' or not data['references']:
            raise ValueError('invalid_reference_bank')
        self.data, self.rows = data, []
        w, h = data['image_size']
        self.region = np.zeros((h, w), np.uint8)
        cv2.fillPoly(self.region, [np.asarray(data['source_polygon'], np.int32)], 1)
        for row in data['references']:
            features = {key: np.asarray(row['features'][key], np.float64) for key in ('hist', 'lab', 'shape')}
            if ([features[k].shape for k in ('hist', 'lab', 'shape')] != [(192,), (3,), (2,)] or
                    not all(np.isfinite(v).all() for v in features.values()) or row['material'] not in ('paper', 'plastic', 'metal')):
                raise ValueError('invalid_reference_features')
            self.rows.append(dict(row, features=features))

    def assist(self, image, mask, image_sha256):
        if mask.shape != self.region.shape or not mask.any() or float(self.region[mask].mean()) < .8:
            return dict(reference_assisted=False, reference_reason='outside_reference_observation_region')
        eligible = [row for row in self.rows if row['image_bgr_sha256'] != image_sha256]
        if not eligible:
            return dict(reference_assisted=False, reference_reason='no_other_reference_image')
        query = descriptor(image, mask)
        score, nearest = min(((distance(query, row['features']), row) for row in eligible), key=lambda pair: pair[0])
        return dict(reference_assisted=True, reference_material=nearest['material'],
            reference_object_id=nearest['object_id'], reference_image=nearest['image'],
            reference_distance=score, reference_distance_is_probability=False)
