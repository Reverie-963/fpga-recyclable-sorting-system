"""Stable identity for a fixed weight plus its inference configuration."""
import hashlib
import json
import math


def configuration(entry):
    confidence = entry.get('confidence', .30)
    if type(confidence) not in (int, float) or not math.isfinite(confidence) or not .01 <= confidence <= .95:
        raise ValueError('invalid_profile_confidence')
    return dict(schema='recognition-profile-v1', profile_id=entry['id'], model_sha256=entry['sha256'],
        classes=entry['classes'], mapping=entry['mapping'], confidence=float(confidence),
        image_size=640, nms_iou=.7, max_detections=100, agnostic_nms=True,
        reference_bank_sha256=entry.get('reference_bank_sha256'))


def configuration_hash(entry):
    return hashlib.sha256(json.dumps(configuration(entry), sort_keys=True, separators=(',', ':')).encode()).hexdigest()
