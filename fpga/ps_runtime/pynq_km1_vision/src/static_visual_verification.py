"""Read-only, provisional visual checkpoint review. No hardware or motion imports.

Images and semantic masks must come from the same recorded frame. A pass means
visual evidence only; independent positive/negative trials remain necessary.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np


def decode_mask(runs, size):
    """Full-image row-major RLE, starting with a background run (possibly 0)."""
    require(isinstance(size, list) and len(size) == 2 and
            all(type(v) is int for v in size) and 1 <= size[0] <= 640 and 1 <= size[1] <= 480,
            'invalid_image_size')
    require(isinstance(runs, list) and runs and
            all(type(v) is int and v >= 0 for v in runs) and sum(runs) == size[0] * size[1],
            'invalid_mask')
    values = np.empty(len(runs), dtype=np.uint8)
    values[::2] = 0
    values[1::2] = 1
    return np.repeat(values, runs)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def roi_mask(roi, size):
    w, h = size
    require(isinstance(roi, list) and len(roi) == 4 and
            all(type(v) is int for v in roi), 'invalid_roi')
    x0, y0, x1, y1 = roi
    require(0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h, 'invalid_roi')
    mask = np.zeros((h, w), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def validate_policy(p):
    size = p['image_size']
    decode_mask([size[0] * size[1]], size)
    for key in ('task_id', 'camera_id', 'model_version', 'calibration_id', 'label', 'bin_id'):
        require(isinstance(p[key], str) and p[key].strip(), 'invalid_policy_identity')
    regions = [roi_mask(p[key], size) for key in ('source_roi', 'grip_roi', 'bin_roi')]
    require(not any(np.any(a & b) for i, a in enumerate(regions)
                    for b in regions[i + 1:]), 'overlapping_action_rois')
    patches = [roi_mask(r, size) for r in p['static_rois']]
    require(len(patches) >= 2, 'need_two_static_patches')
    for i, patch in enumerate(patches):
        require(not any(np.any(patch & r) for r in regions + patches[:i]),
                'static_patch_overlaps_action_or_patch')
    centers = [((r[0] + r[2]) / 2, (r[1] + r[3]) / 2) for r in p['static_rois']]
    require(max(math.dist(a, b) for a in centers for b in centers) >= min(size) / 3,
            'static_patches_too_close')
    require(type(p['stable_frames']) is int and p['stable_frames'] >= 3,
            'invalid_stable_frames')
    require(type(p['reference_frame_id']) is int and p['reference_frame_id'] >= 0 and
            number(p['reference_capture_started_ms']) and p['reference_capture_started_ms'] >= 0,
            'invalid_reference_frame')
    require(type(p['search_margin_px']) is int and p['search_margin_px'] >= 1,
            'invalid_search_margin')
    require(type(p['min_patch_corners']) is int and 4 <= p['min_patch_corners'] <= 100,
            'invalid_patch_corner_count')
    for key in ('min_frame_gap_ms', 'max_age_ms', 'min_target_area_px',
                'max_target_area_px', 'min_bin_gain_px', 'min_patch_std',
                'rgb_difference_threshold', 'min_patch_peak_margin'):
        require(number(p[key]) and p[key] > 0, 'invalid_threshold:' + key)
    for key in ('max_center_jitter_px', 'max_layout_shift_px', 'max_patch_mean_difference'):
        require(number(p[key]) and p[key] >= 0, 'invalid_threshold:' + key)
    for key in ('min_confidence', 'min_source_erased_fraction',
                'min_target_roi_fraction', 'min_persistent_fraction', 'min_patch_correlation'):
        require(number(p[key]) and 0 < p[key] <= 1, 'invalid_threshold:' + key)
    require(p['min_target_area_px'] <= p['max_target_area_px'], 'invalid_area_range')
    require(p['max_layout_shift_px'] < p['search_margin_px'], 'invalid_layout_search')
    return regions


def layout_evidence(reference, current, p):
    """Check only frozen static patches; do not warp images or compensate motion."""
    a = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    w, h = p['image_size']
    reports = []
    margin = p['search_margin_px']
    for x0, y0, x1, y1 in p['static_rois']:
        template = a[y0:y1, x0:x1]
        require(float(template.std()) >= p['min_patch_std'], 'static_patch_lacks_texture')
        corners = cv2.goodFeaturesToTrack(template, maxCorners=100, qualityLevel=.01, minDistance=3)
        require(corners is not None and len(corners) >= p['min_patch_corners'],
                'static_patch_lacks_corners')
        support = float(cv2.contourArea(cv2.convexHull(corners))) / template.size
        require(support >= .1, 'static_patch_corner_support_too_local')
        sx, sy = max(0, x0 - margin), max(0, y0 - margin)
        search = b[sy:min(h, y1 + margin), sx:min(w, x1 + margin)]
        scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(scores)
        ys, xs = np.indices(scores.shape)
        alternatives = scores.copy()
        radius = max(1, p['max_layout_shift_px'])
        alternatives[(xs-location[0])**2 + (ys-location[1])**2 < radius**2] = -1
        peak_margin = score - float(alternatives.max())
        require(peak_margin >= p['min_patch_peak_margin'], 'static_patch_ambiguous_match')
        shift = math.hypot(sx + location[0] - x0, sy + location[1] - y0)
        delta = float(np.abs(current[y0:y1, x0:x1].astype(np.int16) -
                            reference[y0:y1, x0:x1].astype(np.int16)).mean())
        reports.append(dict(correlation=score, shift_px=shift, mean_abs_difference=delta,
                            corners=len(corners), corner_support_fraction=support))
        reports[-1]['peak_margin'] = peak_margin
        require(score >= p['min_patch_correlation'] and shift <= p['max_layout_shift_px']
                and delta <= p['max_patch_mean_difference'], 'static_layout_changed_or_occluded')
    return reports


def review_checkpoint(reference, before_mask_rle, frames, policy, stage, now_ms):
    """Review lift/hold/placed recorded at stationary checkpoints.

    Each frame: image (BGR uint8), observation (identity, frame_id,
    capture_started_ms, instances with label/confidence/mask_rle). Identity
    includes task, camera, model, calibration and bin. No default mock success.
    """
    out = dict(stage=stage, status='needs_review', reasons=[], frames=[],
               automatic_retry=False, automatic_motion_allowed=False,
               physical_success_verified=False, independently_validated=False,
               backend='cpu_recorded_images_and_semantic_masks')
    try:
        require(stage in ('lift', 'hold', 'placed'), 'invalid_stage')
        source_roi, grip_roi, bin_roi = validate_policy(policy)
        w, h = policy['image_size']
        require(isinstance(reference, np.ndarray) and reference.shape == (h, w, 3)
                and reference.dtype == np.uint8, 'invalid_reference_image')
        before = np.array(decode_mask(before_mask_rle, [w, h]), dtype=bool).reshape(h, w)
        require(before.any() and not (before & ~source_roi).any(), 'invalid_source_reference')
        require(isinstance(frames, list) and len(frames) == policy['stable_frames'],
                'wrong_checkpoint_frame_count')
        require(number(now_ms), 'invalid_review_clock')
        previous_id, previous_time = policy['reference_frame_id'], None
        masks, changes, centers = [], [], []
        for row in frames:
            image, observation = row['image'], row['observation']
            require(isinstance(image, np.ndarray) and image.shape == reference.shape
                    and image.dtype == np.uint8, 'invalid_frame_image')
            for key in ('task_id', 'camera_id', 'model_version', 'calibration_id', 'bin_id'):
                require(observation[key] == policy[key], 'frame_identity_mismatch:' + key)
            require(observation['image_size'] == [w, h] and observation['stage'] == stage,
                    'frame_stage_or_size_mismatch')
            frame_id, captured = observation['frame_id'], observation['capture_started_ms']
            require(type(frame_id) is int and frame_id > previous_id, 'repeated_or_unordered_frame')
            require(number(captured) and 0 <= now_ms - captured <= policy['max_age_ms'],
                    'stale_or_future_frame')
            require(captured > policy['reference_capture_started_ms'], 'frame_precedes_reference')
            require(previous_time is None or captured - previous_time >= policy['min_frame_gap_ms'],
                    'insufficient_checkpoint_time_span')
            previous_id, previous_time = frame_id, captured
            layout = layout_evidence(reference, image, policy)
            union = np.zeros((h, w), dtype=bool)
            targets = []
            destination = bin_roi if stage == 'placed' else grip_roi
            for instance in observation['instances']:
                confidence = instance['confidence']
                require(number(confidence) and 0 <= confidence <= 1, 'invalid_confidence')
                mask = np.array(decode_mask(instance['mask_rle'], [w, h]), dtype=bool).reshape(h, w)
                union |= mask  # All returned classes/confidences can contradict source clearing.
                area = int(mask.sum())
                if instance['label'] == policy['label'] and area and (mask & destination).any():
                    targets.append((instance, mask, area))
            require(len(targets) == 1, 'target_missing_or_ambiguous_at_checkpoint')
            instance, mask, area = targets[0]
            require(instance['confidence'] >= policy['min_confidence'], 'uncertain_checkpoint_target')
            require(policy['min_target_area_px'] <= area <= policy['max_target_area_px'],
                    'unexpected_checkpoint_target_size')
            require(int((mask & destination).sum()) / area >= policy['min_target_roi_fraction'],
                    'target_outside_checkpoint_roi')
            erased = 1 - int((union & before).sum()) / int(before.sum())
            # New/replacement objects anywhere in source ROI also block clearance.
            require(erased >= policy['min_source_erased_fraction'] and not (union & source_roi).any(),
                    'source_not_clear')
            ys, xs = np.nonzero(mask)
            centers.append((float(xs.mean()), float(ys.mean())))
            masks.append(mask)
            difference = np.abs(image.astype(np.int16) - reference.astype(np.int16)).sum(axis=2)
            changed = difference > policy['rgb_difference_threshold']
            changes.append(changed & mask & destination)
            out['frames'].append(dict(frame_id=frame_id, capture_started_ms=captured,
                                      source_erased_fraction=erased, target_area_px=area,
                                      source_rgb_changed_fraction=float(changed[before].mean()),
                                      static_patches=layout))
        require(max(math.dist(a, b) for a in centers for b in centers) <=
                policy['max_center_jitter_px'], 'target_unstable_or_slipping')
        persistent = np.logical_and.reduce(masks)
        total = np.logical_or.reduce(masks)
        out['persistent_target_fraction'] = float(persistent.sum() / total.sum())
        require(out['persistent_target_fraction'] >= policy['min_persistent_fraction'],
                'target_mask_unstable')
        if stage == 'placed':
            gain = int(np.logical_and.reduce(changes).sum())
            out['persistent_bin_gain_px'] = gain
            require(gain >= policy['min_bin_gain_px'], 'no_persistent_new_bin_target')
        out['status'] = 'visual_evidence_pass'
    except (ValueError, KeyError, TypeError, IndexError, AttributeError, cv2.error) as exc:
        out['reasons'].append(str(exc) if isinstance(exc, ValueError) else 'malformed_checkpoint_input')
    return out


def load_image(entry, base):
    path = (base / entry['path']).resolve()
    require(hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], 'image_hash_mismatch')
    image = cv2.imread(str(path))
    require(image is not None, 'unreadable_image')
    return image


def review_recording(document, base):
    """Replay only. now_ms is the recorded review clock, never today's clock."""
    require(document['schema_version'] == 'static-visual-checkpoint-v1', 'unsupported_schema')
    reference = load_image(document['reference_image'], base)
    policy, before = document['policy'], document['before_observation']
    validate_policy(policy)
    require(before['image_sha256'] == document['reference_image']['sha256'], 'reference_mask_image_binding_mismatch')
    for key in ('task_id', 'camera_id', 'model_version', 'calibration_id', 'bin_id', 'image_size'):
        require(before[key] == policy[key], 'reference_identity_mismatch:' + key)
    require(before['stage'] == 'before' and before['frame_id'] == policy['reference_frame_id']
            and before['capture_started_ms'] == policy['reference_capture_started_ms'],
            'reference_frame_mismatch')
    require(len(before['instances']) == 1 and before['instances'][0]['label'] == policy['label']
            and number(before['instances'][0]['confidence']) and
            policy['min_confidence'] <= before['instances'][0]['confidence'] <= 1 and
            before['instances'][0]['mask_rle'] == document['before_mask_rle'],
            'invalid_before_semantic_target')
    frames = []
    for row in document['frames']:
        require(row['observation']['image_sha256'] == row['image']['sha256'], 'mask_image_binding_mismatch')
        frames.append(dict(image=load_image(row['image'], base), observation=row['observation']))
    return review_checkpoint(reference, document['before_mask_rle'], frames, document['policy'],
                             document['stage'], document['recorded_review_ms'])


def main():
    parser = argparse.ArgumentParser(description='Replay recorded visual checkpoint evidence; never opens devices.')
    parser.add_argument('--recording', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = review_recording(json.loads(args.recording.read_text(encoding='utf-8-sig')),
                                  args.recording.resolve().parent)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        result = dict(status='needs_review', reasons=[str(exc)], automatic_motion_allowed=False,
                      automatic_retry=False, physical_success_verified=False,
                      independently_validated=False)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result['status'] == 'visual_evidence_pass' else 2


if __name__ == '__main__':
    raise SystemExit(main())
