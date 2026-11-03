"""Guards for a bounded, taught-path camera pilot. No device imports here."""
import copy
import math
import re

from .arm_timing import move_ms, settle_s, validate_timing
from .static_mask_contract import decode_mask, geometry


def check_observations(observations, policy, now_ms):
    """Require fresh, consistent semantic observations inside a taught image ROI.

    This gate does not map image coordinates to new robot poses. A physical
    path and its compatible object pose must be verified independently.
    """
    if len(observations) < policy['stable_frames']:
        raise ValueError('insufficient_stable_frames')
    selected = observations[-policy['stable_frames']:]
    centers = []
    last_frame = -1
    for row in selected:
        if (row['model_version'] != policy['model_version'] or
                row['camera_id'] != policy['camera_id'] or
                row['image_size'] != policy['image_size']):
            raise ValueError('observation_identity_mismatch')
        if type(row['frame_id']) is not int or row['frame_id'] <= last_frame:
            raise ValueError('non_increasing_frame')
        last_frame = row['frame_id']
        age = now_ms - row['capture_started_ms']
        if not math.isfinite(age) or not 0 <= age <= policy['max_age_ms']:
            raise ValueError('stale_observation')
        if len(row['instances']) != 1:
            raise ValueError('ambiguous_scene')
        obj = row['instances'][0]
        confidence = obj['confidence']
        if (obj['label'] != policy['label'] or type(confidence) not in (int, float)
                or not math.isfinite(confidence) or not policy['min_confidence'] <= confidence <= 1):
            raise ValueError('wrong_or_uncertain_class')
        mask = decode_mask(obj['mask_rle'], row['image_size'])
        stats = geometry(mask, row['image_size'], 0)
        if not policy['min_area_px'] <= stats['m00'] <= policy['max_area_px']:
            raise ValueError('unexpected_object_size')
        box = [stats['min_col'], stats['min_row'], stats['max_col']+1, stats['max_row']+1]
        if obj['bbox_xyxy'] != box:
            raise ValueError('bbox_mask_mismatch')
        x0,y0,x1,y1 = policy['source_roi_xyxy']
        if not x0 <= box[0] < box[2] <= x1 or not y0 <= box[1] < box[3] <= y1:
            raise ValueError('outside_taught_source_roi')
        cx, cy = stats['m10']/stats['m00'], stats['m01']/stats['m00']
        expected = policy['reference_center_px']
        if math.hypot(cx-expected[0], cy-expected[1]) > policy['max_reference_shift_px']:
            raise ValueError('outside_taught_pick_position')
        centers.append((cx,cy))
    if any(math.dist(a,b) > policy['max_stable_shift_px'] for a in centers for b in centers):
        raise ValueError('moving_target')
    return dict(status='semantic_gate_passed', label=policy['label'], frames=len(selected),
                center_px=list(centers[-1]), model_version=policy['model_version'],
                physical_success_verified=False)


def retime_taught_plan(plan, timing):
    """Apply PR36 timing to actual PR35 frames, preserving all servo positions.

    Each step must explicitly name its timing_step from arm_timing.PICK_STEPS.
    The supervised runner's 1500 ms minimum remains in force.
    """
    validate_timing(timing)
    result = copy.deepcopy(plan)
    total = 0
    for stage in result['stages']:
        for step in stage['steps']:
            name = step['timing_step']
            from .arm_timing import PICK_STEPS
            if name not in PICK_STEPS:
                raise ValueError('unknown_timing_step')
            duration = move_ms(timing, name)
            if duration < 1500:
                raise ValueError('supervised_path_requires_at_least_1500_ms')
            step['command'] = re.sub(r'T\d{4}!', 'T%04d!' % duration, step['command'])
            step['duration_ms'] = duration
            step['extra_settle_s'] = settle_s(timing, name)
            total += duration/1000 + step['extra_settle_s']
    return result, dict(motion_and_settle_s=total, physical_speed_verified=False)
