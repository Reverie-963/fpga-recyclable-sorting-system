"""Bounded, position-dependent carton grasp previews; never controls hardware."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from .grasp_geometry import fuse, overlap
from .static_mask_contract import decode_mask


def need(value, reason):
    if not value:
        raise ValueError(reason)


def project(matrix, point):
    h = np.asarray(matrix, dtype=float)
    need(h.shape == (3, 3) and np.isfinite(h).all() and abs(np.linalg.det(h)) > 1e-10,
         'invalid_homography')
    v = h @ np.r_[point, 1.]
    need(np.isfinite(v).all() and abs(v[2]) > 1e-9, 'point_at_infinity')
    return (v[:2] / v[2]).tolist()


def validate_calibration(calibration, policy):
    """Reject missing coverage rather than silently accepting every point."""
    need(calibration['checked_for_supervised_preview'] is True, 'calibration_checks_pending')
    height = policy.get('surface_height_mm', 65)
    need(type(height) in (int, float) and np.isfinite(height) and height > 0,
         'invalid_surface_height')
    need(calibration['surface_height_mm'] == height, 'surface_height_mismatch')
    # Legacy files describe the one65mm carton. A different height needs an
    # explicit object binding and robot mapping checked at that actual height.
    if height != 65 or 'object_profile_id' in policy:
        profile = policy.get('object_profile_id')
        need(isinstance(profile, str) and bool(profile.strip()) and
             calibration.get('object_profile_id') == profile, 'object_profile_mismatch')
        need(calibration.get('robot_mapping_height_mm') == height,
             'robot_mapping_height_not_checked')
        for key in ('grip_model_z_mm', 'hover_model_z_mm', 'working_alpha_deg'):
            value = calibration.get(key)
            need(type(value) in (int, float) and np.isfinite(value), 'missing_height_pose')
        need(0 <= calibration['grip_model_z_mm'] < calibration['hover_model_z_mm'] and
             -135 <= calibration['working_alpha_deg'] <= 0, 'invalid_height_pose')
    need(calibration['camera_id'] == policy['camera_id'] and
         calibration['image_size'] == policy['image_size'], 'calibration_camera_or_size_changed')
    hulls = calibration['verified_point_hulls_card_xy_mm']
    need(isinstance(hulls, list) and len(hulls) >= 2, 'missing_height_checked_regions')
    for hull in hulls:
        polygon = np.asarray(hull, np.float32)
        need(polygon.ndim == 2 and polygon.shape[1] == 2 and len(polygon) >= 3 and
             np.isfinite(polygon).all() and cv2.contourArea(polygon) > 0 and
             cv2.isContourConvex(polygon), 'invalid_checked_region')
    for key in ('surface_image_to_card', 'grip_card_to_control',
                'hover_card_to_control_left', 'hover_card_to_control_right'):
        project(calibration[key], [0, 0])


def stable_interior(mask, choice, minimum, fraction, image_to_card=None, checked_hulls=None):
    need(.5 <= fraction <= 1, 'invalid_clearance_fraction')
    x, y, w, h = choice['bbox']
    crop = (mask[y:y+h, x:x+w] != 0).astype(np.uint8)
    _, labels = cv2.connectedComponents(crop, connectivity=8)
    px, py = map(round, choice['center'])
    own = int(labels[py-y, px-x]); need(own > 0, 'missing_owned_component')
    component = (labels == own).astype(np.uint8)
    moments = cv2.moments(component, binaryImage=True)
    cx, cy = moments['m10']/moments['m00'], moments['m01']/moments['m00']
    distance = cv2.distanceTransform(np.pad(component, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    peak = float(distance.max())
    ys, xs = np.where(distance >= max(minimum, peak*fraction)-1e-6)
    need(len(xs) > 0, 'no_safe_interior_band')
    if image_to_card is not None:
        project(image_to_card, [float(cx+x), float(cy+y)])
        hmat = np.asarray(image_to_card, float)
        world = hmat @ np.stack((xs+x, ys+y, np.ones(len(xs))))
        need(np.isfinite(world).all() and np.all(np.abs(world[2]) > 1e-9), 'invalid_band_projection')
        world = (world[:2]/world[2]).T
        keep = np.ones(len(xs), bool)
        for hull in checked_hulls:
            polygon = np.asarray(hull, np.float32)
            keep &= np.array([cv2.pointPolygonTest(polygon, tuple(map(float, p)), False) >= 0 for p in world])
        need(keep.any(), 'point_outside_checked_region:no_safe_interior_band')
        xs, ys = xs[keep], ys[keep]
    i = np.argmin((xs-cx)**2+(ys-cy)**2)
    return [float(xs[i]+x), float(ys[i]+y)], float(distance[ys[i], xs[i]]), peak


def plan_target(rows, policy, calibration, now_ms):
    validate_calibration(calibration, policy)
    need(type(policy['stable_frames']) is int and policy['stable_frames'] >= 2,
         'invalid_stable_frame_count')
    need(np.isfinite(now_ms), 'invalid_clock')
    need(len(rows) == policy['stable_frames'], 'wrong_frame_count')
    candidates, previous_id, previous_capture = [], -1, -1
    for row in rows:
        need(row['model_version'] == policy['model_version'], 'wrong_model')
        need(row['camera_id'] == policy['camera_id'] and row['image_size'] == policy['image_size'],
             'camera_or_size_changed')
        need(type(row['frame_id']) is int and row['frame_id'] > previous_id, 'duplicate_or_old_frame')
        stamp = row['capture_started_ms']
        need(isinstance(stamp, (int, float)) and np.isfinite(stamp) and
             previous_capture < stamp <= now_ms and now_ms - stamp <= policy['max_age_ms'],
             'stale_or_invalid_capture')
        previous_id, previous_capture = row['frame_id'], stamp
        items = row.get('all_full_frame_instances', row['instances'])
        source = [d for d in items if overlap(d['bbox_xyxy'], policy['source_roi_xyxy']) > 0]
        need(len(source) == 1, 'source_not_single_object')
        d = source[0]
        need(d['label'] == policy['label'] and np.isfinite(d['confidence']) and
             policy['min_confidence'] <= d['confidence'] <= 1, 'unsupported_or_weak_target')
        width, height = row['image_size']
        mask = np.asarray(decode_mask(d['mask_rle'], row['image_size']), np.uint8).reshape(height, width)
        ys, xs = np.nonzero(mask)
        need(policy['min_area_px'] <= len(xs) <= policy['max_area_px'], 'invalid_target_area')
        box = [int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1]
        need(box == d['bbox_xyxy'], 'mask_bbox_mismatch')
        a, b, c, e = policy['source_roi_xyxy']
        need(a <= box[0] < box[2] <= c and b <= box[1] < box[3] <= e, 'partial_source_object')
        need((box[3]-box[1]) / (box[2]-box[0]) >= 1.5, 'unsupported_carton_orientation')
        cfg = {'grasp': {'confidence': policy['min_confidence'], 'min_box_coverage': .9,
                         'clearance_px': 3, 'min_interior_px': policy['min_interior_px']},
               'vision': {'min_area': policy['min_area_px'], 'max_area': policy['max_area_px']},
               'arm': {'bins': {'can': None, 'plastic': None, 'cardboard': None}}}
        choices = fuse([{'label': d['label'], 'confidence': d['confidence'], 'xyxy': box}],
                       mask * 255, cfg, None)
        need(len(choices) == 1, 'no_unique_interior_grasp')
        choice = choices[0]
        point, clearance, peak = stable_interior(mask, choice, policy['min_interior_px'],
                                                   policy.get('interior_clearance_fraction', .8),
                                                   calibration['surface_image_to_card'],
                                                   calibration['verified_point_hulls_card_xy_mm'])
        card = project(calibration['surface_image_to_card'], point)
        for hull in calibration['verified_point_hulls_card_xy_mm']:
            need(cv2.pointPolygonTest(np.asarray(hull, np.float32), tuple(card), False) >= 0,
                 'point_outside_checked_region')
        grip = project(calibration['grip_card_to_control'], card)
        hover_matrix = calibration['hover_card_to_control_left' if card[0] < 0 else 'hover_card_to_control_right']
        hover = project(hover_matrix, card)
        candidates.append({'frame_id': row['frame_id'], 'pick_pixel': point, 'card_xy_mm': card,
                           'grip_xyz_nominal': grip+[calibration.get('grip_model_z_mm', 50)],
                           'hover_xyz_nominal': hover+[calibration.get('hover_model_z_mm', 70)],
                           'edge_clearance_px': clearance, 'maximum_clearance_px': peak,
                           'strict_maximum_pixel': choice['center'], 'confidence': d['confidence']})
    need(now_ms - rows[-1]['capture_started_ms'] <= policy['latest_max_age_ms'], 'latest_frame_stale')
    points = np.asarray([c['pick_pixel'] for c in candidates])
    need(float(np.linalg.norm(points[:, None]-points[None, :], axis=2).max()) <=
         policy['max_stable_shift_px'], 'target_moving')
    return {'status': 'coordinate_preview_for_supervised_trial', 'point_algorithm': 'centroid_near_safe_interior_band_with_checked_region_v3', 'frames': candidates,
            'target': candidates[-1], 'camera_id': policy['camera_id'],
            'model_sha256': policy['model_version'], 'physical_motion': False,
            'camera_pose_verified': False, 'motion_plan_generated': False,
            'surface_height_mm': calibration['surface_height_mm'],
            'object_profile_id': policy.get('object_profile_id', 'legacy_65mm_carton'),
            'working_alpha_deg': calibration.get('working_alpha_deg', -75),
            'robot_precision_accepted': False, 'unattended_execution_allowed': False,
            'scope': 'Bound carton height/profile and orientation; checked interior point only. Full silhouette and collision-free grasp not certified.',
            'mandatory_next': 'Fresh-scene check and supervised open-gripper approach; observe before descending or gripping.'}


def draw_overlay(image, calibration, result):
    """Show the intersection of checked point domains at both control heights."""
    need(image is not None and list(image.shape[:2][::-1]) == calibration['image_size'],
         'overlay_image_size_changed')
    inverse = np.linalg.inv(np.asarray(calibration['surface_image_to_card'], float))
    allowed = np.ones(image.shape[:2], np.uint8)
    for hull in calibration['verified_point_hulls_card_xy_mm']:
        polygon = np.rint([project(inverse, p) for p in hull]).astype(np.int32)
        layer = np.zeros(image.shape[:2], np.uint8)
        cv2.fillConvexPoly(layer, polygon, 1)
        allowed &= layer
    output = image.copy()
    tint = output.copy()
    tint[allowed != 0] = (0, 210, 0)
    output = cv2.addWeighted(output, .7, tint, .3, 0)
    contours, _ = cv2.findContours(allowed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, (0, 210, 0), 2)
    if 'target' in result:
        point = tuple(map(round, result['target']['pick_pixel']))
        cv2.drawMarker(output, point, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
    cv2.putText(output, 'Recorded preview only - no motion', (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1, cv2.LINE_AA)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observations', required=True,
                        help='JSON containing observations and policy')
    parser.add_argument('--calibration', required=True)
    parser.add_argument('--replay-now-ms', type=float, required=True,
                        help='Recorded decision time, in the capture clock domain')
    parser.add_argument('--output', required=True)
    parser.add_argument('--image', help='Optional last recorded image for an overlay')
    parser.add_argument('--overlay', help='Overlay output; requires --image')
    args = parser.parse_args()
    if bool(args.image) != bool(args.overlay):
        parser.error('--image and --overlay must be supplied together')
    def read(path):
        return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    data, calibration = read(args.observations), read(args.calibration)
    validate_calibration(calibration, data['policy'])
    try:
        result = plan_target(data['observations'], data['policy'], calibration, args.replay_now_ms)
    except ValueError as exc:
        result = {'status': 'rejected_coordinate_preview', 'reason': str(exc)}
    result.update(recorded_replay_only=True, physical_motion=False,
                  unattended_execution_allowed=False, camera_pose_verified=False,
                  replay_now_ms=args.replay_now_ms,
                  input_sha256={name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                for name, path in [('observations', args.observations),
                                                   ('calibration', args.calibration)]})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    if args.image:
        expected = data['observations'][-1].get('image_sha256')
        need(expected is not None and hashlib.sha256(Path(args.image).read_bytes()).hexdigest() == expected,
             'overlay_image_hash_mismatch')
        overlay = draw_overlay(cv2.imread(args.image), calibration, result)
        target = Path(args.overlay)
        target.parent.mkdir(parents=True, exist_ok=True)
        need(cv2.imwrite(str(target), overlay), 'overlay_write_failed')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
