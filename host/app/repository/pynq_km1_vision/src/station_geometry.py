"""Offline camera relocation and height-plane geometry. Never enables motion."""
import hashlib
import json
import math

import cv2
import numpy as np


def need(ok, message):
    if not ok:
        raise ValueError(message)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def array(value, shape):
    result = np.asarray(value, dtype=float)
    need(result.shape == shape and np.isfinite(result).all(), 'invalid_numeric_shape')
    return result


def polygon(value):
    p = np.asarray(value, dtype=np.float32)
    need(p.ndim == 2 and p.shape[1] == 2 and len(p) >= 3 and np.isfinite(p).all()
         and cv2.isContourConvex(p) and cv2.contourArea(p) > 1, 'invalid_world_polygon')
    return p


def camera_parameters(camera):
    need(camera.get('kind') == 'station-camera-v1', 'wrong_camera_kind')
    need(all(isinstance(camera.get(k), str) and camera[k].strip()
             for k in ('camera_id', 'lens_profile_id')), 'camera_identity_required')
    size = camera['image_size']
    need(isinstance(size, list) and len(size) == 2 and
         all(type(n) is int and n > 0 for n in size), 'invalid_image_size')
    k = array(camera['camera_matrix'], (3, 3))
    need(k[0, 0] > 0 and k[1, 1] > 0 and np.allclose(k[2], [0, 0, 1])
         and abs(k[0, 1])+abs(k[1, 0]) < 1e-9
         and 0 <= k[0, 2] < size[0] and 0 <= k[1, 2] < size[1], 'invalid_camera_matrix')
    d = np.asarray(camera['dist_coeffs'], dtype=float)
    need(d.ndim == 1 and len(d) in (4, 5, 8, 12, 14) and np.isfinite(d).all(), 'invalid_distortion')
    return k, d


def station_parameters(station):
    need(station.get('kind') == 'station-markers-v1' and
         isinstance(station.get('world_frame_id'), str) and station['world_frame_id'].strip(),
         'world_frame_required')
    need(station.get('dictionary') == 'DICT_4X4_50', 'unsupported_marker_dictionary')
    markers = station['markers']
    need(isinstance(markers, list) and len(markers) >= 3, 'at_least_three_fixed_markers_required')
    found = {}
    for marker in markers:
        sid = marker['id']
        need(type(sid) is int and 0 <= sid < 50 and sid not in found, 'duplicate_or_invalid_marker_id')
        points = array(marker['corners_world_mm'], (4, 3))
        need(np.allclose(points[:, 2], 0), 'markers_must_share_world_z_zero')
        edges = np.roll(points, -1, axis=0)-points
        lengths = np.linalg.norm(edges, axis=1)
        cross_z = edges[0, 0]*edges[1, 1]-edges[0, 1]*edges[1, 0]
        need(lengths.min() > 5 and np.allclose(lengths, lengths[0], atol=.01)
             and abs(np.dot(edges[0], edges[1])) < .01 and cross_z < 0,
             'marker_corners_must_be_ordered_square_with_world_z_up')
        found[sid] = points
    polygon(station['work_polygon_world_xy_mm'])
    heights = array(station['height_range_mm'], (2,))
    need(0 <= heights[0] < heights[1], 'invalid_declared_height_range')
    return found


def angle_between(a, b):
    return math.degrees(math.acos(float(np.clip((np.trace(a @ b.T)-1)/2, -1, 1))))


def detect_markers(image, camera):
    camera_parameters(camera)
    need(list(image.shape[:2][::-1]) == camera['image_size'], 'image_size_changed')
    need(hasattr(cv2, 'aruco') and hasattr(cv2.aruco, 'ArucoDetector'), 'opencv_aruco_required')
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    settings = cv2.aruco.DetectorParameters()
    settings.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, _ = cv2.aruco.ArucoDetector(dictionary, settings).detectMarkers(image)
    return [] if ids is None else [dict(id=int(sid), corners_px=c.reshape(4, 2).tolist())
                                   for sid, c in zip(ids.ravel(), corners)]


def estimate_pose(detections, camera, station, min_fixed_markers=3,review_factor=1):
    from .sorting_review_policy import checked_factor
    factor=checked_factor(review_factor)
    need(type(min_fixed_markers) is int and 2<=min_fixed_markers<=4,'invalid_min_fixed_markers')
    k, d = camera_parameters(camera)
    markers = station_parameters(station)
    obj, pix, ids, seen = [], [], [], set()
    width, height = camera['image_size']
    for detection in detections:
        sid = detection['id']
        need(type(sid) is int and sid not in seen, 'duplicate_detection_id')
        seen.add(sid)
        if sid not in markers:
            continue
        p = array(detection['corners_px'], (4, 2))
        need((p[:, 0] >= 0).all() and (p[:, 0] < width).all() and
             (p[:, 1] >= 0).all() and (p[:, 1] < height).all(), 'marker_outside_image')
        need(np.linalg.norm(np.roll(p, -1, axis=0)-p, axis=1).min() >= 12/factor, 'marker_too_small')
        obj.extend(markers[sid]); pix.extend(p); ids.append(sid)
    need(len(ids) >= min_fixed_markers, 'not_enough_visible_known_markers')
    obj, pix = np.asarray(obj, np.float64), np.asarray(pix, np.float64)
    need(cv2.contourArea(cv2.convexHull(pix.astype(np.float32))) >= 800/factor, 'insufficient_marker_coverage')
    solutions = cv2.solvePnPGeneric(obj, pix, k, d, flags=cv2.SOLVEPNP_IPPE)
    candidates = []
    for rvec, tvec in zip(solutions[1], solutions[2]):
        rotation = cv2.Rodrigues(rvec)[0]
        center = (-rotation.T @ tvec).ravel()
        if center[2] <= station['height_range_mm'][1] or ((rotation @ obj.T+tvec)[2] <= 0).any():
            continue
        predicted = cv2.projectPoints(obj, rvec, tvec, k, d)[0].reshape(-1, 2)
        errors = np.linalg.norm(predicted-pix, axis=1)
        if not np.isfinite(errors).all():
            continue
        candidates.append(dict(rotation=rotation, t=tvec.ravel(), center=center,
            rms=float(np.sqrt(np.mean(errors**2))), maximum=float(errors.max())))
    need(candidates, 'no_above_table_pose')
    candidates.sort(key=lambda c: c['rms'])
    best = candidates[0]
    need(best['rms'] <= .8*factor and best['maximum'] <= 1.5*factor, 'marker_reprojection_error')
    if len(candidates) > 1:
        other = candidates[1]
        different = np.linalg.norm(best['center']-other['center']) > 2 or angle_between(best['rotation'], other['rotation']) > .2
        need(not (different and other['rms']-best['rms'] < .05/factor), 'ambiguous_planar_pose')
    rvec, tvec = cv2.solvePnPRefineLM(obj, pix, k, d, cv2.Rodrigues(best['rotation'])[0], best['t'].reshape(3, 1))
    rotation = cv2.Rodrigues(rvec)[0]
    center = (-rotation.T @ tvec).ravel()
    errors = np.linalg.norm(cv2.projectPoints(obj, rvec, tvec, k, d)[0].reshape(-1, 2)-pix, axis=1)
    need(center[2] > station['height_range_mm'][1] and ((rotation @ obj.T+tvec)[2] > 0).all()
         and np.isfinite(errors).all() and np.sqrt(np.mean(errors**2)) <= .8*factor and errors.max() <= 1.5*factor,
         'refined_pose_invalid')
    best.update(rotation=rotation, t=tvec.ravel(), center=center,
                rms=float(np.sqrt(np.mean(errors**2))), maximum=float(errors.max()))
    return dict(rotation_world_to_camera=best['rotation'].tolist(),
        translation_world_to_camera_mm=best['t'].tolist(), camera_center_world_mm=best['center'].tolist(),
        marker_ids=ids, reprojection_rms_px=best['rms'], reprojection_max_px=best['maximum'])


def localize_records(records, camera, station, stability_mode='pose', stability_points_world_mm=None,min_fixed_markers=3,max_marker_shift_px=.5,review_factor=1):
    from .sorting_review_policy import checked_factor
    factor=checked_factor(review_factor)
    need(type(max_marker_shift_px) in (int,float) and np.isfinite(max_marker_shift_px) and
         .5<=max_marker_shift_px<=5*factor,'invalid_initial_marker_shift_limit')
    need(len(records) >= 3, 'need_three_stable_recorded_frames')
    hashes = [r['image_sha256'] for r in records]
    need(len(set(hashes)) == len(hashes) and all(isinstance(h, str) and len(h) == 64 for h in hashes),
         'duplicate_or_invalid_recorded_images')
    def fit(detections):
        if factor==1:return estimate_pose(detections,camera,station,min_fixed_markers)
        return estimate_pose(detections,camera,station,min_fixed_markers,review_factor=factor)
    poses = [fit(r['detections']) for r in records]
    need(stability_mode in ('pose', 'fixed_marker_task_space'), 'unknown_pose_stability_mode')
    final_pose = poses[-1]
    quality = dict(mode=stability_mode,minimum_fixed_markers=min_fixed_markers)
    for a in poses:
        for b in poses:
            need(angle_between(np.asarray(a['rotation_world_to_camera']), np.asarray(b['rotation_world_to_camera'])) <= .5*factor,
                 'camera_not_stable')
            if stability_mode == 'pose':
                need(math.dist(a['camera_center_world_mm'], b['camera_center_world_mm']) <= 2*factor, 'camera_not_stable')
    if stability_mode == 'fixed_marker_task_space':
        # Small planar markers can have coupled rotation/translation noise:
        # the fitted camera centre moves while the actual image and projected
        # working points remain stable. Gate those observed/task quantities
        # explicitly; never use this as an in-motion permission.
        known = station_parameters(station)
        frames = [{d['id']:array(d['corners_px'], (4, 2)) for d in row['detections'] if d['id'] in known}
                  for row in records]
        common = sorted(set.intersection(*(set(frame) for frame in frames)))
        need(len(common) >= min_fixed_markers, 'two_common_fixed_markers_required' if min_fixed_markers==2 else 'three_common_fixed_markers_required')
        shift = max(float(np.linalg.norm(a[sid]-b[sid], axis=1).max())
                    for a in frames for b in frames for sid in common)
        need(shift <= max_marker_shift_px, 'fixed_markers_moved_over_half_pixel' if max_marker_shift_px==.5
             else 'fixed_markers_moved_over_initial_limit')
        k, d = camera_parameters(camera)
        xy = polygon(station['work_polygon_world_xy_mm']).astype(float)
        xy = np.vstack((xy, xy.mean(0)))
        anchors = np.vstack([np.c_[xy, np.full(len(xy), z)] for z in station['height_range_mm']])
        if stability_points_world_mm is not None:
            extra = np.asarray(stability_points_world_mm, float)
            need(extra.ndim == 2 and extra.shape[1] == 3 and len(extra) > 0 and np.isfinite(extra).all()
                 and (extra[:, 2] >= station['height_range_mm'][0]).all()
                 and (extra[:, 2] <= station['height_range_mm'][1]).all(), 'invalid_stability_points')
            anchors = np.vstack((anchors, extra))
        reference = poses[-1]
        reference_r = np.asarray(reference['rotation_world_to_camera'])
        pixels = cv2.projectPoints(anchors, cv2.Rodrigues(reference_r)[0],
            np.asarray(reference['translation_world_to_camera_mm']), k, d)[0]
        rays = np.c_[cv2.undistortPoints(pixels, k, d).reshape(-1, 2), np.ones(len(anchors))]
        averaged = [dict(id=sid, corners_px=np.mean([f[sid] for f in frames], axis=0).tolist()) for sid in common]
        # For two planar tags the extra IPPE solve on averaged, nearly exact
        # corners can be numerically singular. Keep the already validated last
        # pose; every recorded pose still has to meet the same spread checks.
        final_pose = poses[-1] if len(common)==2 else fit(averaged)
        projections = []
        for pose in poses+[final_pose]:
            r = np.asarray(pose['rotation_world_to_camera']); center = np.asarray(pose['camera_center_world_mm'])
            directions = (r.T @ rays.T).T
            need(np.all(directions[:, 2] < -.1), 'unstable_task_plane_rays')
            scale = (anchors[:, 2]-center[2])/directions[:, 2]
            predicted = center+scale[:, None]*directions
            projections.append(predicted[:, :2])
        spread = max(float(np.linalg.norm(a-b, axis=1).max()) for a in projections for b in projections)
        need(spread <= factor, 'fixed_marker_task_points_unstable_over_1mm' if factor==1
             else 'fixed_marker_task_points_unstable_over_limit')
        quality.update(max_observed_marker_shift_px=shift, max_marker_shift_px=max_marker_shift_px,
            max_task_point_spread_mm=factor,review_tolerance_factor=factor,
            max_height_plane_point_spread_mm=spread,
            observed_camera_center_spread_mm=max(math.dist(a['camera_center_world_mm'], b['camera_center_world_mm'])
                                                for a in poses for b in poses),
            evaluated_height_range_mm=station['height_range_mm'],
            fit='last_validated_two_marker_pose' if len(common)==2 else 'mean_common_marker_corners')
        quality['evaluated_task_point_count'] = len(anchors)
    return dict(kind='station-pose-preview-v1', **final_pose,
        camera_sha256=fingerprint(camera), station_sha256=fingerprint(station),
        camera_id=camera['camera_id'], lens_profile_id=camera['lens_profile_id'],
        image_size=camera['image_size'], world_frame_id=station['world_frame_id'],
        evidence=records, per_frame_pose=poses, stability_quality=quality, recorded_replay_only=True,
        automatic_motion_allowed=False, physical_accuracy_verified=False)


def pose_parameters(pose, camera, station):
    k, d = camera_parameters(camera)
    station_parameters(station)
    need(pose.get('kind') == 'station-pose-preview-v1' and
         pose['camera_sha256'] == fingerprint(camera) and pose['station_sha256'] == fingerprint(station),
         'pose_camera_or_station_changed')
    r = array(pose['rotation_world_to_camera'], (3, 3))
    need(np.allclose(r @ r.T, np.eye(3), atol=1e-6) and np.linalg.det(r) > .999, 'invalid_rotation')
    t = array(pose['translation_world_to_camera_mm'], (3,))
    return k, d, r, t


def registration_ray_point(pixel, height_mm, pose, camera, station):
    """Offline geometry for registration targets, including outside pick area.

    This cannot authorize robot reachability. Normal picking uses the separate
    pixel_at_height gate below, which retains its work-polygon check.
    """
    k, d, r, t = pose_parameters(pose, camera, station)
    need(type(height_mm) in (int, float) and math.isfinite(height_mm) and
         station['height_range_mm'][0] <= height_mm <= station['height_range_mm'][1], 'height_outside_declared_range')
    pixel = array(pixel, (2,))
    need(0 <= pixel[0] < camera['image_size'][0] and 0 <= pixel[1] < camera['image_size'][1], 'pixel_outside_image')
    normalized = cv2.undistortPoints(pixel.reshape(1, 1, 2), k, d).ravel()
    direction = r.T @ np.r_[normalized, 1.]
    center = -r.T @ t
    need(center[2] > height_mm and direction[2]/np.linalg.norm(direction) < -.1, 'ray_not_toward_height_plane')
    scale = (height_mm-center[2])/direction[2]
    point = center+scale*direction
    need(scale > 0, 'intersection_behind_camera')
    return dict(world_xyz_mm=point.tolist(), height_source='operator_supplied',
        surface_height_mm=height_mm, world_frame_id=station['world_frame_id'],
        pose_sha256=fingerprint(pose), automatic_motion_allowed=False,
        note='Geometric surface point, not robot model Z or a collision-checked grasp')


def pixel_at_height(pixel, height_mm, pose, camera, station):
    result = registration_ray_point(pixel, height_mm, pose, camera, station)
    need(cv2.pointPolygonTest(polygon(station['work_polygon_world_xy_mm']), tuple(result['world_xyz_mm'][:2]), False) >= 0,
         'point_outside_work_polygon')
    return result


def world_to_pixels(points, pose, camera, station):
    k, d, r, t = pose_parameters(pose, camera, station)
    p = np.asarray(points, dtype=float)
    need(p.ndim == 2 and p.shape[1] == 3 and np.isfinite(p).all(), 'invalid_world_points')
    need(((r @ p.T).T+t)[:, 2].min() > 0, 'world_points_behind_camera')
    return cv2.projectPoints(p, cv2.Rodrigues(r)[0], t, k, d)[0].reshape(-1, 2).tolist()


def homography(value):
    h = array(value, (3, 3))
    need(abs(np.linalg.det(h)) > 1e-12, 'singular_mapping')
    return h


def map_xy(matrix, xy):
    value = homography(matrix) @ np.r_[array(xy, (2,)), 1.]
    need(abs(value[2]) > 1e-9, 'mapping_at_infinity')
    return value[:2]/value[2]


def import_legacy(source, source_sha256, profile_id):
    """Separate height-specific robot mappings from obsolete camera homographies."""
    need(isinstance(profile_id, str) and profile_id.strip(), 'profile_id_required')
    need(isinstance(source_sha256, str) and len(source_sha256) == 64 and
         all(c in '0123456789abcdef' for c in source_sha256), 'legacy_source_hash_required')
    height = source.get('object_height_mm', source.get('surface_height_mm'))
    need(type(height) in (int, float) and math.isfinite(height) and height > 0, 'legacy_height_required')
    base = source.get('image_to_card_at_object_surface', source.get('surface_image_to_card'))
    def control(explicit, composite):
        if explicit is not None:
            return homography(explicit).tolist()
        need(base is not None and composite is not None, 'legacy_control_mapping_missing')
        h = homography(composite) @ np.linalg.inv(homography(base))
        return (h/h[2, 2]).tolist()
    grip = control(source.get('card_to_grip', source.get('grip_card_to_control')), source.get('image_to_grip_mm'))
    hover = control(source.get('card_to_hover', source.get('hover_card_to_control_left')), source.get('image_to_hover_mm'))
    # Two unequal hover maps cannot be silently collapsed to one.
    right = source.get('hover_card_to_control_right')
    need(right is None or np.allclose(homography(right), homography(hover)), 'piecewise_hover_requires_explicit_adapter')
    hulls = source.get('verified_pick_hulls_card_xy_mm', source.get('verified_point_hulls_card_xy_mm'))
    need(isinstance(hulls, list) and len(hulls) >= 2, 'legacy_checked_regions_required')
    for hull in hulls:
        polygon(hull)
    gz, hz = source['grip_model_z_mm'], source['hover_model_z_mm']
    need(all(type(v) in (int, float) and math.isfinite(v) for v in (gz, hz)) and 0 <= gz < hz,
         'legacy_model_heights_invalid')
    return dict(kind='station-legacy-profile-v1', profile_id=profile_id, object_height_mm=height,
        source_sha256=source_sha256, card_to_grip=grip, card_to_hover=hover,
        grip_model_z_mm=gz, hover_model_z_mm=hz, checked_card_polygons_mm=hulls,
        alpha_deg=source.get('working_alpha_deg'), world_registration=None,
        physical_precision_accepted=False, automatic_motion_allowed=False,
        note='Old image homographies are excluded. New marker-world to old card registration is still required.')


def legacy_control(point, profile):
    need(profile.get('kind') == 'station-legacy-profile-v1' and
         abs(point['surface_height_mm']-profile['object_height_mm']) < 1e-6,
         'legacy_profile_height_mismatch_no_interpolation')
    registration = profile.get('world_registration')
    need(isinstance(registration, dict) and registration.get('checked') is True and
         registration['world_frame_id'] == point['world_frame_id'], 'legacy_world_registration_required')
    h = homography(registration['world_to_card_xy'])
    need(np.allclose(h[2], [0, 0, 1]) and np.allclose(h[:2, :2].T @ h[:2, :2], np.eye(2), atol=1e-6),
         'registration_must_preserve_millimetre_scale')
    xy = map_xy(h, point['world_xyz_mm'][:2])
    need(all(cv2.pointPolygonTest(polygon(p), tuple(xy), False) >= 0 for p in profile['checked_card_polygons_mm']),
         'outside_legacy_checked_domain')
    grip, hover = map_xy(profile['card_to_grip'], xy), map_xy(profile['card_to_hover'], xy)
    return dict(profile_id=profile['profile_id'], card_xy_mm=xy.tolist(),
        grip_xyz_nominal=grip.tolist()+[profile['grip_model_z_mm']],
        hover_xyz_nominal=hover.tolist()+[profile['hover_model_z_mm']],
        automatic_motion_allowed=False, physical_precision_accepted=False)


def refit_recorded_plane(source, height_mm, square_mm=20):
    """Refit ONE recorded card image; never pool different camera placements."""
    need(type(height_mm) in (int, float) and math.isfinite(height_mm) and height_mm >= 0,
         'explicit_surface_height_required')
    need(type(square_mm) in (int, float) and math.isfinite(square_mm) and square_mm > 0,
         'positive_measured_square_size_required')
    known_height = source.get('actual_surface_height_mm')
    need(known_height is None or abs(known_height-height_mm) < 1e-6, 'recorded_height_binding_mismatch')
    corners = source['corner_pixels']
    need(source['card_origin'] == 'C2' and source['axes'] == '+X card right, +Y toward TOP',
         'legacy_card_axes_unknown')
    pixels, world, labels = [], [], []
    for row in range(1, 5):
        for col, letter in enumerate('ABCDE'):
            name = letter+str(row)
            need(name in corners, 'need_all_twenty_card_corners')
            pixels.append(array(corners[name], (2,)))
            world.append([(col-2)*square_mm, (2-row)*square_mm]); labels.append(name)
    pixels, world = np.asarray(pixels), np.asarray(world, float)
    # Fixed holdout selection before fitting; same-photo check, not new trials.
    checks = np.array([i in (1, 3, 6, 8, 11, 13, 16, 18) for i in range(20)])
    h = cv2.findHomography(pixels[~checks], world[~checks], 0)[0]
    need(h is not None, 'recorded_plane_fit_failed')
    residual = np.linalg.norm(cv2.perspectiveTransform(pixels[None], h)[0]-world, axis=1)
    need(np.isfinite(residual).all() and residual.max() <= 1, 'recorded_plane_residual_exceeds_1mm')
    return dict(kind='recorded-plane-refit-v1', surface_height_mm=height_mm,
        image_to_card=h.tolist(), fit_labels=[n for n, c in zip(labels, checks) if not c],
        pixel_hull=cv2.convexHull(pixels.astype(np.float32)).reshape(-1, 2).tolist(),
        card_domain_xy_mm=[[-2*square_mm, -2*square_mm], [2*square_mm, -2*square_mm],
                           [2*square_mm, square_mm], [-2*square_mm, square_mm]],
        check_labels=[n for n, c in zip(labels, checks) if c],
        fit_max_mm=float(residual[~checks].max()), check_max_mm=float(residual[checks].max()),
        source_image_sha256=source['image_sha256'], camera_id=source['camera_id'],
        image_size=source['image_size'], world_origin='C2', square_mm=square_mm,
        camera_pose_group='this_recorded_image_only', camera_intrinsics_estimated=False,
        independent_physical_validation=False, automatic_motion_allowed=False,
        note='Same-image held-out corners only. No cross-height or cross-camera interpolation and no robot accuracy claim.')


def fit_robot_affine(data):
    """Learn local XY correction from recorded pairs; keep human uncertainty."""
    card, control = np.asarray(data['card_xy_mm'], float), np.asarray(data['nominal_xy_mm'], float)
    need(card.shape == control.shape and card.ndim == 2 and card.shape[1] == 2 and len(card) >= 3
         and np.isfinite(card).all() and np.isfinite(control).all(), 'invalid_robot_training_pairs')
    x = np.c_[card, np.ones(len(card))]
    need(np.linalg.matrix_rank(x) == 3, 'robot_training_points_collinear')
    coeff = np.linalg.lstsq(x, control, rcond=None)[0]
    h = np.eye(3); h[:2] = coeff.T
    homography(h)
    checks, nominal = np.asarray(data['check_card_xy_mm'], float), np.asarray(data['check_nominal_xy_mm'], float)
    need(checks.shape == nominal.shape and checks.ndim == 2 and checks.shape[1] == 2 and len(checks) >= 1
         and np.isfinite(checks).all() and np.isfinite(nominal).all(), 'robot_independent_check_required')
    need(all(not np.any(np.linalg.norm(card-p, axis=1) < 1e-6) for p in checks), 'robot_check_overlaps_fit')
    for key in ('object_height_mm', 'fixture_height_mm', 'model_z_mm', 'alpha_deg'):
        need(type(data[key]) in (int, float) and math.isfinite(data[key]), 'explicit_robot_height_and_pose_required')
    need(data['object_height_mm'] > 0 and data['fixture_height_mm'] >= 0 and data['model_z_mm'] >= 0
         and -135 <= data['alpha_deg'] <= 0, 'invalid_robot_height_and_pose')
    check_errors = np.linalg.norm(np.c_[checks, np.ones(len(checks))] @ coeff-nominal, axis=1)
    need(check_errors.max() <= 3, 'robot_pair_consistency_error')
    return dict(kind='recorded-robot-affine-fit-v1', card_to_nominal=h.tolist(),
        object_height_mm=data['object_height_mm'], fixture_height_mm=data['fixture_height_mm'],
        model_z_mm=data['model_z_mm'], alpha_deg=data['alpha_deg'],
        fit_pair_residual_max_nominal_mm=float(np.linalg.norm(x @ coeff-control, axis=1).max()),
        check_pair_residual_max_nominal_mm=float(check_errors.max()),
        source_pair_sha256=fingerprint(data), observations=data.get('observations', []),
        physical_error_quantitatively_known=False, physical_precision_accepted=False,
        automatic_motion_allowed=False,
        note='Residual of recorded nominal pairs, not measured robot positioning accuracy. No height interpolation.')


def fit_world_registration(data):
    """Fit a millimetre rigid 2D bridge; reserve independent measured points."""
    need(isinstance(data.get('world_frame_id'), str) and data['world_frame_id'].strip(), 'registration_frame_required')
    world, card = np.asarray(data['world_xy_mm'], float), np.asarray(data['card_xy_mm'], float)
    need(world.shape == card.shape and world.ndim == 2 and world.shape[1] == 2 and
         len(world) >= 3 and np.isfinite(world).all() and np.isfinite(card).all() and
         np.linalg.matrix_rank(world-world.mean(0)) == 2, 'registration_needs_spread_pairs')
    x, y = world-world.mean(0), card-card.mean(0)
    u, _, vt = np.linalg.svd(x.T @ y)
    rotation = vt.T @ u.T
    need(np.linalg.det(rotation) > 0, 'registration_reflection_or_axis_error')
    shift = card.mean(0)-rotation @ world.mean(0)
    h = np.eye(3); h[:2, :2], h[:2, 2] = rotation, shift
    check_world, check_card = np.asarray(data['check_world_xy_mm'], float), np.asarray(data['check_card_xy_mm'], float)
    need(check_world.shape == check_card.shape and check_world.ndim == 2 and check_world.shape[1] == 2 and
         len(check_world) >= 2 and np.isfinite(check_world).all() and np.isfinite(check_card).all(),
         'independent_registration_checks_required')
    need(all(not np.any(np.linalg.norm(world-p, axis=1) < 1e-6) for p in check_world), 'check_pairs_overlap_fit')
    fit_error = np.linalg.norm((rotation @ world.T).T+shift-card, axis=1)
    check_error = np.linalg.norm((rotation @ check_world.T).T+shift-check_card, axis=1)
    need(max(fit_error.max(), check_error.max()) <= 1, 'registration_error_exceeds_1mm')
    return dict(world_frame_id=data['world_frame_id'], world_to_card_xy=h.tolist(), checked=True,
        fit_max_mm=float(fit_error.max()), check_max_mm=float(check_error.max()),
        source_sha256=fingerprint(data), physical_accuracy_verified=False, automatic_motion_allowed=False)
