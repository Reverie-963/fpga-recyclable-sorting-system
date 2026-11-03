"""Existing known-item route checks with an explicit calibrated ray mapper.

Kept separate so an in-progress legacy physical task retains its frozen source
hashes. Equivalence tests bind this copy to the original recipe and guards.
"""
import math
import cv2
import numpy as np
from .arm import ik, packet
from .sorting_known import SCOPE
from .sorting_loop import mapped, need
from openmv_gripper.km1_arm_bridge import validate_packet


def compile_relocated_plan(c, label, mask, geometry, pixel_to_card):
    """Choose an internal checked grasp point, then compile the whole route.

    Pick-center calibration is distinct from the specimen's full silhouette.
    The latter is bounded by source ROI, dimensions, orientation and qualification.
    """
    p = c['profiles'][label]
    need(callable(pixel_to_card), 'explicit_calibrated_pixel_mapper_required')
    card_points = pixel_to_card
    area = int(mask.sum())
    need(p['min_area_px'] <= area <= p['max_area_px'], 'known_object_area_outside_profile')
    need(cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] == 2, 'fragmented_instance')
    need(geometry['m00'] == area, 'geometry_area_mismatch')
    ys, xs = np.nonzero(mask)
    card_all = card_points(np.c_[xs, ys])
    values, vectors = np.linalg.eigh(np.cov(card_all.T))
    need(values[1] >= values[0]*1.2, 'known_object_axis_ambiguous')
    axis = vectors[:, 1]
    angle = math.degrees(math.atan2(axis[1], axis[0])) % 180
    need(abs((angle-90+90) % 180-90) <= p['axis_tolerance_deg'], 'known_object_orientation')
    widths = card_all @ vectors[:, 0]
    lengths = card_all @ vectors[:, 1]
    width, length = float(np.ptp(widths)), float(np.ptp(lengths))
    need(p['width_bounds_mm'][0] <= width <= p['width_bounds_mm'][1] and
         p['length_bounds_mm'][0] <= length <= p['length_bounds_mm'][1], 'known_specimen_dimensions')
    center = np.array([geometry['m10']/area, geometry['m01']/area])
    distance = cv2.distanceTransform(np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    yy, xx = np.where(distance >= max(c['min_interior_px'], float(distance.max())*.8))
    card = card_points(np.c_[xx, yy])
    keep = np.ones(len(xx), bool)
    for hull in p['pick_hulls_card_mm']:
        keep &= np.array([cv2.pointPolygonTest(np.asarray(hull, np.float32), tuple(map(float, pt)), False) >= 0
                          for pt in card])
    need(keep.any(), 'no_internal_grasp_in_checked_domain')
    xx, yy, card = xx[keep], yy[keep], card[keep]
    index = int(np.argmin((xx-center[0])**2+(yy-center[1])**2))
    point = [int(xx[index]), int(yy[index])]
    need(np.linalg.norm(card[index]-card_points([center])[0]) <= p['max_grasp_offset_mm'],
         'grasp_too_far_from_specimen_center')
    grip = mapped(p['card_to_grip_mm'], [card[index]])[0].tolist()
    hover = mapped(p['card_to_hover_mm'], [card[index]])[0].tolist()
    xmin, xmax, ymin, ymax = c['arm']['pick_bounds_mm']
    need(all(xmin <= xy[0] <= xmax and ymin <= xy[1] <= ymax for xy in (grip, hover)),
         'known_source_pose_outside_checked_range')
    mid = [(a+b)/2 for a, b in zip(grip, hover)]
    ms, grip_ms = c['arm']['move_ms'], c['known']['grip_move_ms']
    plan = [('home', packet(list(enumerate(c['known']['prepare_pwm'])) +
                           [(4, p['wrist_pwm']), (5, p['open_pwm'])], ms), ms)]
    def move(name, xy, z):
        plan.append((name, packet(list(enumerate(ik(*xy, z, p['alpha_deg']))), ms), ms))
    move('source_hover', hover, p['hover_z_mm'])
    move('source_descend_mid', mid, p['source_mid_z_mm'])
    move('source_grip', grip, p['grasp_z_mm'])
    plan.append(('close', packet([(5, p['close_pwm'])], grip_ms), grip_ms))
    move('source_lift_mid', mid, p['source_mid_z_mm'])
    move('source_lift_hover', hover, p['hover_z_mm'])
    for i, pose in enumerate(p['transfer_pwm']):
        plan.append(('transfer_' + str(i), packet(list(enumerate(pose)), ms), ms))
    plan.append(('release', packet([(5, p['open_pwm'])], grip_ms), grip_ms))
    for i, pose in enumerate(p['retreat_pwm']):
        name = 'return' if i == len(p['retreat_pwm'])-1 else 'retreat_' + str(i)
        plan.append((name, packet(list(enumerate(pose)), ms), ms))
    for _, data, duration in plan:
        need(validate_packet(data, [[500, 2500]]*6) == duration, 'known_packet_outside_protocol')
    return dict(label=label, bin_id=p['bin_id'], pick_pixel=point, pick_card_xy_mm=card[index].tolist(),
        grip_xy_mm=grip, hover_xy_mm=hover, width_mm=width, length_mm=length, axis_deg=angle,
        geometry=geometry, qualification_sha256=(p.get('qualification_record') or {}).get('sha256'),
        validation=dict(scope=SCOPE, software_protocol_checks_passed=True,
                        measured_mechanical_hard_limits_claimed=False, production_ready=False)), plan
