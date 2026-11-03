"""One supervised cycle for the station's three physically checked specimens.

Production readiness remains in sorting_loop. This narrower contract requires
operator setup, an empty destination, a fixed station and bound physical records.
It never enables production or unattended operation and never retries motion.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np

from .arm import ik, packet
from .sorting_io import CameraYoloe, Motion
from .sorting_loop import (Controller, LABELS, fingerprint, finite, iou, mapped,
                           need, validate_config)
from .sorting_scene import adjust_exposure_reference, check_motion_layout
from .static_visual_verification import roi_mask
from .sorting_motion import PROFILES, compact_plan, motion_wait
from openmv_gripper.km1_arm_bridge import validate_packet

SCOPE = 'supervised-known-item-v1'


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound_file(base, record):
    need(isinstance(record, dict) and isinstance(record.get('path'), str) and
         isinstance(record.get('sha256'), str), 'missing_or_invalid_evidence_reference')
    path = (Path(base) / record['path']).resolve()
    need(path.is_file() and file_hash(path) == record['sha256'], 'evidence_file_changed:' + record['path'])
    return path


def profile_fingerprint(c, label):
    p = copy.deepcopy(c['profiles'][label])
    for key in ('qualification_record', 'commissioned', 'swept_path_verified', 'acceptance_record'):
        p.pop(key, None)
    return fingerprint(dict(station=c['station_id'], camera=c['camera_id'], size=c['image_size'],
        source_roi=c['source_roi'], static_rois=c['layout']['static_rois'],
        initial_pwm=c['known']['initial_pwm'], prepare_pwm=c['known']['prepare_pwm'],
        arm_identity={key: c['arm'].get(key) for key in ('transport', 'port', 'usb_identity')},
        move_ms=c['arm']['move_ms'], grip_move_ms=c['known']['grip_move_ms'],
        settle_s=c['arm']['settle_s'], step_settle_s=c['arm'].get('step_settle_s', {}), profile=p))


def validate_known_config(c):
    validate_config(c)
    need(c['known'].get('motion_profile', 'qualified') in PROFILES, 'unknown_known_motion_profile')
    need(c['known']['scope'] == SCOPE and c['max_objects'] == 1, 'single_known_item_scope_required')
    need(isinstance(c['known'].get('device_lease'), str) and c['known']['device_lease'] and
         isinstance(c['known'].get('stop_files'), list) and c['known']['stop_files'] and
         all(isinstance(p, str) and p for p in c['known']['stop_files']), 'device_lease_and_stop_files_required')
    need(c.get('model_family') in ('yolov8s', 'yolo11s') and c['prompt_mode'] == 'supervised',
         'fixed_trained_model_required')
    need(c['arm']['transport'] == 'direct', 'known_route_transport_not_checked')
    need(c['min_confidence'] >= .7 and c['min_scene_iou'] >= .95 and c['max_age_ms'] <= 2000 and
         c['min_interior_px'] >= 3 and c['rgb_difference_threshold'] <= 90 and
         c['max_scene_changed_fraction'] <= .02 and c['max_other_bin_changed_fraction'] <= .02,
         'known_decision_guards_relaxed')
    need(c['arm']['move_ms'] == 5000 and c['known']['grip_move_ms'] == 2000,
         'known_route_low_speed_required')
    need(c['arm']['settle_s'] >= .5 and c['arm'].get('step_settle_s', {}).get('release', 0) >= 1,
         'known_route_settle_required')
    initial, prepare = c['known']['initial_pwm'], c['known']['prepare_pwm']
    need(len(initial) == 6 and len(prepare) == 4, 'explicit_initial_and_prepare_pose_required')
    for values in (initial, prepare):
        need(all(type(v) is int and 500 <= v <= 2500 for v in values), 'invalid_known_pose_pwm')
    from .sorting_review_policy import review_factor
    factor=review_factor(c)
    need(c['layout']['min_patch_correlation'] >= .98/factor and 0 < c['layout']['max_layout_shift_px'] <= 5*factor,
         'known_layout_limits_relaxed')
    need(c['layout']['min_motion_patch_dynamic_range'] >= 80/factor, 'motion_marker_visibility_required')
    for label in LABELS:
        p = c['profiles'][label]
        need(finite(p['object_height_mm']) and p['object_height_mm'] > 0 and
             (p.get('object_length_mm') is None or finite(p['object_length_mm']) and p['object_length_mm'] > 0),
             'known_specimen_dimensions_required')
        need(finite(p['axis_tolerance_deg']) and 0 < p['axis_tolerance_deg'] <= 20,
             'known_orientation_window_relaxed')
        need(p['alpha_deg'] == -75 and 0 <= p['grasp_z_mm'] < p['source_mid_z_mm'] < p['hover_z_mm'],
             'invalid_known_source_heights')
        need(p['wrist_pwm'] == initial[4] and p['open_pwm'] == initial[5] and
             1000 <= p['open_pwm'] < p['close_pwm'] <= 1500, 'invalid_known_gripper_profile')
        for key in ('image_to_card_mm', 'card_to_grip_mm', 'card_to_hover_mm'):
            mapped(p[key], [[0, 0]])
        for key in ('width_bounds_mm', 'length_bounds_mm'):
            bounds = p[key]
            need(len(bounds) == 2 and all(finite(v) for v in bounds) and 0 < bounds[0] < bounds[1],
                 'invalid_known_shape_bounds')
        need(p['max_grasp_offset_mm'] <= 10 and p['max_grasp_offset_mm'] > 0,
             'known_grasp_offset_limit')
        need(len(p['pick_hulls_card_mm']) >= 2, 'two_height_pick_checks_required')
        for hull in p['pick_hulls_card_mm']:
            poly = np.asarray(hull, np.float32)
            need(poly.ndim == 2 and poly.shape[1] == 2 and len(poly) >= 3 and
                 np.isfinite(poly).all() and cv2.isContourConvex(poly) and cv2.contourArea(poly) > 0,
                 'invalid_known_pick_hull')
        for key in ('transfer_pwm', 'retreat_pwm'):
            poses = p[key]
            need(isinstance(poses, list) and 1 <= len(poses) <= 12, 'missing_checked_route')
            need(all(len(pose) == 4 and all(type(v) is int and 500 <= v <= 2500 for v in pose)
                     for pose in poses), 'invalid_checked_route_pwm')
        need(p['retreat_pwm'][-1] == initial[:4], 'route_must_end_at_neutral')
    return c


def known_readiness(c, base, require_empty=True, require_vision_binding=True):
    blockers = []
    try:
        validate_known_config(c)
    except (KeyError, TypeError, ValueError, cv2.error) as exc:
        return ['known_configuration:' + str(exc)]
    if c.get('synthetic_fixture', False):
        blockers.append('synthetic_fixture')
    if c['known'].get('experiment_authorized') is not True:
        blockers.append('known.experiment_authorized')
    if any((Path(base) / p).resolve().exists() for p in c['known']['stop_files']):
        blockers.append('operator_stop_requested')
    if (Path(base) / c['known']['device_lease']).resolve().exists():
        blockers.append('arm_in_use_or_stale_device_lease')
    board_local = c['statistics']['backend'] == 'pl_board_local'
    if c['statistics']['backend'] not in ('pl_http', 'pl_board_local'):
        blockers.append('known.physical_statistics_requires_pl_http')
    for key in (('station_vision.bit', 'station_vision.hwh') if board_local else ('vision.bit', 'vision.hwh')):
        value = c['statistics'].get('assets', {}).get(key, '')
        if not isinstance(value, str) or len(value) != 64 or any(v not in '0123456789abcdef' for v in value):
            blockers.append('statistics.assets.' + key)
    if board_local:
        try:
            from .sorting_station_io import station_policy
            signature = station_policy(c)
            if require_vision_binding:
                record = json.loads(bound_file(base, c['known']['vision_binding']).read_text(encoding='utf-8'))
                need(record['kind'] == 'known-station-vision-binding-v1' and
                     record['stream_id'] == c['known']['expected_camera_stream_id'] and
                     record['config_sha256'] == signature and record['assets'] == c['statistics']['assets'] and
                     record['geometry_profile_sha256'] == {label: profile_fingerprint(c, label) for label in LABELS} and
                     record['physical_motion'] is False and len(record['frames']) >= 3,
                     'station_binding_changed')
                for frame in record['frames']:
                    bound_file(base, frame)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            blockers.append('known.station_vision_binding:' + str(exc))
    identity = c['arm'].get('usb_identity', {})
    if any(type(identity.get(k)) is not int for k in ('vid', 'pid')):
        blockers.append('arm.usb_identity')
    for label, p in c['profiles'].items():
        try:
            q = json.loads(bound_file(base, p['qualification_record']).read_text(encoding='utf-8'))
            need(q['kind'] == 'known-item-geometry-observation-v1' and q['label'] == label and
                 q['station_id'] == c['station_id'] and q['scope'] == SCOPE and
                 q['operator_confirmed'] is True and q['completed_supervised_trial'] is True and
                 q['profile_sha256'] == profile_fingerprint(c, label), 'known_profile_not_bound')
            need(len(q['evidence']) >= 2, 'physical_evidence_required')
            for record in q['evidence']:
                bound_file(base, record)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            blockers.append(label + '.qualification:' + str(exc))
    for record in c['known'].get('implementation_sources', []):
        try:
            bound_file(base, record)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            blockers.append('implementation_source:' + str(exc))
    for key in ['layout_reference'] + (['empty_reference'] if require_empty else []):
        try:
            image = cv2.imread(str(bound_file(base, c[key])))
            need(image is not None and list(image.shape[:2][::-1]) == c['image_size'], 'reference_size')
            check_motion_layout(image, image, dict(c['layout'], image_size=c['image_size']))
        except (OSError, KeyError, TypeError, ValueError, cv2.error) as exc:
            blockers.append(key + ':' + str(exc))
    return blockers


def compile_known_plan(c, label, mask, geometry):
    """Choose an internal checked grasp point, then compile the whole route.

    Pick-center calibration is distinct from the specimen's full silhouette.
    The latter is bounded by source ROI, dimensions, orientation and qualification.
    """
    p = c['profiles'][label]
    area = int(mask.sum())
    need(p['min_area_px'] <= area <= p['max_area_px'], 'known_object_area_outside_profile')
    need(cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] == 2, 'fragmented_instance')
    need(geometry['m00'] == area, 'geometry_area_mismatch')
    ys, xs = np.nonzero(mask)
    card_all = mapped(p['image_to_card_mm'], np.c_[xs, ys])
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
    card = mapped(p['image_to_card_mm'], np.c_[xx, yy])
    keep = np.ones(len(xx), bool)
    for hull in p['pick_hulls_card_mm']:
        keep &= np.array([cv2.pointPolygonTest(np.asarray(hull, np.float32), tuple(map(float, pt)), False) >= 0
                          for pt in card])
    need(keep.any(), 'no_internal_grasp_in_checked_domain')
    xx, yy, card = xx[keep], yy[keep], card[keep]
    index = int(np.argmin((xx-center[0])**2+(yy-center[1])**2))
    point = [int(xx[index]), int(yy[index])]
    need(np.linalg.norm(card[index]-mapped(p['image_to_card_mm'], [center])[0]) <= p['max_grasp_offset_mm'],
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


class KnownController(Controller):
    def __init__(self, config, source, statistics, motion, journal, reference, empty_reference,
                 config_base, mode='preview', ready=False, motion_profile='qualified', **kwargs):
        validate_known_config(config)
        need(motion_profile in PROFILES, 'unknown_known_motion_profile')
        self.motion_profile = motion_profile
        need(mode in ('preview', 'simulation', 'known_physical'), 'invalid_known_mode')
        if mode == 'known_physical':
            blockers = known_readiness(config, config_base)
            need(ready and not blockers, 'known_scope_not_ready:' + ','.join(blockers))
            need(statistics.backend == config['statistics']['backend'] and
                 statistics.backend in ('pl_http', 'pl_board_local') and
                 statistics.provenance()['assets'] == config['statistics']['assets'],
                 'known_statistics_backend_mismatch')
        super().__init__(config, source, statistics, motion, journal, reference,
                         mode='simulation' if mode == 'simulation' else 'preview', **kwargs)
        self.mode = mode
        self.config_base, self.empty_reference = Path(config_base), empty_reference
        if isinstance(motion, GeometryGuardedMotion):
            motion.expected_stream = lambda: self.stream
            if statistics.backend == 'pl_board_local':
                motion.camera.statistics = statistics

    def check_layout(self, image):
        return check_motion_layout(self.reference, image, dict(self.c['layout'], image_size=self.c['image_size']))

    def movement_wait(self, name, duration):
        return motion_wait(self.c['arm'], name, duration, self.motion_profile)

    def prepare_motion(self, row, task_id):
        if self.stats.backend == 'pl_board_local':
            result = self.stats.bind(row)
            same_frame=(self.stats.baseline_row is row and row.get('layout_evidence') is not None and
                row.get('layout_guard_bgr_sha256')==row['bgr_sha256'] and
                row.get('layout_guard_signature') is not None and
                row.get('layout_guard_signature')==getattr(getattr(self,'geometry',None),'signature',None))
            if not same_frame:self.check_layout(self.stats.baseline_row['image'])
            return dict(result, evidence=self.journal.capture(task_id, 'board_baseline', self.stats.baseline_row))
        return None

    def acquire(self, not_before_ms=None):
        row = super().acquire(not_before_ms)
        expected = self.c['known'].get('expected_camera_stream_id')
        need(not expected or row['observation']['source_stream_id'] == expected, 'qualified_camera_stream_changed')
        return row

    def changed_pixels(self, before, after):
        reference, evidence = adjust_exposure_reference(before['image'], after['image'],
            dict(self.c['layout'], image_size=self.c['image_size']))
        after['photometry'] = evidence
        return np.abs(after['image'].astype(np.int16)-reference.astype(np.int16)).sum(axis=2) > self.c['rgb_difference_threshold']

    def plan(self, row, target):
        obj, mask = target
        region = roi_mask(self.c['profiles'][obj['label']]['bin_roi'], self.c['image_size'])
        need(not any((m & region).any() for m in row['masks']), 'known_destination_not_empty_semantically')
        need(self.empty_reference is not None, 'known_empty_reference_required')
        empty, _ = adjust_exposure_reference(self.empty_reference, row['image'],
            dict(self.c['layout'], image_size=self.c['image_size']))
        difference = np.abs(row['image'].astype(np.int16)-empty.astype(np.int16)).sum(axis=2)
        need(float((difference[region] > self.c['rgb_difference_threshold']).mean()) <=
             self.c['max_scene_changed_fraction'], 'known_destination_rgb_not_empty')
        choice, plan = compile_known_plan(self.c, obj['label'], mask, self.stats.geometry(mask))
        if self.motion_profile == 'compact':
            plan, evidence = compact_plan(self.c, plan)
            choice['motion_optimization'] = evidence
        return choice, plan

    def verify(self, before, target, after):
        board = None
        if self.stats.backend == 'pl_board_local':
            task = self.journal.pending()['task_id']
            def save_extra(index, row):
                self.journal.capture(task, 'board_window_%02d' % index, row)
            board = self.stats.verify_cycle(after, self.c['profiles'][target[0]['label']]['bin_id'],
                                           self.source, save_extra)
            self.journal.save(task+'.board-vision.json', board)
        result = super().verify(before, target, after)
        if board is not None:
            result['board_rgb'] = board
            result['semantic_mask_backend'] = 'pc_numpy'
        for evidence, row in zip(result['frames'], after):
            evidence['photometry'] = row['photometry']
        return result

    def cycle(self):
        report = super().cycle()
        report.update(operation_scope=SCOPE, single_cycle_only=True, production_quality_claimed=False,
                      unattended_accepted=False, in_grip_vision_claimed=False,
                      motion_profile=self.motion_profile,
                      supervised_motion_trial=self.motion_profile == 'compact')
        if self.mode == 'preview':
            report['execution_blockers'] = known_readiness(self.c, self.config_base)
        self.journal.save(report['task_id'] + '.json', report)
        return report


class GeometryGuardedMotion(Motion):
    """Reuse transport/echo waits, checking original camera marks before every write."""
    def __init__(self, c, journal, reference, lease, stop_files):
        super().__init__(c['arm'], stop_files[0])
        self.c, self.journal, self.reference = c, journal, reference
        self.lease, self.stop_files = Path(lease), [Path(p) for p in stop_files]
        self.lease_owned = False
        if c['statistics']['backend'] == 'pl_board_local':
            from .sorting_station_io import StationCamera, StationStatistics
            self.camera = StationCamera(c, None, StationStatistics(c), lossless=False)
        else:
            self.camera = CameraYoloe(c, None)
        self.expected_stream = lambda: None
        self.guard_index = 0

    def check_stop(self):
        need(not any(path.exists() for path in self.stop_files), 'operator_stop_requested')

    def open(self, plan):
        self.check_stop()
        from serial.tools import list_ports
        identity = self.arm['usb_identity']
        devices = [p.device for p in list_ports.comports() if p.vid == identity['vid'] and
                   p.pid == identity['pid'] and (identity.get('serial_number') is None or
                   p.serial_number == identity['serial_number'])]
        need(devices == [self.arm['port']], 'known_station_usb_not_unique')
        self.lease.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.lease, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        self.lease_owned = True
        try:
            os.write(fd, str(os.getpid()).encode())
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            for _, data, duration in plan:
                need(validate_packet(data, [[500, 2500]]*6) == duration, 'invalid_known_packet')
            super().open(plan)
        except BaseException:
            self.close()
            raise

    def send(self, data, duration, guard=None):
        def checked():
            self.check_stop()
            row = self.camera.raw(self.c['frame_timeout_s'])
            need(row['observation']['source_stream_id'] == self.expected_stream(), 'motion_camera_stream_changed')
            geometry = check_motion_layout(self.reference, row['image'],
                dict(self.c['layout'], image_size=self.c['image_size']))
            task = self.journal.pending()['task_id']
            row['observation']['evidence_type'] = 'raw_fixed_marker_motion_guard_only'
            evidence = self.journal.capture(task, 'motion_%02d' % self.guard_index, row)
            self.journal.save(task + '.motion_%02d.json' % self.guard_index, dict(layout=geometry, evidence=evidence))
            self.guard_index += 1
            if guard:
                guard()
        super().send(data, duration, checked)

    def close(self):
        try:
            super().close()
        finally:
            if self.lease_owned:
                self.lease.unlink()
                self.lease_owned = False

    def wait(self, seconds):
        try:
            return super().wait(seconds)
        finally:
            if getattr(self, 'last_receipt', None) is not None:
                task = self.journal.pending()['task_id']
                self.journal.save(task + '.transport_%02d.json' % (self.guard_index-1), self.last_receipt)
