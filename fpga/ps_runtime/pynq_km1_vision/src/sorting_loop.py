"""Static material sorting controller. Device access lives in sorting_io.

One item per transaction, durable motion intent, no replay of uncertain motion.
All geometry is image scale; arm coordinates are mm and angles are degrees.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

import cv2
import numpy as np

from .arm import ik, packet
from .arm_plan import inspect_plan
from .arm_timing import move_ms, settle_s, validate_timing
from .static_visual_verification import decode_mask, layout_evidence, roi_mask

LABELS = ('plastic', 'metal', 'paper')


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def mask_of(obj, size):
    decoded = decode_mask(obj['mask_rle'], size).reshape(size[1], size[0])
    mask = decoded.view(np.bool_)
    x, y, width, height = cv2.boundingRect(decoded)
    need(width > 0 and height > 0, 'empty_instance_mask')
    need(obj['bbox_xyxy'] == [x, y, x+width, y+height],
         'mask_bbox_mismatch')
    return mask


def iou(a, b):
    union = int((a | b).sum())
    return float((a & b).sum() / union) if union else 1.


def resolve_material_duplicates(instances, masks, policy):
    """Resolve only near-identical masks; preserve evidence of competing labels.

    A distinct weak object is never dropped. Close scores on overlapping labels
    reject the observation. This is a decision policy, not a model-quality pass.
    """
    overlap, margin = policy['min_mask_iou'], policy['min_score_margin']
    need(finite(overlap) and .9 <= overlap <= 1 and finite(margin) and 0 < margin <= 1,
         'invalid_duplicate_resolution_policy')
    need(len(instances) == len(masks), 'instance_mask_count_mismatch')
    order = sorted(range(len(instances)), key=lambda i: (-instances[i]['confidence'], i))
    consumed, selected, reports = set(), [], []
    for i in order:
        if i in consumed:
            continue
        group = [j for j in order if j not in consumed and iou(masks[i], masks[j]) >= overlap]
        alternatives = [j for j in group if instances[j]['label'] != instances[i]['label']]
        if alternatives:
            runner = max(instances[j]['confidence'] for j in alternatives)
            need(instances[i]['confidence']-runner >= margin, 'ambiguous_material_on_same_instance')
        selected.append((instances[i], masks[i]))
        consumed.update(group)
        if len(group) > 1:
            reports.append(dict(selected_index=i, selected_label=instances[i]['label'],
                selected_confidence=instances[i]['confidence'],
                duplicate_candidates=[dict(index=j, label=instances[j]['label'],
                    confidence=instances[j]['confidence'], mask_iou=iou(masks[i],masks[j])) for j in group if j != i]))
    return selected, reports


def mapped(matrix, points):
    matrix = np.asarray(matrix, float)
    need(matrix.shape == (3, 3) and np.isfinite(matrix).all() and
         abs(np.linalg.det(matrix)) > 1e-10, 'invalid_height_homography')
    points = np.atleast_2d(points)
    result = np.c_[points, np.ones(len(points))] @ matrix.T
    need(np.isfinite(result).all() and np.all(np.abs(result[:, 2]) > 1e-9), 'projection_at_infinity')
    return result[:, :2] / result[:, 2:]


def validate_config(c):
    need(c['schema_version'] == 'sorting-loop-v1', 'unsupported_loop_config')
    need(isinstance(c['station_id'], str) and c['station_id'].strip(), 'missing_station_id')
    size = c['image_size']
    decode_mask([size[0]*size[1]], size)
    need(c['prompt_mode'] in ('text', 'visual', 'supervised'), 'invalid_prompt_mode')
    if c['prompt_mode'] == 'supervised':
        need(c.get('model_family') in ('yolo11s', 'yolov8s'), 'supervised_mode_requires_explicit_model_family')
    for key in ('camera_id', 'model_version', 'runtime_id', 'calibration_id'):
        need(isinstance(c[key], str) and c[key].strip(), 'missing_identity:' + key)
    need(set(c['profiles']) == set(LABELS), 'require_three_material_profiles')
    need(c['statistics']['backend'] in ('cpu_reference', 'pl_http', 'pl_board_local'), 'unknown_statistics_backend')
    need(type(c['stable_frames']) is int and 3 <= c['stable_frames'] <= 10, 'invalid_stable_frames')
    need(type(c['max_objects']) is int and 1 <= c['max_objects'] <= 3, 'invalid_object_limit')
    for key in ('max_age_ms', 'frame_timeout_s', 'min_frame_gap_ms', 'min_separation_px',
                'min_interior_px', 'max_cycle_s', 'observe_timeout_s'):
        need(finite(c[key]) and c[key] > 0, 'invalid_threshold:' + key)
    for key in ('min_confidence', 'min_scene_iou', 'min_erased_fraction', 'min_persistent_fraction'):
        need(finite(c[key]) and 0 < c[key] <= 1, 'invalid_fraction:' + key)
    for key in ('max_scene_changed_fraction', 'max_other_bin_changed_fraction'):
        need(finite(c[key]) and 0 <= c[key] < 1, 'invalid_fraction:' + key)
    need(finite(c['rgb_difference_threshold']) and 0 < c['rgb_difference_threshold'] <= 765,
         'invalid_rgb_threshold')
    need(type(c['min_bin_gain_px']) is int and c['min_bin_gain_px'] > 0, 'invalid_bin_gain')
    need(finite(c['retreat_settle_s']) and 0 <= c['retreat_settle_s'] <= 10, 'invalid_retreat_settle')
    need(type(c['require_distinct_jpeg']) is bool, 'invalid_distinct_frame_policy')
    regions = [roi_mask(c['source_roi'], size)]
    bins = set()
    for label, profile in c['profiles'].items():
        need(profile['bin_id'] not in bins and isinstance(profile['bin_id'], str), 'duplicate_bin_id')
        bins.add(profile['bin_id'])
        regions.append(roi_mask(profile['bin_roi'], size))
    patches = [roi_mask(r, size) for r in c['layout']['static_rois']]
    need(len(patches) >= 2, 'need_static_layout_patches')
    all_regions = regions + patches
    need(not any((a & b).any() for i, a in enumerate(all_regions) for b in all_regions[i+1:]),
         'overlapping_station_rois')
    validate_timing(c['arm'])
    return c


def readiness(c):
    """Never turn readiness on. List measured inputs still required by the station."""
    blockers = []
    for key in ('quality_gate_passed', 'station_commissioned', 'initial_pose_confirmed'):
        if c.get(key) is not True:
            blockers.append(key)
    if c['arm'].get('commissioned') is not True:
        blockers.append('arm.commissioned')
    from openmv_gripper.km1_arm_bridge import checked_limits
    try:
        need(all(v is not None for v in checked_limits(c['arm'].get('joint_limits_pwm'))), 'unmeasured_joint')
    except (ValueError, TypeError):
        blockers.append('arm.joint_limits_pwm')
    if c['statistics']['backend'] != 'pl_http':
        blockers.append('statistics.backend: physical execution requires pl_http')
    for key in ('vision.bit', 'vision.hwh'):
        value = c['statistics'].get('assets', {}).get(key)
        if not isinstance(value, str) or len(value) != 64 or any(v not in '0123456789abcdef' for v in value):
            blockers.append('statistics.assets.' + key)
    for label, p in c['profiles'].items():
        for key in ('commissioned', 'swept_path_verified'):
            if p.get(key) is not True:
                blockers.append(label + '.' + key)
        for key in ('image_to_grip_mm', 'image_to_hover_mm', 'verified_polygon_px',
                    'grasp_z_mm', 'hover_z_mm', 'bin_xyz_mm', 'wrist_pwm', 'open_pwm',
                    'close_pwm', 'alpha_deg', 'axis_deg', 'axis_tolerance_deg', 'min_area_px',
                    'max_area_px', 'min_width_mm', 'max_width_mm', 'object_profile_id',
                    'object_height_mm', 'acceptance_record'):
            if p.get(key) is None:
                blockers.append(label + '.' + key)
    if c.get('synthetic_fixture', False):
        blockers.append('synthetic_fixture: replace illustrative data with measured station configuration')
    if not c['arm'].get('home'):
        blockers.append('arm.home')
    if not finite(c['arm'].get('home_alpha_deg')):
        blockers.append('arm.home_alpha_deg')
    identity = c['arm'].get('usb_identity')
    if (not isinstance(identity, dict) or any(type(identity.get(k)) is not int or
        not 0 <= identity[k] <= 65535 for k in ('vid', 'pid'))):
        blockers.append('arm.usb_identity')
    return blockers


def acceptance_fingerprint(c):
    """Commissioning records bind all behavior, including thresholds and timing.

    Only readiness declarations and evidence-file locations are excluded, to
    allow recording acceptance before enabling the reviewed configuration.
    """
    checked = copy.deepcopy(c)
    for key in ('quality_gate_passed', 'station_commissioned', 'initial_pose_confirmed'):
        checked.pop(key, None)
    checked['arm'].pop('commissioned', None)
    for profile in checked['profiles'].values():
        for key in ('commissioned', 'swept_path_verified', 'acceptance_record'):
            profile.pop(key, None)
    return fingerprint(checked)


class Journal:
    """Exclusive station ledger. An abandoned lock requires deliberate inspection.

    The pending file is written before USB opens. A crash, a partial write, or
    an ambiguous visual verdict keeps the station latched across restarts.
    """
    def __init__(self, directory):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = self.root / 'controller.lock'
        self.fd = None

    def __enter__(self):
        self.fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.write(self.fd, str(os.getpid()).encode())
        os.fsync(self.fd)
        return self

    def __exit__(self, *args):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
            self.lock.unlink()

    def save(self, name, value):
        path = self.root / name
        tmp = self.root / (name + '.tmp')
        with tmp.open('w', encoding='utf-8') as out:
            json.dump(value, out, ensure_ascii=False, indent=2, allow_nan=False)
            out.flush()
            os.fsync(out.fileno())
        # Windows readers/scanners may briefly deny replacement. Retry only
        # this already-fsynced file operation, never the enclosing arm step.
        for attempt in range(10):
            try:
                os.replace(tmp, path)
                break
            except PermissionError as exc:
                if getattr(exc,'winerror',None) not in (5,32,33) or attempt==9:
                    raise
                time.sleep(.05)

    def pending(self):
        path = self.root / 'pending.json'
        return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None

    def clear(self):
        (self.root / 'pending.json').unlink()

    def resolve(self, task_id, note):
        need(note.strip(), 'operator_observation_required')
        pending = self.pending()
        need(pending is not None and pending['task_id'] == task_id, 'wrong_pending_task')
        self.save(task_id + '.review.json', dict(task_id=task_id, observation=note,
                  outcome='operator_resolved_not_automatic_success', pending=pending))
        self.clear()

    def capture(self, task_id, phase, row):
        name = task_id + '.' + phase
        ok, encoded = cv2.imencode('.png', row['image'])
        need(ok, 'cannot_encode_evidence')
        raw = encoded.tobytes()
        with (self.root / (name + '.png')).open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        evidence = dict(observation=row['observation'], image_path=name+'.png',
                        png_sha256=hashlib.sha256(raw).hexdigest(),
                        decoded_bgr_sha256=row['bgr_sha256'], source_jpeg_sha256=row['jpeg_sha256'],
                        source_encoding=row.get('source_encoding', 'jpeg'),
                        source_bytes_sha256=row.get('source_bytes_sha256', row['jpeg_sha256']))
        self.save(name + '.observation.json', evidence)
        return dict(image=name+'.png', observation=name+'.observation.json', png_sha256=evidence['png_sha256'])


class Controller:
    def __init__(self, config, source, statistics, motion, journal, reference,
                 mode='preview', clock=time.monotonic, sleep=time.sleep):
        self.c = copy.deepcopy(validate_config(config))
        need(mode in ('preview', 'simulation', 'physical'), 'invalid_mode')
        if mode == 'physical':
            need(not config.get('synthetic_fixture', False), 'synthetic_config_forbidden')
            need(not readiness(config), 'station_not_ready:' + ','.join(readiness(config)))
            need(statistics.backend == 'pl_http' and statistics.provenance()['assets'] ==
                 config['statistics']['assets'], 'physical_statistics_backend_mismatch')
        self.source, self.stats, self.motion, self.journal = source, statistics, motion, journal
        self.reference, self.mode, self.clock, self.sleep = reference, mode, clock, sleep
        self.stream = None
        self.frame_id = -1
        self.capture_ms = -1
        self.jpeg_sha = None

    def movement_wait(self, name, duration):
        return duration/1000 + settle_s(self.c['arm'], name)

    def prepare_motion(self, row, task_id):
        return None

    def check_layout(self, image):
        return layout_evidence(self.reference, image, dict(self.c['layout'], image_size=self.c['image_size']))

    def changed_pixels(self, before, after):
        difference = np.abs(after['image'].astype(np.int16)-before['image'].astype(np.int16)).sum(axis=2)
        return difference > self.c['rgb_difference_threshold']

    def acquire(self, not_before_ms=None):
        row = self.source.next(self.c['frame_timeout_s'])
        return self.accept_frame(row,not_before_ms)

    def accept_frame(self,row,not_before_ms=None,*,semantic=True,layout_check=None):
        obs, image = row['observation'], row['image']
        now = self.clock()*1000
        for key in (('camera_id','model_version','runtime_id','image_size','prompt_mode') if semantic else ('camera_id',)):
            need(obs[key] == self.c[key], 'observation_identity_mismatch:' + key)
        need(image.dtype == np.uint8 and image.shape == (self.c['image_size'][1], self.c['image_size'][0], 3),
             'invalid_frame_image')
        source_hash = row.get('source_bytes_sha256', row['jpeg_sha256'])
        need(obs['image_sha256'] == source_hash, 'mask_image_binding_mismatch')
        need(row['bgr_sha256'] == hashlib.sha256(image.tobytes()).hexdigest(), 'decoded_image_changed')
        stream = obs['source_stream_id']
        need(isinstance(stream, str) and stream, 'invalid_camera_stream')
        need(self.stream is None or self.stream == stream, 'camera_stream_restarted')
        need(type(obs['frame_id']) is int and obs['frame_id'] > self.frame_id, 'repeated_or_old_frame')
        stamp = obs['capture_started_ms']
        need(finite(stamp) and 0 <= now-stamp <= self.c['max_age_ms'], 'stale_or_future_frame')
        need(stamp-self.capture_ms >= self.c['min_frame_gap_ms'], 'insufficient_frame_gap')
        need(not_before_ms is None or stamp > not_before_ms, 'frame_precedes_retreat')
        need(not self.c['require_distinct_jpeg'] or self.jpeg_sha != source_hash, 'frozen_camera_image')
        self.stream, self.frame_id, self.capture_ms = stream, obs['frame_id'], stamp
        self.jpeg_sha = source_hash
        # Calibrated layout, not just comparison against an arbitrary new frame.
        row['layout_evidence']=(layout_check or self.check_layout)(image)
        row['layout_guard_bgr_sha256']=row['bgr_sha256']
        row['layout_guard_signature']=getattr(getattr(self,'geometry',None),'signature',None)
        row['masks'] = []
        if not semantic:
            need(row.get('source_encoding')=='png','rgb_verification_requires_lossless_frame')
            need('instances' not in obs,'rgb_only_frame_contains_unverified_instances')
            obs['instances']=[]
            row['perception_kind']='raw_rgb_only_no_model_inference'
            return row
        for obj in obs['instances']:
            need(obj['label'] in LABELS and finite(obj['confidence']) and 0 <= obj['confidence'] <= 1,
                 'invalid_material_instance')
            row['masks'].append(mask_of(obj, self.c['image_size']))
        return row

    def source_objects(self, row):
        roi = roi_mask(self.c['source_roi'], self.c['image_size'])
        found = []
        pairs = list(zip(row['observation']['instances'], row['masks']))
        if self.c.get('duplicate_instance_policy') is not None:
            pairs, report = resolve_material_duplicates(row['observation']['instances'], row['masks'],
                                                       self.c['duplicate_instance_policy'])
            row['duplicate_resolution'] = report
        for obj, mask in pairs:
            if (mask & roi).any():
                need(not (mask & ~roi).any(), 'partial_source_object')
                need(obj['confidence'] >= self.c['min_confidence'], 'uncertain_source_object')
                found.append((obj, mask))
        need(len(found) <= self.c['max_objects'], 'too_many_source_objects')
        for i, (_, a) in enumerate(found):
            distance = cv2.distanceTransform((~a).astype(np.uint8), cv2.DIST_L2, 5)
            for _, b in found[i+1:]:
                need(float(distance[b].min()) >= self.c['min_separation_px'], 'objects_touching_or_too_close')
        return sorted(found, key=lambda x: (x[0]['label'], x[0]['bbox_xyxy']))

    def unchanged(self, before, after):
        a, b = self.source_objects(before), self.source_objects(after)
        need(len(a) == len(b), 'source_scene_changed')
        for (oa, ma), (ob, mb) in zip(a, b):
            need(oa['label'] == ob['label'] and iou(ma, mb) >= self.c['min_scene_iou'],
                 'source_scene_changed')
        changed = self.changed_pixels(before, after)
        for roi in [self.c['source_roi']] + [p['bin_roi'] for p in self.c['profiles'].values()]:
            region = roi_mask(roi, self.c['image_size'])
            need(float(changed[region].mean()) <= self.c['max_scene_changed_fraction'], 'scene_rgb_changed')

    def plan(self, row, target):
        obj, mask = target
        p, c = self.c['profiles'][obj['label']], self.c
        # All profile fields are explicit, even for a dry preview.
        need(p.get('image_to_grip_mm') is not None, 'missing_profile_calibration:' + obj['label'])
        need(isinstance(p['object_profile_id'], str) and p['object_profile_id'].strip() and
             finite(p['object_height_mm']) and p['object_height_mm'] > 0, 'invalid_object_height_profile')
        area = int(mask.sum())
        need(p['min_area_px'] <= area <= p['max_area_px'], 'object_size_outside_profile')
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        need(count == 2, 'fragmented_instance')
        polygon = np.asarray(p['verified_polygon_px'], np.float32)
        need(polygon.ndim == 2 and polygon.shape[1] == 2 and len(polygon) >= 3 and
             np.isfinite(polygon).all() and cv2.isContourConvex(polygon) and cv2.contourArea(polygon) > 0,
             'invalid_verified_polygon')
        contour, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        need(all(cv2.pointPolygonTest(polygon, tuple(map(float, pt)), False) >= 0
                 for pt in contour[0][:, 0]), 'silhouette_outside_verified_region')
        geometry = self.stats.geometry(mask)
        need(geometry['m00'] == area, 'geometry_area_mismatch')
        center = np.array([geometry['m10']/area, geometry['m01']/area])
        distance = cv2.distanceTransform(np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
        yy, xx = np.where(distance >= max(c['min_interior_px'], float(distance.max())*.8))
        need(len(xx) > 0, 'no_interior_grasp_point')
        i = int(np.argmin((xx-center[0])**2 + (yy-center[1])**2))
        point = [int(xx[i]), int(yy[i])]
        ys, xs = np.nonzero(mask)
        covariance = np.array([[geometry['m20']/area-center[0]**2, geometry['m11']/area-center.prod()],
                               [geometry['m11']/area-center.prod(), geometry['m02']/area-center[1]**2]])
        values, vectors = np.linalg.eigh(covariance)
        need(values[1] > 0 and values[1] >= values[0]*1.2, 'ambiguous_object_axis')
        axis = vectors[:, 1]
        angle = math.degrees(math.atan2(axis[1], axis[0])) % 180
        need(finite(p['axis_deg']) and finite(p['axis_tolerance_deg']) and
             0 < p['axis_tolerance_deg'] <= 90 and
             abs((angle-p['axis_deg']+90) % 180-90) <= p['axis_tolerance_deg'], 'unverified_grasp_orientation')
        world = mapped(p['image_to_grip_mm'], np.c_[xs, ys])
        # PCA in robot coordinates determines the short dimension; fixed wrist
        # is only legal in the narrow, physically checked orientation profile.
        eigen, basis = np.linalg.eigh(np.cov(world.T))
        short = world @ basis[:, 0]
        width = float(short.max()-short.min())
        need(finite(p['min_width_mm']) and finite(p['max_width_mm']) and
             0 < p['min_width_mm'] <= width <= p['max_width_mm'], 'jaw_width_outside_profile')
        grip = mapped(p['image_to_grip_mm'], [point])[0].tolist()
        hover = mapped(p['image_to_hover_mm'], [point])[0].tolist()
        xmin, xmax, ymin, ymax = c['arm']['pick_bounds_mm']
        need(np.all((world[:, 0] >= xmin) & (world[:, 0] <= xmax) &
                    (world[:, 1] >= ymin) & (world[:, 1] <= ymax)), 'object_outside_arm_bounds')
        need(xmin <= hover[0] <= xmax and ymin <= hover[1] <= ymax, 'hover_outside_arm_bounds')
        a = copy.deepcopy(c['arm'])
        need(all(finite(p[k]) for k in ('grasp_z_mm', 'hover_z_mm', 'alpha_deg')) and
             0 <= p['grasp_z_mm'] < p['hover_z_mm'] and -135 <= p['alpha_deg'] <= 0, 'invalid_grasp_height_or_alpha')
        for key in ('wrist_pwm', 'open_pwm', 'close_pwm'):
            need(type(p[key]) is int and 500 <= p[key] <= 2500, 'invalid_profile_pwm')
            a[key] = p[key]
        bx, by, bz = p['bin_xyz_mm']
        need(all(finite(v) for v in (bx, by, bz)) and 0 <= bz < p['hover_z_mm'], 'invalid_bin_height')
        plan = []
        def move(name, xyz):
            ms = move_ms(a, name)
            alpha = a['home_alpha_deg'] if name in ('home', 'return') else p['alpha_deg']
            pwm = ik(*xyz, alpha, a.get('base_yaw_calibration'))
            plan.append((name, packet(list(enumerate(pwm)), ms), ms))
        def servo(name, sid, value):
            ms = move_ms(a, name)
            plan.append((name, packet([(sid, value)], ms), ms))
        move('home', a['home'])
        servo('wrist', 4, p['wrist_pwm'])
        servo('open', 5, p['open_pwm'])
        move('above_target', hover+[p['hover_z_mm']])
        move('descend', grip+[p['grasp_z_mm']])
        servo('close', 5, p['close_pwm'])
        move('lift', hover+[p['hover_z_mm']])
        move('above_bin', [bx, by, p['hover_z_mm']])
        move('lower_bin', [bx, by, bz])
        servo('release', 5, p['open_pwm'])
        move('leave_bin', [bx, by, p['hover_z_mm']])
        move('return', a['home'])
        report = inspect_plan(plan, dict(arm=a))
        # Apply measured limits to direct UART too, including wrist and gripper.
        from openmv_gripper.km1_arm_bridge import checked_limits, validate_packet
        limits = checked_limits(a['joint_limits_pwm'])
        for _, data, ms in plan:
            need(validate_packet(data, limits) == ms, 'invalid_motion_packet')
        blockers = [v for v in report['execution_blockers'] if v != 'Arm not commissioned']
        need(not blockers, 'invalid_plan:' + ';'.join(blockers))
        return dict(label=obj['label'], bin_id=p['bin_id'], pick_pixel=point,
                    grip_xy_mm=grip, hover_xy_mm=hover, width_mm=width,
                    axis_deg=angle, geometry=geometry, validation=report), plan

    def verify(self, before, target, after):
        c = self.c
        obj, selected = target
        p = c['profiles'][obj['label']]
        source = roi_mask(c['source_roi'], c['image_size'])
        destination = roi_mask(p['bin_roi'], c['image_size'])
        before_union = np.logical_or.reduce(before['masks']) if before['masks'] else np.zeros_like(selected)
        previous_bin = before_union & destination
        targets, gains = [], []
        remaining = [t for t in self.source_objects(before) if t[0] is not obj]
        reports = []
        for row in after:
            union = np.logical_or.reduce(row['masks']) if row['masks'] else np.zeros_like(selected)
            erased = self.stats.visual(selected, union & source)
            need(erased['erased']/int(selected.sum()) >= c['min_erased_fraction'], 'selected_object_still_in_source')
            actual_remaining = self.source_objects(row)
            need(len(actual_remaining) == len(remaining), 'unexpected_source_change')
            for (oa, ma), (ob, mb) in zip(remaining, actual_remaining):
                need(oa['label'] == ob['label'] and iou(ma, mb) >= c['min_scene_iou'], 'other_source_object_changed')
            new_candidates = []
            # Existing bin contents must remain independently visible. A moved
            # old item, merged mask or increased area is not a new deposit.
            for old, old_mask in zip(before['observation']['instances'], before['masks']):
                if (old_mask & source).any():
                    continue
                matches = [(item, m) for item, m in zip(row['observation']['instances'], row['masks'])
                           if item['label'] == old['label'] and iou(old_mask, m) >= c['min_scene_iou']]
                need(len(matches) == 1, 'existing_scene_object_changed_or_occluded')
            for item, mask in zip(row['observation']['instances'], row['masks']):
                if item['label'] != obj['label'] or item['confidence'] < c['min_confidence']:
                    continue
                if (mask & destination).sum() < mask.sum()*.95:
                    continue
                if any(iou(mask, old_mask) > .1 for old_mask in before['masks']
                       if (old_mask & destination).any()):
                    continue
                novel = mask & ~previous_bin
                if int(novel.sum()) >= c['min_bin_gain_px']:
                    new_candidates.append(mask)
            need(len(new_candidates) == 1, 'no_unique_new_target_in_correct_bin')
            mask = new_candidates[0]
            need(p['min_area_px'] <= int(mask.sum()) <= p['max_area_px'], 'unexpected_bin_object_size')
            changed = self.changed_pixels(before, row)
            gain_mask = mask & destination & ~previous_bin & changed
            gain = self.stats.visual(np.zeros_like(mask), gain_mask)
            need(gain['beyond'] >= c['min_bin_gain_px'], 'no_semantic_and_rgb_bin_gain')
            for other in c['profiles'].values():
                if other['bin_id'] != p['bin_id']:
                    region = roi_mask(other['bin_roi'], c['image_size'])
                    need(float(changed[region].mean()) <= c['max_other_bin_changed_fraction'], 'wrong_bin_or_external_intervention')
            targets.append(mask)
            gains.append(gain_mask)
            reports.append(dict(frame_id=row['observation']['frame_id'], source=erased, bin=gain))
        need(int(np.logical_and.reduce(gains).sum()) >= c['min_bin_gain_px'], 'bin_gain_not_persistent')
        need(np.logical_and.reduce(targets).sum()/np.logical_or.reduce(targets).sum() >=
             c['min_persistent_fraction'], 'bin_target_not_stable')
        return dict(status='visual_evidence_pass', frames=reports, backend=self.stats.backend,
                    independently_validated=False, physical_success_verified=False)

    def cycle(self):
        need(self.journal.pending() is None, 'unresolved_previous_task')
        task_id = uuid.uuid4().hex
        report = dict(task_id=task_id, station_id=self.c['station_id'], mode=self.mode,
                      config_sha256=fingerprint(self.c), status='observing', steps=[],
                      evidence=[],
                      automatic_retry=False, physical_success_verified=False)
        attempted = False
        started = self.clock()
        try:
            rows = []
            for _ in range(self.c['stable_frames']):
                row = self.acquire()
                if rows:
                    self.unchanged(rows[-1], row)
                rows.append(row)
                need(self.clock()-started <= self.c['observe_timeout_s'], 'observation_deadline_exceeded')
            need(all(self.clock()*1000-r['observation']['capture_started_ms'] <= self.c['max_age_ms']
                     for r in rows), 'stable_observation_window_expired')
            before = rows[-1]
            objects = self.source_objects(before)
            if not objects:
                report['status'] = 'empty_scene'
                return report
            # Deterministic ordering, reobserve after each placement. Reject an
            # ungraspable chosen item rather than silently losing its ledger entry.
            target = objects[0]
            choice, plan = self.plan(before, target)
            report.update(choice=choice, plan=[dict(step=n, command=d.decode('ascii'), duration_ms=ms)
                                              for n, d, ms in plan])
            report['statistics'] = self.stats.provenance()
            report['evidence'].append(self.journal.capture(task_id, 'before', before))
            if self.mode == 'preview':
                report.update(status='preview_only', execution_blockers=readiness(self.c))
                return report
            fresh = self.acquire()
            self.unchanged(before, fresh)
            need(self.clock()*1000-fresh['observation']['capture_started_ms'] <= self.c['max_age_ms'],
                 'precommit_frame_expired')
            report['status'] = 'motion_intent_reserved'
            report['evidence'].append(self.journal.capture(task_id, 'precommit', fresh))
            report['motion_baseline'] = self.prepare_motion(fresh, task_id)
            self.journal.save('pending.json', report)
            attempted = True
            self.motion.open(plan)
            for name, data, duration in plan:
                need(self.clock()-started <= self.c['max_cycle_s'], 'cycle_deadline_exceeded')
                self.motion.check_stop()
                guard = None
                # Opening serial can reset devices and takes seconds. Recheck
                # immediately before the first command and again before approach.
                if name in ('home', 'above_target', 'descend'):
                    current = self.acquire()
                    self.unchanged(before, current)
                    report['evidence'].append(self.journal.capture(task_id, 'guard_'+name, current))
                    need(self.clock()*1000-current['observation']['capture_started_ms'] <= self.c['max_age_ms'],
                         'motion_guard_frame_expired')
                    stamp = current['observation']['capture_started_ms']
                    def guard(stamp=stamp):
                        need(0 <= self.clock()*1000-stamp <= self.c['max_age_ms'], 'motion_guard_frame_expired')
                entry = dict(step=name, status='may_be_in_flight', duration_ms=duration)
                report['steps'].append(entry)
                self.journal.save('pending.json', report)
                self.motion.send(data, duration, guard=guard)
                entry['required_wait_s'] = self.movement_wait(name, duration)
                self.motion.wait(entry['required_wait_s'])
                if getattr(self.motion, 'last_receipt', None) is not None:
                    entry['transport_receipt'] = copy.deepcopy(self.motion.last_receipt)
                entry['status'] = 'sent_and_waited_unverified'
                self.journal.save('pending.json', report)
            self.motion.close()
            self.sleep(self.c['retreat_settle_s'])
            barrier = self.clock()*1000
            # Discard cached pre-retreat exposures in the source adapter.
            self.source.after(barrier)
            report['status'] = 'verifying'
            self.journal.save('pending.json', report)
            after = []
            for index in range(self.c['stable_frames']):
                after.append(self.acquire(not_before_ms=barrier))
                report['evidence'].append(self.journal.capture(task_id, 'placed_'+str(index), after[-1]))
            need(self.clock()-started <= self.c['max_cycle_s'], 'cycle_deadline_exceeded')
            report['verification'] = self.verify(before, target, after)
            # Validate ages at the final decision, not only when rows arrived.
            need(all(self.clock()*1000-r['observation']['capture_started_ms'] <= self.c['max_age_ms']
                     for r in after), 'verification_evidence_expired')
            report['status'] = 'simulated_visual_pass' if self.mode == 'simulation' else 'completed_visual_evidence'
            self.journal.save(task_id + '.json', report)
            self.journal.clear()
            return report
        except (Exception, KeyboardInterrupt) as exc:
            report.update(status='needs_review' if attempted else 'rejected',
                          reason=type(exc).__name__ + ':' + str(exc))
            if attempted:
                self.journal.save('pending.json', report)
            return report
        finally:
            if attempted:
                self.motion.close()
            report['elapsed_s'] = self.clock()-started
            self.journal.save(task_id + '.json', report)
