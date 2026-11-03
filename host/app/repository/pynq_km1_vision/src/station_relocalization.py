"""Frame-bound marker-only relocation and legacy-domain previews; no motion I/O."""
from collections import deque
import copy
import time

import cv2
import numpy as np

from .station_geometry import (detect_markers, fingerprint, homography, legacy_control,
    localize_records, map_xy, need, polygon, registration_ray_point, world_to_pixels)


def legacy_domains(registration, profiles, station):
    need(registration.get('checked') is True and registration['world_frame_id'] == station['world_frame_id'],
         'checked_world_registration_required')
    h = homography(registration['world_to_card_xy'])
    need(np.allclose(h[2], [0, 0, 1]) and np.linalg.det(h[:2, :2]) > 0 and
         np.allclose(h[:2, :2].T@h[:2, :2], np.eye(2), atol=1e-6), 'rigid_registration_required')
    inverse = np.linalg.inv(h)
    domains = {}
    for name, source in profiles.items():
        profile = copy.deepcopy(source)
        profile['world_registration'] = registration
        hulls = [polygon(p) for p in profile['checked_card_polygons_mm']]
        common = hulls[0]
        for hull in hulls[1:]:
            area, common = cv2.intersectConvexConvex(common, hull)
            need(area > 1 and common is not None, 'legacy_domains_have_no_intersection')
            common = common.reshape(-1, 2)
        center = common.mean(0)
        card_xy = np.vstack((common, center))
        world_xy = np.array([map_xy(inverse, xy) for xy in card_xy])
        height = profile['object_height_mm']
        need(station['height_range_mm'][0] <= height <= station['height_range_mm'][1], 'legacy_height_outside_station')
        control = legacy_control(dict(surface_height_mm=height, world_xyz_mm=world_xy[-1].tolist()+[height],
                    world_frame_id=station['world_frame_id']), profile)
        inside = [cv2.pointPolygonTest(polygon(station['work_polygon_world_xy_mm']), tuple(xy), False) >= 0
                  for xy in world_xy]
        domains[name] = dict(profile_id=profile['profile_id'], profile_sha256=fingerprint(source),
            height_mm=height, card_xy_mm=card_xy.tolist(), world_xyz_mm=np.c_[world_xy, np.full(len(world_xy), height)].tolist(),
            center_reference_control=control, inside_declared_work_polygon=all(inside),
            automatic_motion_allowed=False)
    return domains


def reuse_preview(pose, camera, station, domains):
    result = {}
    for name, domain in domains.items():
        pixels = world_to_pixels(domain['world_xyz_mm'], pose, camera, station)
        # Round-trip is an arithmetic check, never independent physical accuracy.
        restored = [registration_ray_point(p, domain['height_mm'], pose, camera, station)['world_xyz_mm']
                    for p in pixels]
        error = float(np.linalg.norm(np.asarray(restored)-domain['world_xyz_mm'], axis=1).max())
        need(error <= .01, 'legacy_projection_roundtrip_failed')
        result[name] = dict(domain, projected_pixels=pixels, arithmetic_roundtrip_max_mm=error,
                            physical_accuracy_verified=False)
    return result


class FixedMarkerTracker:
    """Continuously require fresh independent frames; never expose an old valid pose."""
    def __init__(self, camera, station, registration, profiles, clock=time.monotonic):
        self.camera, self.station = camera, station
        self.registration = registration
        self.clock = clock
        self.domains = legacy_domains(registration, profiles, station)
        self.task_points = [p for d in self.domains.values() for p in d['world_xyz_mm']]
        self.rows = deque(maxlen=3)
        self.stream = None; self.last_id = -1; self.last_received = None
        self.expiry = None; self.source_age = 0
        self.pose = None; self.result = None; self.anchor = None; self.generation = 0
        self.reason = 'waiting_for_three_marker_frames'; self.visible_ids = []
        self.stream_changed = False

    def invalidate(self, reason, clear=True):
        self.pose = None; self.result = None; self.reason = reason
        if clear: self.rows.clear()

    def update(self, row):
        self.last_received = self.clock()
        self.source_age = row['age']; self.expiry = self.last_received+max(0, (1500-row['age'])/1000)
        if self.stream is not None and self.stream != row['stream']:
            self.stream_changed = True
        if self.stream_changed:
            self.invalidate('camera_stream_changed_recheck_settings'); return
        self.stream = row['stream']
        if row['frame_id'] <= self.last_id:
            self.invalidate('waiting_for_new_frame'); return
        self.last_id = row['frame_id']
        if not 0 <= row['age'] <= 1500:
            self.invalidate('camera_frame_stale'); return
        try:
            markers = detect_markers(row['image'], self.camera)
            known = {m['id'] for m in self.station['markers']}
            self.visible_ids = sorted(m['id'] for m in markers if m['id'] in known)
            need(len(self.visible_ids) >= 3, 'need_three_visible_fixed_markers')
            self.rows.append(dict(row, markers=markers))
            need(len(self.rows) == 3, 'waiting_for_three_marker_frames')
            records = [dict(image_sha256=r['sha'], detections=r['markers'], source_frame_id=r['frame_id'],
                            source_stream_id=r['stream'], source_age_ms=r['age']) for r in self.rows]
            pose = localize_records(records, self.camera, self.station, 'fixed_marker_task_space',
                                    stability_points_world_mm=self.task_points or None)
            preview = reuse_preview(pose, self.camera, self.station, self.domains)
            current = {m['id']:np.asarray(m['corners_px']) for m in markers if m['id'] in known}
            common = set(current) & set(self.anchor or {})
            changed = self.anchor is None or len(common) < 3 or max(
                float(np.linalg.norm(current[sid]-self.anchor[sid], axis=1).max()) for sid in common) > 1
            if changed:
                self.generation += 1; self.anchor = current
            self.pose = pose
            self.result = dict(kind='station-live-localization-preview-v1', ready=True,
                camera_generation=self.generation, frame_id=row['frame_id'], stream_id=row['stream'],
                image_sha256=row['sha'], camera_sha256=fingerprint(self.camera), station_sha256=fingerprint(self.station),
                registration_sha256=fingerprint(self.registration), pose=pose, legacy_domains=preview,
                automatic_motion_allowed=False, physical_accuracy_verified=False, production_integrated=False)
            self.reason = None
        except ValueError as exc:
            reason = str(exc)
            self.invalidate(reason, clear=reason in ('need_three_visible_fixed_markers', 'image_size_changed'))

    def snapshot(self):
        if self.expiry is not None and self.clock() > self.expiry:
            self.invalidate('camera_frame_stale')
        if self.result:
            return dict(self.result, frame_age_ms=self.source_age+(self.clock()-self.last_received)*1000)
        return dict(kind='station-live-localization-preview-v1', ready=False,
            reason=self.reason, visible_marker_ids=self.visible_ids, automatic_motion_allowed=False,
            production_integrated=False)
