import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.station_calibration import generate_kit, calibrate_intrinsics, write_json
from src.station_geometry import (camera_parameters, detect_markers, estimate_pose,
    fingerprint, fit_robot_affine, fit_world_registration, import_legacy, legacy_control, localize_records,
    pixel_at_height, refit_recorded_plane, station_parameters, world_to_pixels)
from src.static_supervised_trial import load_plan


class StationGeometryTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)/'kit'
        generate_kit(self.root)
        self.station = json.loads((self.root/'station.design.json').read_text())
        self.camera = dict(kind='station-camera-v1', camera_id='synthetic-camera',
            lens_profile_id='fixed', image_size=[640, 480], camera_matrix=[
                [650., 0., 320.], [0., 650., 240.], [0., 0., 1.]], dist_coeffs=[-.08, .01, 0., 0., 0.])

    def observations(self, center=(0, -40, 650), vector=(2.9, .1, -.05)):
        k, d = camera_parameters(self.camera)
        rotation = cv2.Rodrigues(np.array(vector, float))[0]
        translation = -rotation @ np.array(center, float)
        return [dict(id=m['id'], corners_px=cv2.projectPoints(np.asarray(m['corners_world_mm'], float),
            np.array(vector, float), translation, k, d)[0].reshape(4, 2).tolist()) for m in self.station['markers']]

    def pose(self, center=(0, -40, 650), vector=(2.9, .1, -.05)):
        records = [dict(image_sha256=hashlib.sha256(str(i).encode()).hexdigest(),
            detections=self.observations((center[0]+i*.1, center[1], center[2]), vector)) for i in range(3)]
        return localize_records(records, self.camera, self.station)

    def test_camera_relocation_recovers_same_point_at_supplied_height(self):
        for center, vector in (((0, -40, 650), (2.9, .1, -.05)), ((60, -10, 700), (3., -.08, .04))):
            pose = self.pose(center, vector)
            for height in (0, 53, 56, 65):
                xyz = [10., 10., float(height)]
                k, d = camera_parameters(self.camera)
                r = cv2.Rodrigues(np.array(vector, float))[0]
                actual_center = np.asarray(center)+[.2, 0, 0]
                pixel = cv2.projectPoints(np.asarray([xyz]), np.asarray(vector, float), -r @ actual_center, k, d)[0].ravel()
                point = pixel_at_height(pixel, height, pose, self.camera, self.station)
                np.testing.assert_allclose(point['world_xyz_mm'], xyz, atol=1e-5)
                self.assertFalse(point['automatic_motion_allowed'])

    def test_wrong_height_changes_xy_and_is_not_model_z(self):
        pose = self.pose()
        px = world_to_pixels([[10., 10., 65.]], pose, self.camera, self.station)[0]
        wrong = pixel_at_height(px, 0, pose, self.camera, self.station)
        self.assertGreater(np.linalg.norm(np.array(wrong['world_xyz_mm'][:2])-[10, 10]), 1)

    def test_three_of_four_markers_support_partial_occlusion(self):
        pose = estimate_pose(self.observations()[:3], self.camera, self.station)

    def test_explicit_two_marker_pose_retains_reprojection_and_stability_checks(self):
        detections=self.observations()[:2]
        with self.assertRaisesRegex(ValueError,'not_enough_visible'):
            estimate_pose(detections,self.camera,self.station)
        pose=estimate_pose(detections,self.camera,self.station,min_fixed_markers=2)
        self.assertEqual(len(pose['marker_ids']),2)
        self.assertLess(pose['reprojection_rms_px'],.8)
        rows=[dict(image_sha256=hashlib.sha256(str(i).encode()).hexdigest(),detections=detections) for i in range(3)]
        result=localize_records(rows,self.camera,self.station,'fixed_marker_task_space',min_fixed_markers=2)
        self.assertEqual(len(result['marker_ids']),2)
        with self.assertRaisesRegex(ValueError,'not_enough_visible'):
            estimate_pose(detections[:1],self.camera,self.station,min_fixed_markers=2)
        self.assertLess(pose['reprojection_max_px'], 1e-4)
        with self.assertRaisesRegex(ValueError, 'not_enough'):
            estimate_pose(self.observations()[:2], self.camera, self.station)

    def test_duplicate_unknown_or_mismatched_marker_layout_rejected(self):
        detections = self.observations()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            estimate_pose(detections+[detections[0]], self.camera, self.station)
        detections[0]['id'] = 40; detections[1]['id'] = 41
        with self.assertRaisesRegex(ValueError, 'not_enough'):
            estimate_pose(detections, self.camera, self.station)
        bad = self.observations()
        bad[0]['corners_px'][0][0] += 10
        with self.assertRaises(ValueError): estimate_pose(bad, self.camera, self.station)

    def test_frame_stability_and_duplicate_recordings_rejected(self):
        records = [dict(image_sha256=hashlib.sha256(str(i).encode()).hexdigest(),
            detections=self.observations((i*5, -40, 650))) for i in range(3)]
        with self.assertRaisesRegex(ValueError, 'not_stable'):
            localize_records(records, self.camera, self.station)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            localize_records([records[0]]*3, self.camera, self.station)

    def test_fixed_registration_checks_task_coordinates_despite_pose_center_noise(self):
        rows = [dict(image_sha256=hashlib.sha256(str(i).encode()).hexdigest(),
                     detections=self.observations()) for i in range(3)]
        base = estimate_pose(rows[0]['detections'], self.camera, self.station)
        # Coupled planar-fit uncertainty preserves table rays but changes the
        # estimated camera centre. It must remain bounded at supplied heights.
        poses = []
        for epsilon in (0, .0016, .0032):
            p = copy.deepcopy(base)
            r = np.asarray(base['rotation_world_to_camera']) @ cv2.Rodrigues(np.array([0., epsilon, 0.]))[0]
            c = np.asarray(base['camera_center_world_mm'])+[-650*epsilon, 0, 0]
            p.update(rotation_world_to_camera=r.tolist(), camera_center_world_mm=c.tolist(),
                     translation_world_to_camera_mm=(-r@c).tolist())
            poses.append(p)
        with patch('src.station_geometry.estimate_pose', side_effect=poses):
            with self.assertRaisesRegex(ValueError, 'camera_not_stable'):
                localize_records(rows, self.camera, self.station)
        with patch('src.station_geometry.estimate_pose', side_effect=poses+[base]):
            result = localize_records(rows, self.camera, self.station, 'fixed_marker_task_space')
        quality = result['stability_quality']
        self.assertGreater(quality['observed_camera_center_spread_mm'], 2)
        self.assertLess(quality['max_height_plane_point_spread_mm'], 1)
        self.assertEqual(quality['evaluated_height_range_mm'], [0, 120])
        self.assertFalse(result['automatic_motion_allowed'])
        bad = copy.deepcopy(poses[0]); bad['camera_center_world_mm'][0] += 1.1
        with patch('src.station_geometry.estimate_pose', side_effect=[base, bad, base, base]):
            with self.assertRaisesRegex(ValueError, 'task_points_unstable'):
                localize_records(rows, self.camera, self.station, 'fixed_marker_task_space')

    def test_fixed_registration_rejects_observed_marker_movement(self):
        rows = [dict(image_sha256=hashlib.sha256(str(i).encode()).hexdigest(),
                     detections=self.observations()) for i in range(3)]
        for marker in rows[1]['detections']:
            marker['corners_px'] = (np.asarray(marker['corners_px'])+[.6, 0]).tolist()
        with self.assertRaisesRegex(ValueError, 'moved_over_half_pixel'):
            localize_records(rows, self.camera, self.station, 'fixed_marker_task_space')

    def test_initial_five_pixel_margin_keeps_millimetre_quality_check(self):
        rows=[dict(image_sha256=hashlib.sha256(str(i).encode()).hexdigest(),
                   detections=self.observations()) for i in range(3)]
        base=estimate_pose(rows[0]['detections'],self.camera,self.station)
        for marker in rows[1]['detections']:
            marker['corners_px']=(np.asarray(marker['corners_px'])+[4,0]).tolist()
        with patch('src.station_geometry.estimate_pose',return_value=base):
            result=localize_records(rows,self.camera,self.station,'fixed_marker_task_space',max_marker_shift_px=5)
        self.assertEqual(result['stability_quality']['max_marker_shift_px'],5)
        for marker in rows[1]['detections']:
            marker['corners_px']=(np.asarray(marker['corners_px'])+[2,0]).tolist()
        with patch('src.station_geometry.estimate_pose',return_value=base):
            with self.assertRaisesRegex(ValueError,'moved_over_initial_limit'):
                localize_records(rows,self.camera,self.station,'fixed_marker_task_space',max_marker_shift_px=5)
        rows[1]['detections']=self.observations()
        bad=copy.deepcopy(base);bad['camera_center_world_mm'][0]+=1.1
        with patch('src.station_geometry.estimate_pose',side_effect=[base,bad,base,base]):
            with self.assertRaisesRegex(ValueError,'task_points_unstable'):
                localize_records(rows,self.camera,self.station,'fixed_marker_task_space',max_marker_shift_px=5)

    def test_explicit_threefold_review_allows_two_mm_but_rejects_over_three(self):
        rows=[dict(image_sha256=hashlib.sha256(str(i).encode()).hexdigest(),
                   detections=self.observations()) for i in range(3)]
        base=estimate_pose(rows[0]['detections'],self.camera,self.station)
        for delta,allowed in [(2,True),(3.1,False)]:
            bad=copy.deepcopy(base);bad['camera_center_world_mm'][0]+=delta
            with patch('src.station_geometry.estimate_pose',side_effect=[base,bad,base,base]):
                if allowed:
                    result=localize_records(rows,self.camera,self.station,'fixed_marker_task_space',review_factor=3)
                    self.assertEqual(result['stability_quality']['max_task_point_spread_mm'],3)
                else:
                    with self.assertRaisesRegex(ValueError,'task_points_unstable_over_limit'):
                        localize_records(rows,self.camera,self.station,'fixed_marker_task_space',review_factor=3)

    def test_camera_lens_and_station_changes_invalidate_pose(self):
        pose = self.pose()
        for field in ('camera_id', 'lens_profile_id'):
            camera = copy.deepcopy(self.camera); camera[field] = 'changed'
            with self.assertRaisesRegex(ValueError, 'changed'):
                pixel_at_height([320, 240], 56, pose, camera, self.station)
        station = copy.deepcopy(self.station); station['world_frame_id'] = 'different-origin'
        with self.assertRaisesRegex(ValueError, 'changed'):
            pixel_at_height([320, 240], 56, pose, self.camera, station)

    def test_invalid_height_and_outside_work_area_rejected(self):
        pose = self.pose()
        for height in (-1, 121, float('nan'), True):
            with self.assertRaises(ValueError): pixel_at_height([320, 240], height, pose, self.camera, self.station)
        px = world_to_pixels([[80., 0., 56.]], pose, self.camera, self.station)[0]
        with self.assertRaisesRegex(ValueError, 'work_polygon'):
            pixel_at_height(px, 56, pose, self.camera, self.station)

    def test_invalid_camera_and_marker_geometry_rejected(self):
        for value in (None, [[0]*3]*3):
            bad = copy.deepcopy(self.camera); bad['camera_matrix'] = value
            with self.assertRaises(ValueError): camera_parameters(bad)
        bad = copy.deepcopy(self.station); bad['markers'][0]['corners_world_mm'][0][2] = 1
        with self.assertRaises(ValueError): station_parameters(bad)

    def legacy(self):
        source = dict(object_height_mm=56, image_to_card_at_object_surface=np.eye(3).tolist(),
            image_to_grip_mm=[[1, 0, 0], [0, 1.25, 145], [0, 0, 1]],
            image_to_hover_mm=[[1, 0, 0], [0, .75, 150], [0, 0, 1]],
            grip_model_z_mm=10, hover_model_z_mm=70,
            verified_pick_hulls_card_xy_mm=[[[-20, 0], [20, 0], [0, 12]]]*2)
        return source, import_legacy(source, 'a'*64, 'can56')

    def test_legacy_camera_mapping_excluded_and_target_preserved(self):
        source, profile = self.legacy()
        self.assertNotIn('image_to_card_at_object_surface', profile)
        self.assertNotIn('image_to_grip_mm', profile)
        np.testing.assert_allclose(profile['card_to_grip'], source['image_to_grip_mm'])
        point = dict(surface_height_mm=56, world_xyz_mm=[0, 4, 56], world_frame_id='board')
        with self.assertRaisesRegex(ValueError, 'registration_required'): legacy_control(point, profile)
        profile['world_registration'] = dict(world_frame_id='board', checked=True, world_to_card_xy=np.eye(3).tolist())
        mapped = legacy_control(point, profile)
        self.assertEqual(mapped['grip_xyz_nominal'], [0, 150, 10])
        self.assertFalse(mapped['automatic_motion_allowed'])

    def test_legacy_height_domain_and_scale_cannot_be_extended(self):
        _, profile = self.legacy()
        profile['world_registration'] = dict(world_frame_id='board', checked=True, world_to_card_xy=np.eye(3).tolist())
        for height, xy, reason in ((53, [0, 4], 'height_mismatch'), (56, [30, 0], 'checked_domain')):
            with self.assertRaisesRegex(ValueError, reason):
                legacy_control(dict(surface_height_mm=height, world_xyz_mm=xy+[height], world_frame_id='board'), profile)
        profile['world_registration']['world_to_card_xy'][0][0] = 2
        with self.assertRaisesRegex(ValueError, 'scale'):
            legacy_control(dict(surface_height_mm=56, world_xyz_mm=[0, 4, 56], world_frame_id='board'), profile)

    def test_rigid_registration_reserves_independent_points(self):
        data = dict(world_frame_id='board', world_xy_mm=[[0, 0], [20, 0], [0, 20]],
            card_xy_mm=[[5, 10], [25, 10], [5, 30]],
            check_world_xy_mm=[[10, 10], [5, 5]], check_card_xy_mm=[[15, 20], [10, 15]])
        result = fit_world_registration(data)
        self.assertLess(result['check_max_mm'], 1e-9)
        data['check_world_xy_mm'][0] = [0, 0]
        with self.assertRaisesRegex(ValueError, 'overlap'): fit_world_registration(data)

    def recorded(self):
        corners = {letter+str(row): [320+(col-2)*20, 180+(row-2)*20]
                   for row in range(1, 5) for col, letter in enumerate('ABCDE')}
        return dict(corner_pixels=corners, card_origin='C2', axes='+X card right, +Y toward TOP',
            image_sha256='b'*64, camera_id='recorded-camera', image_size=[640, 480])

    def test_refit_uses_disjoint_corners_and_keeps_height_identity(self):
        result = refit_recorded_plane(self.recorded(), 53)
        self.assertEqual(result['surface_height_mm'], 53)
        self.assertFalse(set(result['fit_labels']) & set(result['check_labels']))
        self.assertEqual(len(result['fit_labels']), 12)
        self.assertEqual(len(result['check_labels']), 8)
        self.assertLess(result['check_max_mm'], 1e-8)
        self.assertFalse(result['camera_intrinsics_estimated'])
        self.assertFalse(result['independent_physical_validation'])

    def test_refit_rejects_outlier_or_wrong_axes(self):
        source = self.recorded(); source['corner_pixels']['B1'][0] += 10
        with self.assertRaisesRegex(ValueError, 'residual'): refit_recorded_plane(source, 53)
        source = self.recorded(); source['axes'] = 'unknown'
        with self.assertRaises(ValueError): refit_recorded_plane(source, 53)

    def test_robot_training_preserves_height_and_uncertainty(self):
        data = dict(card_xy_mm=[[0, 0], [20, 0], [0, 12]], nominal_xy_mm=[[0, 145], [20, 145], [0, 160]],
            check_card_xy_mm=[[-20, 0]], check_nominal_xy_mm=[[-20, 145]],
            object_height_mm=56, fixture_height_mm=0, model_z_mm=10, alpha_deg=-75)
        fit = fit_robot_affine(data)
        np.testing.assert_allclose(fit['card_to_nominal'], [[1, 0, 0], [0, 1.25, 145], [0, 0, 1]], atol=1e-10)
        self.assertFalse(fit['physical_error_quantitatively_known'])
        self.assertEqual(fit['object_height_mm'], 56)
        data['check_card_xy_mm'] = [[0, 0]]
        with self.assertRaisesRegex(ValueError, 'overlap'): fit_robot_affine(data)

    def test_generated_kit_has_exact_physical_size_and_detectable_codes(self):
        text = (self.root/'fixed-markers-a4.svg').read_text()
        self.assertIn('width="210mm" height="297mm"', text)
        for m in self.station['markers']:
            self.assertAlmostEqual(np.linalg.norm(np.array(m['corners_world_mm'][0])-m['corners_world_mm'][1]), 45)
        canvas = np.full((480, 640, 3), 255, np.uint8)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        canvas[100:220, 100:220] = cv2.cvtColor(cv2.aruco.generateImageMarker(dictionary, 10, 120), cv2.COLOR_GRAY2BGR)
        self.assertEqual([d['id'] for d in detect_markers(canvas, self.camera)], [10])

    def test_output_never_overwrites_or_loads_as_motion(self):
        with self.assertRaises(FileExistsError): generate_kit(self.root)
        path = self.root/'pose.json'; write_json(path, self.pose())
        with self.assertRaises(FileExistsError): write_json(path, {})
        with self.assertRaises(ValueError): load_plan(path, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_insufficient_intrinsics_images_refused(self):
        manifest = json.loads((self.root/'intrinsics-manifest.pending.json').read_text())
        with self.assertRaisesRegex(ValueError, '12_fit'): calibrate_intrinsics(manifest, self.root)

    def test_efficient_mode_cannot_reduce_minimum_below_ten(self):
        manifest = json.loads((self.root/'intrinsics-manifest.pending.json').read_text())
        for minimum in (9, True, 0):
            manifest['minimum_fit_images'] = minimum
            with self.assertRaisesRegex(ValueError, '10_to_30'): calibrate_intrinsics(manifest, self.root)


if __name__ == '__main__': unittest.main()
