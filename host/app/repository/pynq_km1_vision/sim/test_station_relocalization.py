import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.station_calibration import generate_kit
from src.station_geometry import camera_parameters, estimate_pose, localize_records, map_xy, world_to_pixels
from src.station_guided import GuidedSession
from src.station_relocalization import FixedMarkerTracker, legacy_domains, reuse_preview


class RelocalizationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); generate_kit(self.root/'kit')
        self.station = json.loads((self.root/'kit/station.design.json').read_text())
        self.camera = dict(kind='station-camera-v1', camera_id='synthetic', lens_profile_id='fixed',
            image_size=[640, 480], camera_matrix=[[650., 0, 320], [0, 650., 240], [0, 0, 1]], dist_coeffs=[0.]*5)
        self.registration = dict(checked=True, world_frame_id=self.station['world_frame_id'],
            world_to_card_xy=[[1., 0, -10.], [0, 1., 5.], [0, 0, 1.]])
        profile = dict(kind='station-legacy-profile-v1', profile_id='synthetic65', object_height_mm=65,
            checked_card_polygons_mm=[[[-20., 0], [20., 0], [0, 10.]], [[-20., 0], [20., 0], [0, 20.]]],
            card_to_grip=[[1., 0, 0], [0, .75, 145.], [0, 0, 1.]], card_to_hover=[[1., 0, 0], [0, .8, 150.], [0, 0, 1.]],
            grip_model_z_mm=50., hover_model_z_mm=70.)
        self.profiles = {k:dict(profile, profile_id=k, object_height_mm=h) for k,h in [('paper',65),('metal',56),('plastic',53)]}
        self.now = 0.; self.tracker = FixedMarkerTracker(self.camera, self.station, self.registration,
                                                        self.profiles, clock=lambda:self.now)

    def detections(self, offset=0):
        rv = np.array([3., .05, -.02]); r = cv2.Rodrigues(rv)[0]
        tv = -r@np.array([offset, -40., 650.]); k,d=camera_parameters(self.camera)
        return [dict(id=m['id'], corners_px=cv2.projectPoints(np.array(m['corners_world_mm'],float),rv,tv,k,d)[0]
                     .reshape(4,2).tolist()) for m in self.station['markers']]

    def row(self, i, stream='fixed', age=10):
        raw=str(i).encode()
        return dict(frame_id=i, stream=stream, age=age, sha=hashlib.sha256(raw).hexdigest(), raw=raw,
                    image=np.full((480,640,3),200,np.uint8))

    def settle(self, start=1, offset=0):
        with patch('src.station_relocalization.detect_markers', return_value=self.detections(offset)):
            for i in range(start,start+3): self.tracker.update(self.row(i))
        return self.tracker.snapshot()

    def test_marker_only_ready_without_chessboard_and_height_specific_maps(self):
        result=self.settle()
        self.assertTrue(result['ready']); self.assertFalse(result['automatic_motion_allowed'])
        self.assertFalse(result['production_integrated'])
        self.assertEqual(len(result['pose']['evidence']),3)
        self.assertEqual([d['height_mm'] for d in result['legacy_domains'].values()], [65,56,53])
        self.assertGreater(result['pose']['stability_quality']['evaluated_task_point_count'],10)
        self.assertEqual(result['frame_id'],3)

    def test_moving_camera_invalidates_immediately_then_recovers_without_button(self):
        self.settle()
        with patch('src.station_relocalization.detect_markers', return_value=self.detections(20)):
            self.tracker.update(self.row(4)); self.assertFalse(self.tracker.snapshot()['ready'])
            self.assertIsNone(self.tracker.pose)
            self.tracker.update(self.row(5)); self.tracker.update(self.row(6))
        result=self.tracker.snapshot()
        self.assertTrue(result['ready']);self.assertEqual(result['camera_generation'],2)

    def test_occlusion_stale_frames_duplicates_and_stream_change_never_reuse_pose(self):
        self.settle()
        self.now=2.;self.assertFalse(self.tracker.snapshot()['ready'])
        self.now=2.1;self.settle(4)
        with patch('src.station_relocalization.detect_markers', return_value=self.detections()[:2]):
            self.tracker.update(self.row(7));self.assertFalse(self.tracker.snapshot()['ready'])
        self.settle(8)
        self.tracker.update(self.row(10));self.assertFalse(self.tracker.snapshot()['ready'])
        self.settle(11)
        self.tracker.update(self.row(14,age=2000));self.assertFalse(self.tracker.snapshot()['ready'])
        self.settle(15)
        self.tracker.update(self.row(18,stream='changed'));self.assertFalse(self.tracker.snapshot()['ready'])
        self.tracker.update(self.row(19));self.assertFalse(self.tracker.snapshot()['ready'])

    def test_projection_changes_but_robot_mapping_does_not_across_camera_poses(self):
        original=self.settle()
        moved=self.settle(4,20)
        for material in self.profiles:
            a=original['legacy_domains'][material];b=moved['legacy_domains'][material]
            self.assertGreater(np.linalg.norm(np.array(a['projected_pixels'])-b['projected_pixels']),1)
            self.assertEqual(a['center_reference_control'],b['center_reference_control'])
            self.assertLess(b['arithmetic_roundtrip_max_mm'],1e-5)
        self.assertEqual(self.profiles['paper'].get('world_registration'),None)

    def test_preview_keeps_outside_station_domain_explicit(self):
        reg=copy.deepcopy(self.registration);reg['world_to_card_xy'][0][2]=-140
        domains=legacy_domains(reg,self.profiles,self.station)
        self.assertTrue(all(not d['inside_declared_work_polygon'] for d in domains.values()))
        self.assertTrue(all(not d['automatic_motion_allowed'] for d in domains.values()))
        reg['world_to_card_xy'][0][0]=2
        with self.assertRaisesRegex(ValueError,'rigid'):legacy_domains(reg,self.profiles,self.station)

    def test_delivery_age_counts_towards_expiration(self):
        with patch('src.station_relocalization.detect_markers', return_value=self.detections()):
            for i in range(1,4): self.tracker.update(self.row(i,age=1400))
        self.assertTrue(self.tracker.snapshot()['ready'])
        self.now=.11;self.assertFalse(self.tracker.snapshot()['ready'])

    def test_guided_tracking_runs_even_when_old_chessboard_is_absent(self):
        history=self.root/'history';history.mkdir()
        bundle=dict(intrinsics_seeds=[],classes={})
        for material,profile in self.profiles.items():
            (history/(material+'.json')).write_text(json.dumps(profile))
            bundle['classes'][material]=dict(parameters=dict(height_mm=profile['object_height_mm']),
                historical_commands=1,profile_file=material+'.json')
        (history/'bundle.json').write_text(json.dumps(bundle))
        session=GuidedSession(self.root/'session',history,self.station,'http://127.0.0.1:8780/frame.jpg',
                              camera_id='synthetic',lens_profile='fixed')
        session.camera=self.camera;session.registration=self.registration;session._start_relocator()
        with patch('src.station_guided.inspect_board',return_value=(None,'no board')), \
             patch('src.station_relocalization.detect_markers',return_value=self.detections()):
            for i in range(1,4):session.process(self.row(i))
        self.assertTrue(session.localization()['ready'])
        self.assertEqual(len(list((self.root/'session').glob('relocate-*.json'))),1)
        session.relocator.invalidate('lost');self.assertFalse(session.localization()['ready'])
        # A restart preserves the checked bridge/verification but never revives
        # a previously ready frame without seeing three new independent frames.
        (session.root/'camera.json').write_text(json.dumps(self.camera))
        (session.root/'world-registration.json').write_text(json.dumps(self.registration))
        from src.station_geometry import fingerprint
        verification=dict(previous_registration_sha256=fingerprint(self.registration),move_check_max_mm=.2)
        (session.root/'verify-1-registration.json').write_text(json.dumps(verification))
        resumed=GuidedSession(session.root,history,self.station,'http://127.0.0.1:8780/frame.jpg',
                              camera_id='synthetic',lens_profile='fixed',resume=True)
        self.assertEqual(resumed.stage,'verified');self.assertFalse(resumed.localization()['ready'])
        self.assertAlmostEqual(resumed.state()['verification']['move_check_max_mm'],.2)


if __name__=='__main__':unittest.main()
