import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.station_calibration import generate_kit, write_json
from src.station_geometry import map_xy
from src.station_guided import GuidedSession, corner_distance, inspect_board, novel_board, registration_from_frames, stable_boards


def grid(offset=(0, 0)):
    return np.array([[280+col*20+offset[0], 140+row*20+offset[1]] for row in range(4) for col in range(5)], np.float32)


class GuidedCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        generate_kit(self.root/'kit')
        self.station = json.loads((self.root/'kit/station.design.json').read_text())
        history = self.root/'history'; history.mkdir()
        self.history = dict(intrinsics_seeds=[], classes={k: dict(parameters=dict(height_mm=h), historical_commands=n)
            for k, h, n in [('paper', 65, 17), ('metal', 56, 20), ('plastic', 53, 19)]})
        write_json(history/'bundle.json', self.history)
        self.session = GuidedSession(self.root/'session', history, self.station, 'http://127.0.0.1:8780/frame.jpg')

    def row(self, i, stream='fixed'):
        raw = ('original-frame-'+str(i)).encode()
        return dict(raw=raw, sha=hashlib.sha256(raw).hexdigest(),
            image=np.full((480, 640, 3), 200, np.uint8), frame_id=i, stream=stream, age=10)

    def test_start_paused_and_preserves_three_classes(self):
        state = self.session.state()
        self.assertFalse(state['auto']); self.assertFalse(state['automatic_motion_allowed'])
        self.assertEqual(sum(x['commands'] for x in state['classes'].values()), 56)
        self.assertFalse(list((self.root/'session/images').iterdir()))

    def test_resume_preserves_photos_split_and_pauses_capture(self):
        self.session._capture(dict(self.row(1), corners=grid()), 'fit')
        self.session._capture(dict(self.row(2), corners=grid((20, 0))), 'check')
        before = {p.name:p.read_bytes() for p in (self.root/'session/images').iterdir()}
        resumed = GuidedSession(self.root/'session', self.root/'history', self.station,
            'http://127.0.0.1:8780/frame.jpg', resume=True)
        self.assertEqual(len(resumed.fit), 1); self.assertEqual(len(resumed.checks), 1)
        self.assertFalse(resumed.auto); self.assertEqual(resumed.stage, 'review_existing')
        self.assertEqual(before, {p.name:p.read_bytes() for p in (self.root/'session/images').iterdir()})

    def test_resume_rejects_changed_photograph(self):
        self.session._capture(dict(self.row(1), corners=grid()), 'fit')
        (self.root/'session'/self.session.fit[0]).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'photo_or_lens_changed'):
            GuidedSession(self.root/'session', self.root/'history', self.station,
                'http://127.0.0.1:8780/frame.jpg', resume=True)

    def test_quality_failure_does_not_start_another_three_photo_loop(self):
        self.session.stage = 'need_more'; goal = self.session.goal_fit
        with self.assertRaises(ValueError): self.session.begin()
        self.assertEqual(self.session.goal_fit, goal); self.assertFalse(self.session.auto)

    def test_auto_capture_needs_stability_and_new_pose(self):
        self.session.begin()
        with patch('src.station_guided.inspect_board', return_value=(grid(), '')):
            self.session.process(self.row(1)); self.session.process(self.row(2))
            self.assertEqual(len(self.session.fit), 0)
            self.session.process(self.row(3))
            self.assertEqual(len(self.session.fit), 1)
            for i in range(4, 9): self.session.process(self.row(i))
            self.assertEqual(len(self.session.fit), 1)
        with patch('src.station_guided.inspect_board', return_value=(grid((25, 0)), '')):
            for i in range(9, 12): self.session.process(self.row(i))
        self.assertEqual(len(self.session.fit), 2)
        self.assertEqual((self.root/'session'/self.session.fit[0]).read_bytes(), self.row(3)['raw'])

    def test_pause_and_camera_restart_prevent_capture(self):
        self.session.begin(); self.session.action('pause', {})
        with patch('src.station_guided.inspect_board', return_value=(grid(), '')):
            for i in range(1, 4): self.session.process(self.row(i))
            self.assertEqual(len(self.session.fit), 0)
            self.session.begin(); self.session.process(self.row(4, 'restarted'))
        self.assertFalse(self.session.auto)
        self.assertIn('重启', self.session.error)

    def test_duplicate_frame_id_does_not_make_stability_evidence(self):
        self.session.begin()
        with patch('src.station_guided.inspect_board', return_value=(grid(), '')):
            for i in range(5): self.session.process(self.row(1))
        self.assertEqual(len(self.session.rows), 1)
        self.assertEqual(len(self.session.fit), 0)

    def test_unrecognized_or_blurred_board_resets_stability(self):
        self.session.begin()
        with patch('src.station_guided.inspect_board', return_value=(grid(), '')):
            self.session.process(self.row(1)); self.session.process(self.row(2))
        with patch('src.station_guided.inspect_board', return_value=(None, '模糊')):
            self.session.process(self.row(3))
        self.assertEqual(len(self.session.rows), 0)
        self.assertEqual(self.session.note, '模糊')

    def test_reversed_detection_is_same_pose_not_a_new_photo(self):
        self.assertLess(corner_distance(grid(), grid()[::-1]), 1e-6)
        self.assertFalse(novel_board(grid()[::-1], [grid()]))
        self.assertFalse(stable_boards([dict(corners=grid()), dict(corners=grid((8, 0))), dict(corners=grid())]))

    def test_registration_requires_lens_and_explicit_restored_origin(self):
        with self.assertRaises(ValueError): self.session.action('register', {})
        self.session.camera = {'present': True}
        with self.assertRaises(ValueError): self.session.action('register', {'board_scale_confirmed': True})
        self.session.action('register', dict(board_scale_confirmed=True, old_card_restored=True))
        self.assertEqual(self.session.stage, 'register')

    def test_camera_url_cannot_access_external_or_arbitrary_local_routes(self):
        for index, url in enumerate(('http://example.com/frame.jpg', 'http://127.0.0.1:8780/admin', 'http://x:y@127.0.0.1/frame.jpg')):
            with self.assertRaises(ValueError):
                GuidedSession(self.root/('bad-'+str(index)), self.root/'history', self.station, url)

    def test_automatic_registration_uses_20_detected_points_and_8_checks(self):
        camera = dict(kind='station-camera-v1', camera_id='synthetic-only', lens_profile_id='fixed', image_size=[640, 480],
            camera_matrix=[[650., 0, 320], [0, 650., 240], [0, 0, 1]], dist_coeffs=[0.]*5)
        k = np.asarray(camera['camera_matrix']); rv = np.array([3., .05, -.02]); tv = np.array([0., 0., 650.])
        card = np.array([[(col-2)*20., (2-row)*20.] for row in range(1, 5) for col in range(5)])
        angle = .05; rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        world = (rotation @ card.T).T+[8, 4]
        pix = cv2.projectPoints(np.c_[world, np.zeros(20)], rv, tv, k, np.zeros(5))[0].reshape(20, 2)
        markers = [dict(id=m['id'], corners_px=cv2.projectPoints(np.asarray(m['corners_world_mm'], float), rv, tv, k,
            np.zeros(5))[0].reshape(4, 2).tolist()) for m in self.station['markers']]
        rows = [dict(sha=hashlib.sha256(str(i).encode()).hexdigest(), markers=markers, corners=pix) for i in range(3)]
        _, registration, pairs = registration_from_frames(rows, camera, self.station)
        self.assertEqual(len(pairs['world_xy_mm']), 12)
        self.assertEqual(len(pairs['check_world_xy_mm']), 8)
        self.assertFalse(registration['automatic_motion_allowed'])
        for xy, target in zip(world, card):
            np.testing.assert_allclose(map_xy(registration['world_to_card_xy'], xy), target, atol=1e-5)

    def test_live_board_detection_works_on_printable_pattern(self):
        image = np.full((480, 640, 3), 255, np.uint8)
        for row in range(5):
            for col in range(6):
                if (row+col)%2 == 0: image[130+row*30:160+row*30, 220+col*30:250+col*30] = 0
        points, reason = inspect_board(image)
        self.assertIsNotNone(points, reason)
        self.assertEqual(points.shape, (20, 2))


if __name__ == '__main__': unittest.main()
