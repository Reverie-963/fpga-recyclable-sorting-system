import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.station_calibration import calibrate_intrinsics


class CameraModelTests(unittest.TestCase):
    def test_four_coefficient_model_uses_same_constraints_in_every_refit(self):
        camera = np.array([[650., 0, 320.], [0, 650., 240.], [0, 0, 1.]])
        obj = np.zeros((20, 3), np.float32)
        obj[:, :2] = np.mgrid[:5, :4].T.reshape(-1, 2)*20
        points = []
        for i in range(15):
            r = np.array([.16*(i%5-2), .19*(i//5-1), .025*(i%3-1)])
            t = np.array([125.*(i%3-1), 50.*(i%4-1.5), 500.+(i%3)*20])
            points.append(cv2.projectPoints(obj, r, t, camera, np.array([-.05, .01, .001, -.001, 0.]))[0].astype(np.float32))
        manifest = dict(camera_id='synthetic', lens_profile_id='fixed', image_size=[640, 480],
            pattern_inner_corners=[5, 4], square_mm=20, fit_images=[str(i) for i in range(12)],
            check_images=[str(i) for i in range(12, 15)], calibration_model='brown4', require_leave_one_out_stability=True)
        def image(path):
            i = int(path.name)
            return np.full((480, 640, 3), i, np.uint8), hashlib.sha256(str(i).encode()).hexdigest()
        original = cv2.calibrateCamera
        with patch('src.station_calibration.read_image', side_effect=image), \
             patch('src.station_calibration.cv2.findChessboardCornersSB', side_effect=lambda gray, pattern:(True, points[int(gray[0, 0])])), \
             patch('src.station_calibration.cv2.calibrateCamera', wraps=original) as fit:
            result = calibrate_intrinsics(manifest, Path('.'))
        self.assertEqual(fit.call_count, 13)
        self.assertTrue(all(call.kwargs['flags'] == cv2.CALIB_FIX_K3 for call in fit.call_args_list))
        self.assertEqual(result['fixed_coefficients'], {'k3': 0.})
        self.assertEqual(result['dist_coeffs'][4], 0.)
        self.assertLess(result['check_rms_px'], .001)
        self.assertFalse(result['automatic_motion_allowed'])
        np.testing.assert_allclose(np.array(result['camera_matrix'])[:2, :2], camera[:2, :2], atol=.1)

    def test_unknown_model_fails_before_image_access(self):
        manifest = dict(fit_images=['unused']*12, check_images=['unused']*3, calibration_model='unknown')
        with self.assertRaisesRegex(ValueError, 'unsupported_calibration_model'):
            calibrate_intrinsics(manifest, Path('.'))


if __name__ == '__main__': unittest.main()
