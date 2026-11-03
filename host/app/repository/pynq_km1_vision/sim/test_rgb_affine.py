import unittest
import numpy as np
from src.rgb_affine import affine_lut,apply_rgb_affine_u8


class AffineTests(unittest.TestCase):
    def check_values(self,gain,bias):
        values=np.arange(256,dtype=np.uint8).reshape(16,16)
        image=np.stack((values,np.flipud(values),np.fliplr(values)),axis=2)
        expected=np.clip(image.astype(float)*gain+bias,0,255).astype(np.uint8)
        actual=apply_rgb_affine_u8(image,gain,bias)
        np.testing.assert_array_equal(actual,expected)
        self.assertFalse(np.shares_memory(actual,image))

    def test_all_byte_values_clipping_and_integer_boundaries(self):
        for gain,bias in (([1,1,1],[0,0,0]),([1.3,.7,1.01],[32,-32,-2]),
                          ([-.2,2.2,0],[255,-80,128]),
                          ([1+1e-15,1-1e-15,.3333333333333333],[-1e-13,1e-13,127.99999999999999])):
            self.check_values(gain,bias)

    def test_random_coefficients_match_float64_reference_exhaustively(self):
        rng=np.random.default_rng(216)
        for _ in range(100):self.check_values(rng.uniform(-.5,2.5,3),rng.uniform(-100,100,3))

    def test_non_contiguous_camera_roi(self):
        image=np.random.default_rng(2).integers(0,256,(48,64,3),dtype=np.uint8)[3:36,4:42]
        self.assertFalse(image.flags.c_contiguous)
        gain,bias=[1.2,.8,1.01],[-2,3,4]
        np.testing.assert_array_equal(apply_rgb_affine_u8(image,gain,bias),np.clip(image.astype(float)*gain+bias,0,255).astype(np.uint8))

    def test_rejects_invalid_parameters_and_images(self):
        for gain,bias in (([1,2],[0,0,0]),([1,1,float('nan')],[0,0,0]),([1,1,1],[0,float('inf'),0])):
            with self.assertRaises(ValueError):affine_lut(gain,bias)
        for image in (np.zeros((2,3),np.uint8),np.zeros((2,3,3),float),np.zeros((0,3,3),np.uint8)):
            with self.assertRaises(ValueError):apply_rgb_affine_u8(image,[1,1,1],[0,0,0])

    def test_photometry_checks_and_evidence_match_original_transform(self):
        from unittest.mock import patch
        from sim.test_sorting_known import config
        from src.sorting_demo import DemoClock,DemoSource
        from src.sorting_scene import adjust_exposure_reference
        c=config();source=DemoSource(c,DemoClock());before=source.reference
        after=np.clip(before.astype(float)*.9+12,0,255).astype(np.uint8)
        after[50:60,40:50]=255
        policy=dict(c['layout'],image_size=c['image_size'])
        actual,evidence=adjust_exposure_reference(before,after,policy)
        legacy=lambda image,gain,bias:np.clip(image.astype(float)*gain+bias,0,255).astype(np.uint8)
        with patch('src.rgb_affine.apply_rgb_affine_u8',side_effect=legacy):
            expected,old_evidence=adjust_exposure_reference(before,after,policy)
        np.testing.assert_array_equal(actual,expected);self.assertEqual(evidence,old_evidence)
        with self.assertRaises(ValueError):adjust_exposure_reference(before,np.roll(after,3,axis=0),policy)


if __name__=='__main__':unittest.main()
