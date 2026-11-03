import unittest
import numpy as np
from src.phase_pixel_reference import packed_frame,packed_lut,reference_evidence
from src.rgb_affine import apply_rgb_affine_u8


class PixelContractTests(unittest.TestCase):
    def test_bgr_and_eight_overlapping_masks_have_exact_word_positions(self):
        image=np.asarray([[[3,5,7],[255,0,1]]],dtype=np.uint8)
        masks=[np.asarray([[True,i%2==0]],dtype=bool) for i in range(8)]
        words=packed_frame(image,masks)
        self.assertEqual(int(words[0,0]),0xff070503)
        self.assertEqual(int(words[0,1]),0x550100ff)
        np.testing.assert_array_equal(image,np.asarray([[[3,5,7],[255,0,1]]],np.uint8))

    def test_lut_and_full_channel_sum_match_existing_affine_and_strict_boundary(self):
        before=np.tile(np.arange(256,dtype=np.uint8)[None,:,None],(2,1,3))
        gain,bias=[.7,1.1,1.3],[-32,17,32]
        current=apply_rgb_affine_u8(before,gain,bias);current[0,0]=0
        masks=[np.ones((2,256),bool)]
        roi=[[0,0,256,2]]*6
        actual=reference_evidence(before,current,masks,packed_lut(gain,bias),roi,45,81)
        gold=np.abs(current.astype(np.int16)-apply_rgb_affine_u8(before,gain,bias).astype(np.int16)).sum(2)
        np.testing.assert_array_equal(actual['changed'],gold>45)
        self.assertEqual(int(actual['moments'][0,0]),512)
        self.assertEqual(actual['moments'][0,6:].tolist(),[0,0,255,1])
        self.assertFalse(actual['moments'][1:].any())
        self.assertEqual(actual['regions'][0].tolist(),[int((gold>45).sum()),int(gold.sum())])
        self.assertFalse(actual['physical_success_verified'])

    def test_765_difference_and_thresholds_are_not_uint8_saturated(self):
        before=np.zeros((1,1,3),np.uint8);current=np.full_like(before,255)
        for threshold,expected in ((764,1),(765,0)):
            out=reference_evidence(before,current,[],packed_lut([1]*3,[0]*3),[[0,0,1,1]]*6,threshold,1)
            self.assertEqual(int(out['changed'][0,0]),expected)
            self.assertEqual(int(out['regions'][0,1]),765)

    def test_invalid_jobs_roi_masks_and_contracts_are_rejected(self):
        image=np.zeros((2,3,3),np.uint8);lut=packed_lut([1]*3,[0]*3);rois=[[0,0,3,2]]*6
        for job in (0,2**32,True):
            with self.assertRaisesRegex(ValueError,'job_id'):reference_evidence(image,image,[],lut,rois,45,job)
        with self.assertRaisesRegex(ValueError,'half_open'):reference_evidence(image,image,[],lut,[[0,0,4,2]]*6,45,1)
        with self.assertRaisesRegex(ValueError,'eight'):packed_frame(image,[np.zeros((2,3),bool)]*9)
        with self.assertRaisesRegex(ValueError,'bool_masks'):packed_frame(image,[np.zeros((2,3),np.uint8)])
