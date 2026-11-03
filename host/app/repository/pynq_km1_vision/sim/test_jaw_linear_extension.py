import copy
import unittest
from src.contour_grasp import jaw_command


class LinearExtensionTests(unittest.TestCase):
    def setUp(self):
        self.profile={'open_pwm':1000}
        self.policy=dict(jaw_open_span_mm=61,jaw_calibration_operator_statement='measured original',
            jaw_calibration=dict(kind='measured-jaw-gap-v1',measurement_id='synthetic-original-9mm',
                samples=[dict(gap_mm=9,pwm=1600),dict(gap_mm=13,pwm=1575),dict(gap_mm=61,pwm=1000)],
                compression_ratio=.2,compression_mm=0,opening_margin_mm=0),
            jaw_gap_extension=dict(kind='operator-supervised-linear-extension-v1',min_gap_mm=8,
                operator_statement='explicit operator proportional extension',physical_validation_completed=False))

    def test_8mm_is_estimated_1606_and_never_added_as_a_measured_sample(self):
        original=copy.deepcopy(self.policy)
        result=jaw_command(self.profile,self.policy,10)
        self.assertEqual(result['target_gap_mm'],8)
        self.assertEqual(result['close_pwm'],1606)
        self.assertEqual(result['mode'],'estimated_width_extrapolation')
        self.assertFalse(result['linear_extension']['measured'])
        self.assertFalse(result['physical_contact_verified'])
        self.assertEqual(original,self.policy)
        self.assertEqual(jaw_command(self.profile,self.policy,11.25)['mode'],'measured_width_interpolation')

    def test_below_8mm_and_unapproved_extension_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'requested_extension'):jaw_command(self.profile,self.policy,9.99)
        del self.policy['jaw_gap_extension']
        with self.assertRaisesRegex(ValueError,'measured_range'):jaw_command(self.profile,self.policy,10)


if __name__=='__main__':unittest.main()
