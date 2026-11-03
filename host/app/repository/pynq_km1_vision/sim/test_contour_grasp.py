"""Synthetic silhouettes and jaw measurements only; never opens hardware."""
import copy
import unittest
import cv2
import numpy as np
from src.contour_grasp import contour_section, jaw_command
from src.sorting_unified_plan import compile_route, validate_policy
from src.sorting_unified import UnifiedController
from src.sorting_io import Statistics
from sim.test_sorting_unified import test_config, setup_policy


def calibration(open_pwm=1000):
    return dict(kind='measured-jaw-gap-v1', measurement_id='synthetic-not-physical',
                samples=[dict(gap_mm=0, pwm=1900), dict(gap_mm=10, pwm=1700),
                         dict(gap_mm=50, pwm=open_pwm)],
                compression_mm=.5, opening_margin_mm=2)


class ContourTests(unittest.TestCase):
    def section(self, mask, point, angle=0, mapper=lambda p: np.asarray(p, float)):
        return contour_section(mask, mapper, point, angle)

    def test_horizontal_vertical_and_one_pixel_straw_use_pixel_edges(self):
        for thickness in (1, 4, 8):
            mask=np.zeros((80, 100), bool);mask[30:30+thickness,10:90]=1
            for m,p,angle in ((mask,[50,30],0),(mask.T,[30,50],90)):
                result=self.section(m,p,angle)
                self.assertAlmostEqual(result['width_mm'],thickness)

    def test_rotated_straw_width_is_not_axis_aligned_bounding_box(self):
        mask=np.zeros((120,120), np.uint8)
        cv2.fillPoly(mask,[cv2.boxPoints(((60,60),(80,6),35)).astype(np.int32)],1)
        result=self.section(mask,[60,60],35)
        self.assertAlmostEqual(result['width_mm'],6,delta=2)
        self.assertLess(result['width_mm'],10)

    def test_metric_projection_and_center_between_contacts(self):
        mask=np.zeros((80,100),bool);mask[30:36,10:90]=1
        result=self.section(mask,[50,31],mapper=lambda p:np.asarray(p)*[2,.5]+[100,-20])
        self.assertAlmostEqual(result['width_mm'],3)
        np.testing.assert_allclose(result['center_card_xy_mm'],[200,-3.75])

    def test_bottle_neck_uses_local_section_not_whole_body_width(self):
        mask=np.zeros((100,120),bool);mask[20:80,10:70]=1;mask[45:55,70:110]=1
        result=self.section(mask,[90,49])
        self.assertEqual(result['width_mm'],10)
        self.assertEqual(result['center_card_xy_mm'],[90,49.5])

    def test_hole_is_not_bridged_and_empty_center_is_rejected(self):
        mask=np.zeros((100,100),bool);mask[10:90,10:90]=1;mask[30:70,30:70]=0
        result=self.section(mask,[50,20])
        self.assertEqual(result['width_mm'],20)
        self.assertEqual(result['cross_section_intervals'],2)
        with self.assertRaisesRegex(ValueError,'not_inside_material'):
            self.section(mask,[50,50])

    def test_empty_mask_and_invalid_projection_rejected(self):
        mask=np.zeros((20,20),bool)
        with self.assertRaises(ValueError):self.section(mask,[10,10])
        mask[5:15,5:15]=1
        with self.assertRaises(ValueError):
            self.section(mask,[10,10],mapper=lambda p:np.asarray(p)*float('nan'))


class JawTests(unittest.TestCase):
    def setUp(self):
        self.c=test_config();self.policy=setup_policy(self.c)
        self.policy['jaw_open_span_mm']=50
        self.p=self.c['profiles']['plastic']
        self.p['jaw_calibration']=calibration(self.p['open_pwm'])

    def test_thin_straw_closes_further_than_thick_object(self):
        thin=jaw_command(self.p,self.policy,5.5)
        thick=jaw_command(self.p,self.policy,30.5)
        self.assertEqual(thin['target_gap_mm'],5)
        self.assertEqual(thin['close_pwm'],1800)
        self.assertGreater(thin['close_pwm'],thick['close_pwm'])
        self.assertTrue(thin['width_controls_closure'])

    def test_reverse_servo_direction_is_supported(self):
        self.p['open_pwm']=2000
        self.p['jaw_calibration']['samples']=[dict(gap_mm=0,pwm=1100),dict(gap_mm=50,pwm=2000)]
        self.assertLess(jaw_command(self.p,self.policy,5)['close_pwm'],
                        jaw_command(self.p,self.policy,30)['close_pwm'])

    def test_missing_calibration_is_explicit_and_preserves_old_command(self):
        del self.p['jaw_calibration']
        result=jaw_command(self.p,self.policy,5)
        self.assertEqual(result['close_pwm'],self.p['close_pwm'])
        self.assertFalse(result['width_controls_closure'])
        self.assertIn('missing',result['limitation'])

    def test_out_of_range_or_missing_width_never_extrapolated(self):
        for width in (None,0,-1,.1,47,100,float('nan'),True):
            with self.subTest(width=width),self.assertRaises(ValueError):
                jaw_command(self.p,self.policy,width)

    def test_bad_calibration_never_falls_back_to_fixed_pwm(self):
        for mutate in (
            lambda c:c.update(measurement_id=''),
            lambda c:c.update(compression_mm=-1),
            lambda c:c.update(opening_margin_mm=float('nan')),
            lambda c:c['samples'][1].update(pwm=2000),
            lambda c:c['samples'][1].update(gap_mm=0),
            lambda c:c['samples'][-1].update(pwm=1100),
            lambda c:c['samples'][-1].update(gap_mm=100),
        ):
            self.p['jaw_calibration']=calibration(self.p['open_pwm'])
            mutate(self.p['jaw_calibration'])
            with self.assertRaises(ValueError):validate_policy(self.policy,self.c)

    def test_complete_route_uses_measured_closure_and_preserves_other_targets(self):
        slot=self.policy['bin_slots']['plastic'][0]
        old=copy.deepcopy(self.c);del old['profiles']['plastic']['jaw_calibration']
        baseline=compile_route(old,'plastic',[0,5],90,self.policy,slot)
        route=compile_route(self.c,'plastic',[0,5],90,self.policy,slot,grasp_width_mm=5.5)
        for before,after in zip(baseline['steps'],route['steps']):
            self.assertEqual(before['name'],after['name'])
            if after['name']=='close':self.assertIn('#005P1800',after['command'])
            else:self.assertEqual(before,after)
        self.assertEqual(route['nominal_interpolation'][-1]['target_pwm'],self.c['known']['initial_pwm'])
        self.assertEqual(route['jaw_closure']['observed_width_mm'],5.5)

    def test_candidate_passes_local_width_to_real_route(self):
        controller=object.__new__(UnifiedController)
        controller.c=self.c;controller.policy=self.policy;controller.item_heights=None
        controller.allowed_labels={'plastic'};controller.stats=Statistics()
        controller.policy['live_object_admission']=dict(kind='operator-live-geometry-v1',
            operator_statement='synthetic test',physical_validation_completed=False)
        mask=np.zeros((120,160),bool);mask[50:58,20:70]=1
        obj=dict(label='plastic',confidence=.99)
        row=dict(image=np.zeros((120,160,3),np.uint8),masks=[mask],observation=dict(instances=[obj]))
        controller.map_object=lambda o,points:np.asarray(points,float)-[45,49]
        controller.source_objects=lambda r:[(obj,mask)]
        controller.free_slots=lambda *a,**k:self.policy['bin_slots']['plastic']
        choice,plan=controller.candidate_at_height(row,(obj,mask),1)
        self.assertAlmostEqual(choice['grasp_width_mm'],8)
        self.assertAlmostEqual(choice['pick_card_xy_mm'][1],4.5)
        self.assertEqual(choice['jaw_closure']['close_pwm'],1750)
        self.assertTrue(any(name=='close' and b'#005P1750' in packet for name,packet,_ in plan))
        # A measured jaw width must not alter visual location/obstacle geometry.
        from types import SimpleNamespace
        controller.item_heights=SimpleNamespace(measurement=lambda o:dict(grasp_width_mm=5.5,
            measurement_profile=dict(id='a'*32,revision=1,name='synthetic',pose='flat')))
        measured,measured_plan=controller.candidate_at_height(row,(obj,mask),1)
        self.assertEqual(measured['grasp_width_mm'],choice['grasp_width_mm'])
        self.assertEqual(measured['pick_card_xy_mm'],choice['pick_card_xy_mm'])
        self.assertEqual(measured['jaw_input_width_mm'],5.5)
        self.assertEqual(measured['jaw_width_source'],'operator_measured')
        self.assertEqual(measured['jaw_closure']['close_pwm'],1800)
        self.assertTrue(any(name=='close' and b'#005P1800' in packet for name,packet,_ in measured_plan))

    def test_round_target_uses_commanded_axis_for_neighbor_footprint(self):
        controller=object.__new__(UnifiedController)
        controller.c=self.c;controller.policy=self.policy;controller.item_heights=None
        controller.c['min_separation_px']=0
        controller.allowed_labels={'plastic'};controller.stats=Statistics()
        controller.policy['live_object_admission']=dict(kind='operator-live-geometry-v1',
            operator_statement='synthetic test',physical_validation_completed=False)
        mask=np.zeros((120,160),bool);mask[50:58,40:48]=1
        neighbor=np.zeros_like(mask);neighbor[53:55,60:62]=1
        obj=dict(label='plastic',confidence=.99);other=dict(label='plastic',confidence=.99)
        pairs=[(obj,mask),(other,neighbor)]
        row=dict(image=np.zeros((120,160,3),np.uint8),masks=[mask,neighbor],
                 observation=dict(instances=[obj,other]))
        controller.map_object=lambda o,points:np.asarray(points,float)-[44,49]
        controller.source_objects=lambda r:pairs
        controller.free_slots=lambda *a,**k:self.policy['bin_slots']['plastic']
        choice,_=controller.candidate_at_height(row,pairs[0],1)
        self.assertEqual(choice['axis_deg'],0)
        self.assertGreater(choice['nominal_neighbor_clearance_mm'],0)


if __name__=='__main__':unittest.main()
