import copy
import unittest
from sim.test_sorting_known import config
from src.sorting_position_plan import compile_position_route

class PositionPlanTests(unittest.TestCase):
    def setUp(self):
        self.c=config();self.c['profiles']['paper']['object_height_mm']=65
        self.c['profiles']['paper']['close_pwm']=1500
        self.model=dict(kind='carton-position-model-preview-v1',base_center_pwm=1500,
            base_pwm_per_degree=7.66,radial_bias_mm=10,wrist_pwm_per_degree=600/90)
    def plan(self,point,axis=90):return compile_position_route(self.c,'paper',point,axis,self.model)
    def test_both_sides_change_base_wrist_and_source_commands(self):
        left,right=self.plan([-70,22]),self.plan([51,38])
        self.assertGreater(left['base_pwm'],1500);self.assertLess(right['base_pwm'],1500)
        self.assertGreater(left['wrist_pwm'],1500);self.assertLess(right['wrist_pwm'],1500)
        self.assertNotEqual(left['steps'][5]['command'],right['steps'][5]['command'])
    def test_route_is_compiled_to_neutral_before_any_execution_is_possible(self):
        result=self.plan([0,8]);names=[s['name'] for s in result['steps']]
        self.assertLess(names.index('source_grip'),names.index('close'))
        self.assertLess(names.index('close'),names.index('source_lift_high'))
        self.assertLess(names.index('restore_wrist_high'),names.index('transfer_0'))
        self.assertEqual(result['nominal_interpolation'][-1]['target_pwm'],self.c['known']['initial_pwm'])
        self.assertFalse(result['execution_permitted']);self.assertFalse(result['robot_mapping_is_measured'])
    def test_axis_180_degree_symmetry(self):
        self.assertEqual(self.plan([10,8],90)['wrist_pwm'],self.plan([10,8],270)['wrist_pwm'])
    def test_original_config_and_measured_transfer_route_are_preserved(self):
        original=copy.deepcopy(self.c);result=self.plan([0,8]);self.assertEqual(self.c,original)
        transfers=[s for s in result['steps'] if s['name'].startswith('transfer_')]
        self.assertEqual(len(transfers),len(self.c['profiles']['paper']['transfer_pwm']))
    def test_unchecked_point_is_explicit_and_does_not_extend_domain(self):
        result=self.plan([51,38]);self.assertFalse(result['inside_legacy_checked_point_domain'])
        self.assertIn('workspace_measurements',result['calibration_status'])
    def test_invalid_and_unreachable_targets_do_not_return_partial_plans(self):
        for point in ([200,30],[0,500],[float('nan'),0]):
            with self.assertRaises(ValueError):self.plan(point)
        with self.assertRaises(ValueError):compile_position_route(self.c,'metal',[0,8],90,self.model)
    def test_unmeasured_model_values_cannot_escape_protocol_checks(self):
        for key,value in [('base_pwm_per_degree',0),('radial_bias_mm',100),('wrist_pwm_per_degree',float('nan'))]:
            model=dict(self.model,**{key:value})
            with self.assertRaises(ValueError):compile_position_route(self.c,'paper',[0,8],90,model)

if __name__=='__main__':unittest.main()
