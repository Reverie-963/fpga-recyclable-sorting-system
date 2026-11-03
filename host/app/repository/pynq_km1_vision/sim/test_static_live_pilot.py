import copy
import unittest

from src.static_live_pilot import check_observations, retime_taught_plan


class LivePilotTests(unittest.TestCase):
    def setUp(self):
        self.policy = dict(stable_frames=3,model_version='frozen',camera_id='camera',image_size=[4,4],
                           max_age_ms=100,label='cardboard',min_confidence=.8,min_area_px=4,max_area_px=8,
                           source_roi_xyxy=[0,0,4,4],reference_center_px=[1.5,1.5],
                           max_reference_shift_px=1,max_stable_shift_px=.5)
        self.rows = [dict(model_version='frozen',camera_id='camera',image_size=[4,4],frame_id=i,
                          capture_started_ms=i,instances=[dict(label='cardboard',confidence=.9,
                          mask_rle=[5,2,2,2,5],bbox_xyxy=[1,1,3,3])]) for i in range(3)]

    def test_valid_gate_is_not_physical_success(self):
        result=check_observations(self.rows,self.policy,10)
        self.assertEqual(result['status'],'semantic_gate_passed')
        self.assertFalse(result['physical_success_verified'])

    def test_wrong_class_and_competing_instance_rejected(self):
        self.rows[1]['instances'][0]['label']='plastic'
        with self.assertRaisesRegex(ValueError,'wrong_or_uncertain'):check_observations(self.rows,self.policy,10)
        self.rows[1]=copy.deepcopy(self.rows[0]);self.rows[1]['frame_id']=1
        self.rows[1]['instances']*=2
        with self.assertRaisesRegex(ValueError,'ambiguous_scene'):check_observations(self.rows,self.policy,10)

    def test_stale_and_repeated_frames_rejected(self):
        with self.assertRaisesRegex(ValueError,'stale'):check_observations(self.rows,self.policy,1000)
        self.rows[-1]['frame_id']=1
        with self.assertRaisesRegex(ValueError,'non_increasing'):check_observations(self.rows,self.policy,10)

    def test_unknown_position_and_inconsistent_mask_rejected(self):
        self.policy['reference_center_px']=[3,3]
        with self.assertRaisesRegex(ValueError,'outside_taught_pick'):check_observations(self.rows,self.policy,10)
        self.policy['reference_center_px']=[1.5,1.5]
        self.rows[0]['instances'][0]['bbox_xyxy']=[0,0,4,4]
        with self.assertRaisesRegex(ValueError,'bbox_mask'):check_observations(self.rows,self.policy,10)

    def test_identity_confidence_and_roi_rejected(self):
        for key,value in [('model_version','other'),('camera_id','other'),('image_size',[2,8])]:
            rows=copy.deepcopy(self.rows);rows[0][key]=value
            with self.assertRaisesRegex(ValueError,'identity'):check_observations(rows,self.policy,10)
        for value in [True,float('nan'),.79,1.01]:
            rows=copy.deepcopy(self.rows);rows[0]['instances'][0]['confidence']=value
            with self.assertRaises(ValueError):check_observations(rows,self.policy,10)
        self.policy['source_roi_xyxy']=[0,0,2,2]
        with self.assertRaisesRegex(ValueError,'outside_taught_source'):check_observations(self.rows,self.policy,10)

    def test_timing_changes_only_time_and_keeps_supervised_minimum(self):
        plan=dict(stages=[dict(steps=[dict(command='{#000P0800T5000!#001P1600T5000!}',
                                          duration_ms=5000,timing_step='above_bin')])])
        original=copy.deepcopy(plan)
        timed,report=retime_taught_plan(plan,dict(move_ms=3000,settle_s=.2))
        self.assertEqual(plan,original)
        self.assertEqual(timed['stages'][0]['steps'][0]['command'],'{#000P0800T3000!#001P1600T3000!}')
        self.assertEqual(report['motion_and_settle_s'],3.2)
        with self.assertRaises(ValueError):retime_taught_plan(plan,dict(move_ms=1000))


if __name__=='__main__':unittest.main()
