"""Two distinct registered marker IDs remain subject to frozen displacement."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
from src.sorting_relocated_geometry import RelocatedGeometry
from src.sorting_relocated_run import motion_marker_limit,initial_marker_shift_limit,initial_window_age_limit


class MotionMarkerCountTests(unittest.TestCase):
    def test_initial_window_only_allows_explicit_fifty_percent_extension(self):
        c=dict(max_age_ms=2000)
        self.assertEqual(initial_window_age_limit({},c),2000)
        p=dict(initial_window_max_age_ms=3000)
        with self.assertRaisesRegex(ValueError,'operator_statement'):
            initial_window_age_limit(dict(motion_guard_policy=p),c)
        p['initial_window_operator_statement']='对应门槛放宽50%'
        self.assertEqual(initial_window_age_limit(dict(motion_guard_policy=p),c),3000)
        with self.assertRaisesRegex(ValueError,'invalid_initial_window'):
            initial_window_age_limit(dict(motion_guard_policy=dict(p,initial_window_max_age_ms=3001)),c)
        self.assertEqual(c['max_age_ms'],2000)

    def test_initial_shift_is_bounded_and_requires_explicit_operator_record(self):
        self.assertEqual(initial_marker_shift_limit({}),.5)
        policy=dict(initial_max_displacement_px=5)
        with self.assertRaisesRegex(ValueError,'operator_statement'):
            initial_marker_shift_limit(dict(motion_guard_policy=policy))
        policy['initial_localization_operator_statement']='同步放宽，快一点'
        self.assertEqual(initial_marker_shift_limit(dict(motion_guard_policy=policy)),5)
        for value in (0,5.1,float('nan'),True):
            with self.assertRaisesRegex(ValueError,'invalid_initial'):
                initial_marker_shift_limit(dict(motion_guard_policy=dict(policy,initial_max_displacement_px=value)))

    def fixture(self,minimum):
        g=RelocatedGeometry.__new__(RelocatedGeometry)
        g.camera={};g.signature='synthetic';g.motion_shift_px=5;g.min_fixed_markers=minimum
        detections=[dict(id=i,corners_px=[[10+i*20,10],[20+i*20,10],[20+i*20,20],[10+i*20,20]]) for i in [10,11]]
        g.expected={d['id']:np.array(d['corners_px'],float) for d in detections}
        g.pose=dict(marker_ids=[10,11],evidence=[dict(detections=detections)])
        g.reference=np.zeros((30,300,3),np.uint8);g.runtime=dict(layout={},image_size=[300,30])
        return g,copy.deepcopy(detections)

    def check(self,g,detections):
        with patch('src.station_geometry.detect_markers',return_value=detections),patch(
                'src.sorting_relocated_geometry.check_motion_layout',return_value={'checked':True}):
            return g.check(g.reference)

    def test_two_requires_explicit_binding_statement(self):
        self.assertEqual(motion_marker_limit({}),3)
        with self.assertRaisesRegex(ValueError,'operator_statement'):
            motion_marker_limit({'motion_guard_policy':{'min_fixed_markers':2}})
        self.assertEqual(motion_marker_limit({'motion_guard_policy':{
            'min_fixed_markers':2,'marker_count_operator_statement':'captain requests two'}}),2)

    def test_two_distinct_marks_pass_only_with_explicit_minimum(self):
        g,d=self.fixture(2);r=self.check(g,d)
        self.assertEqual(r['minimum_fixed_markers'],2);self.assertEqual(r['marker_ids'],[10,11])
        g.min_fixed_markers=3
        with self.assertRaisesRegex(ValueError,'three_fixed'):self.check(g,d)

    def test_one_duplicate_unknown_and_six_pixel_shift_are_rejected(self):
        g,d=self.fixture(2)
        with self.assertRaisesRegex(ValueError,'two_fixed'):self.check(g,d[:1])
        with self.assertRaisesRegex(ValueError,'duplicate_marker'):self.check(g,[d[0],d[0]])
        unknown=copy.deepcopy(d);unknown[1]['id']=999
        with self.assertRaisesRegex(ValueError,'two_fixed'):self.check(g,unknown)
        moved=copy.deepcopy(d);moved[0]['corners_px']=(np.array(moved[0]['corners_px'])+[6,0]).tolist()
        with self.assertRaisesRegex(ValueError,'board_moved'):self.check(g,moved)
