import copy
import unittest
from src.sorting_review_policy import review_factor
from src.sorting_relocated_run import motion_shift_limit,initial_marker_shift_limit
from src.sorting_unified_plan import validate_policy
from sim.test_sorting_unified import setup_policy,test_config

def allowance():
    return dict(kind='operator-vision-review-margin-v1',operator_statement='放宽200%',
                tolerance_factor=3,preserve_timings=True)

class ReviewPolicyTests(unittest.TestCase):
    def test_margin_requires_explicit_record_and_is_bounded(self):
        self.assertEqual(review_factor({}),1)
        self.assertEqual(review_factor(dict(visual_review_policy=allowance())),3)
        for change in ({'operator_statement':''},{'preserve_timings':False},
                       {'tolerance_factor':3.1},{'tolerance_factor':True}):
            with self.assertRaises(ValueError):review_factor(dict(visual_review_policy=dict(allowance(),**change)))

    def test_pixel_margins_above_five_require_the_matching_review_record(self):
        p=dict(kind='operator-approved-station-motion-margin-v1',operator_statement='放宽200%',
               production_point_domains_changed=False,max_displacement_px=15,
               initial_max_displacement_px=15,initial_localization_operator_statement='放宽200%')
        with self.assertRaises(ValueError):motion_shift_limit(dict(motion_guard_policy=p))
        b=dict(motion_guard_policy=p,visual_review_policy=allowance())
        self.assertEqual(motion_shift_limit(b),15);self.assertEqual(initial_marker_shift_limit(b),15)
        p['max_displacement_px']=15.1
        with self.assertRaises(ValueError):motion_shift_limit(b)

    def test_decision_limits_lower_only_with_explicit_factor_and_keep_structure(self):
        c=test_config();p=setup_policy(c)
        p.update(decision_thresholds={'min_confidence':.4/3,'min_scene_iou':.9/3,
            'min_erased_fraction':.8/3,'min_persistent_fraction':.6/3,
            'max_scene_changed_fraction':.15,'max_other_bin_changed_fraction':.15},
            threshold_operator_statement='放宽200%',min_geometry_component_fraction=.95/3,
            empty_bin_max_changed_fraction=.6)
        with self.assertRaises(ValueError):validate_policy(p,c)
        p['visual_review_policy']=allowance();validate_policy(p,c)
        bad=copy.deepcopy(p);bad['workspace_physically_qualified']=True
        with self.assertRaises(ValueError):validate_policy(bad,c)
        bad=copy.deepcopy(p);bad['decision_thresholds']['min_confidence']=0
        with self.assertRaises(ValueError):validate_policy(bad,c)
        self.assertEqual(c,test_config())

    def test_known_visual_floor_changes_only_with_record_and_mechanics_stay_required(self):
        from sim.test_sorting_known import config
        from src.sorting_known import validate_known_config
        c=config();c['layout']['min_patch_correlation']=.98/3
        c['layout']['max_layout_shift_px']=15;c['layout']['min_motion_patch_dynamic_range']=80/3
        with self.assertRaisesRegex(ValueError,'layout_limits_relaxed'):validate_known_config(c)
        c['visual_review_policy']=allowance();validate_known_config(c)
        c['arm']['move_ms']=400
        with self.assertRaisesRegex(ValueError,'low_speed_required'):validate_known_config(c)
