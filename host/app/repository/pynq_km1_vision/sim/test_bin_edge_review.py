import unittest
from unittest.mock import patch
import numpy as np
from src.sorting_unified import UnifiedController
from src.sorting_prompted_preview import PromptedPreviewController
from src.sorting_unified_plan import validate_policy
from sim.test_sorting_unified import setup_policy,test_config

class BinEdgeReviewTests(unittest.TestCase):
    def controller(self):
        controller=UnifiedController.__new__(UnifiedController)
        controller.c=dict(image_size=[100,100],profiles={'metal':{'bin_roi':[40,20,90,80]}},
                          layout={},rgb_difference_threshold=90,max_scene_changed_fraction=.02)
        controller.policy=dict(relax_bin_edge_overlap=True,bin_edge_operator_statement='放宽200%',
            visual_review_policy=dict(kind='operator-vision-review-margin-v1',tolerance_factor=3,
                operator_statement='放宽200%',preserve_timings=True),bin_padding_px=5,
            empty_bin_max_changed_fraction=.6,
            bin_slots={'metal':[dict(slot_id='original',origin='legacy_qualified_profile',roi_fraction=[0,0,1,1])]})
        controller.empty_reference=np.zeros((100,100,3),np.uint8);controller.empty_by_label={}
        controller.slot_checks=[];controller.mode='preview'
        return controller

    def edge_mask(self):
        mask=np.zeros((100,100),bool);mask[36:64,32:45]=True;return mask

    def test_external_edge_is_reviewed_without_erasing_obstacle(self):
        controller=self.controller();mask=self.edge_mask();original=mask.copy()
        row=dict(image=controller.empty_reference.copy(),masks=[mask],observation=dict(instances=[{'label':'paper'}]))
        with patch('src.sorting_unified.adjust_exposure_reference',return_value=(controller.empty_reference,{})):
            self.assertTrue(controller.free_slots(row,'metal'))
        check=controller.slot_checks[-1]
        self.assertEqual(check['occupied_px'],0);self.assertGreater(check['raw_occupied_px'],0)
        self.assertEqual(check['boundary_reviews'][0]['inner_slot_overlap_px'],0)
        self.assertLessEqual(check['boundary_reviews'][0]['padded_slot_overlap_fraction'],.15+1e-12)
        np.testing.assert_array_equal(row['masks'][0],original)

    def test_inside_centroid_core_intrusion_and_large_edge_stay_blocked(self):
        controller=self.controller()
        masks=[]
        inside=np.zeros((100,100),bool);inside[35:50,42:47]=True;masks.append(inside)
        core=np.zeros((100,100),bool);core[35:42,10:55]=True;masks.append(core)
        broad=np.zeros((100,100),bool);broad[25:75,20:48]=True;masks.append(broad)
        for mask in masks:
            with self.subTest(pixels=int(mask.sum())):
                self.assertIsNone(controller.bin_edge_review(mask,[40,20,90,80]))
                row=dict(image=controller.empty_reference,masks=[mask],observation=dict(instances=[{'label':'paper'}]))
                with patch('src.sorting_unified.adjust_exposure_reference',return_value=(controller.empty_reference,{})):
                    self.assertEqual(controller.free_slots(row,'metal'),[])
                self.assertGreater(controller.slot_checks[-1]['occupied_px'],0)

    def test_original_policy_stays_strict_and_opt_in_requires_record(self):
        controller=self.controller();controller.policy['relax_bin_edge_overlap']=False
        self.assertIsNone(controller.bin_edge_review(self.edge_mask(),[40,20,90,80]))
        c=test_config();p=setup_policy(c);p['relax_bin_edge_overlap']=True
        with self.assertRaisesRegex(ValueError,'explicit_margin_record'):validate_policy(p,c)

    def prompted(self):
        controller=PromptedPreviewController.__new__(PromptedPreviewController)
        controller.c=dict(image_size=[100,100],pickup_source_roi=[0,0,40,100])
        controller.policy=dict(relax_bin_edge_overlap=False,bin_padding_px=5)
        return controller

    def test_prompted_source_edge_uses_normal_bound_and_keeps_raw_mask(self):
        controller=self.prompted();mask=np.zeros((100,100),bool);mask[42:50,32:42]=True;raw=mask.copy()
        review=controller.target_bin_edge_review(mask,[40,20,90,80])
        self.assertIsNotNone(review);self.assertEqual(review['max_slot_overlap_fraction'],.05)
        self.assertEqual(review['inner_slot_overlap_px'],0);self.assertTrue(review['centroid_inside_pickup'])
        self.assertFalse(review['expanded_legacy_tolerance_used']);np.testing.assert_array_equal(mask,raw)
        legacy=UnifiedController.__new__(UnifiedController);legacy.c=controller.c;legacy.policy=controller.policy
        self.assertIsNone(legacy.target_bin_edge_review(mask,[40,20,90,80]))

    def test_prompted_core_inside_bin_broad_overlap_and_non_source_stay_rejected(self):
        controller=self.prompted();rect=[40,20,90,80]
        for bounds in ((32,42,45,50),(41,42,46,50),(30,25,42,75)):
            a,b,d,e=bounds;mask=np.zeros((100,100),bool);mask[b:e,a:d]=True
            with self.subTest(bounds=bounds):self.assertIsNone(controller.target_bin_edge_review(mask,rect))
        mask=np.zeros((100,100),bool);mask[42:50,32:42]=True;controller.c['pickup_source_roi']=[0,0,20,100]
        self.assertIsNone(controller.target_bin_edge_review(mask,rect))
