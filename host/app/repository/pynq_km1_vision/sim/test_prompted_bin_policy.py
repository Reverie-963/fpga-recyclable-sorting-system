import unittest
from unittest.mock import patch
from src.sorting_prompted_preview import PromptedPreviewController


class DestinationPolicyTests(unittest.TestCase):
    def controller(self,enabled=False,mode='known_physical'):
        controller=object.__new__(PromptedPreviewController)
        controller.motion_options={'destination_empty_check_enabled':enabled}
        controller.mode=mode;controller.slot_checks=[]
        controller.c={'profiles':{m:{'bin_roi':[0,0,40,40]} for m in ('paper','plastic','metal')}}
        controller.policy={'bin_slots':{m:[dict(slot_id=m+'-measured',origin='legacy_qualified_profile',roi_fraction=[0,0,1,1]),
            dict(slot_id=m+'-new',origin='candidate',roi_fraction=[0,0,.5,.5])] for m in controller.c['profiles']}}
        return controller

    def test_all_materials_skip_semantic_and_rgb_occupancy_without_empty_reference(self):
        controller=self.controller()
        for material in ('paper','plastic','metal'):
            # No image/masks/reference supplied: disabled policy cannot inspect
            # any of them or report an occupied bin as verified empty.
            slots=controller.free_slots({},material)
            self.assertEqual([s['slot_id'] for s in slots],[material+'-measured'])
        self.assertEqual(len(controller.slot_checks),3)
        for row in controller.slot_checks:
            self.assertFalse(row['semantic_occupancy_checked'])
            self.assertFalse(row['rgb_empty_reference_checked'])
            self.assertEqual(row['reason'],'operator_disabled_destination_empty_check')

    def test_reenable_delegates_to_original_checks(self):
        controller=self.controller(enabled=True)
        with patch('src.sorting_unified.UnifiedController.free_slots',return_value=[]) as original:
            self.assertEqual(controller.free_slots({'occupied':True},'metal'),[])
            original.assert_called_once_with({'occupied':True},'metal',None)

    def test_preview_can_list_candidate_slots_without_promoting_them_for_motion(self):
        controller=self.controller(mode='preview')
        self.assertEqual(len(controller.free_slots({},'paper')),2)
        controller.mode='known_physical'
        self.assertEqual(len(controller.free_slots({},'paper')),1)

    def test_disabling_perimeter_gap_preserves_other_candidate_errors_and_restores_config(self):
        controller=self.controller()
        controller.c['min_separation_px']=5
        controller.motion_options['minimum_silhouette_separation_check_enabled']=False
        target=({'semantic_gate_passed':True,'zone':'source'},None)
        def collision(*args):
            self.assertEqual(controller.c['min_separation_px'],0)
            raise ValueError('candidate_route_rejected')
        with patch('src.sorting_unified.UnifiedController.candidate',side_effect=collision):
            with self.assertRaisesRegex(ValueError,'candidate_route_rejected'):
                controller.candidate({},target,'1')
        self.assertEqual(controller.c['min_separation_px'],5)


if __name__=='__main__':unittest.main()
