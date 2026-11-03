import copy
import hashlib
import io
import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from src.sorting_prompted_motion import PromptedHTTP,PromptedMotionController,observation_timing
from src.prompted_recyclables import read_policy,settings_for


class MotionTests(unittest.TestCase):
    def setUp(self):
        self.policy=read_policy();self.settings=settings_for(self.policy,{})

    def test_temporal_override_is_twelve_seconds_without_mutating_old_profile(self):
        original=dict(max_age_ms=2000,frame_timeout_s=2,min_confidence=.8,stable_frames=3)
        updated=observation_timing(original)
        self.assertEqual(updated['max_age_ms'],12000)
        self.assertEqual(updated['frame_timeout_s'],12)
        self.assertEqual(original['max_age_ms'],2000)
        self.assertEqual(updated['min_confidence'],.8)
        self.assertEqual(updated['stable_frames'],1)
        self.assertEqual(original['stable_frames'],3)

    def test_twelve_second_runtime_can_compile_route_with_original_mechanical_profile(self):
        from sim.test_sorting_unified import UnifiedBatchTests
        from src.sorting_io import Statistics
        fixture=UnifiedBatchTests();fixture.setUp()
        try:
            fixture.source.items=[i for i in fixture.source.items if i['label']=='metal']
            fixture.geometry.runtime=observation_timing(fixture.c)
            fixture.policy.update(verification_timeout_s=12,verification_max_age_ms=12000)
            with fixture.journal:
                controller=PromptedMotionController(fixture.c,fixture.source,Statistics(),None,
                    fixture.journal,fixture.geometry,fixture.source.reference,fixture.root,
                    policy=fixture.policy,tracker=fixture.tracker,mode='preview',
                    prompt_settings=self.settings,trial_labels=['Drink can'],
                    clock=fixture.clock.monotonic,sleep=fixture.clock.sleep)
                row=controller.acquire()
                row['observation']['instances'][0].update(official_label='Drink can',semantic_gate_passed=True)
                choice,plan=controller.candidate(row,(row['observation']['instances'][0],row['masks'][0]),'test')
                self.assertEqual(controller.c['max_age_ms'],12000)
                self.assertLessEqual(fixture.c['max_age_ms'],2000)
                self.assertGreater(len(plan),5)
        finally:fixture.doCleanups()

    def test_http_control_refuses_changed_settings_pixels_or_weights(self):
        image=np.zeros((20,30,3),np.uint8)
        result=dict(settings_sha256=self.settings['settings_sha256'],model_sha256=self.policy['model']['sha256'],
                    source_bgr_sha256=hashlib.sha256(image.tobytes()).hexdigest(),image_size=[30,20],instances=[])
        detector=PromptedHTTP(self.settings)
        for key in ('settings_sha256','model_sha256','source_bgr_sha256'):
            changed=dict(result,**{key:'changed'})
            with patch.object(detector.opener,'open',return_value=io.BytesIO(json.dumps(changed).encode())):
                with self.assertRaises(ValueError):detector.predict(image)
        with patch.object(detector.opener,'open',return_value=io.BytesIO(json.dumps(result).encode())):
            accepted=detector.predict(image)
        self.assertEqual(accepted['runtime_id'],self.settings['settings_sha256'])
        self.assertEqual(accepted['prompt_mode'],'text')

    def test_unvalidated_original_label_cannot_enter_physical_route(self):
        controller=object.__new__(PromptedMotionController)
        controller.mode='known_physical';controller.trial_labels=['Drink can']
        with self.assertRaisesRegex(ValueError,'trial_scope'):
            controller.candidate({},({'official_label':'Aluminium foil'},None),'1')

    def test_bin_confidence_and_material_required_without_exact_raw_class(self):
        controller=object.__new__(PromptedMotionController)
        controller.placement_verification={'mode':'semantic_rgb'}
        controller.c={'min_scene_iou':.85}
        controller.outside_objects=lambda r:[] if r is None else [(o,np.ones((2,2),bool)) for o in r['observation']['instances']]
        target=({'label':'metal','official_label':'Drink can'},None)
        for material,passed in [('plastic',True),('metal',False)]:
            row=dict(observation=dict(instances=[dict(label=material,zone='bin_metal',official_label='Food Can',semantic_gate_passed=passed)]))
            with self.assertRaisesRegex(ValueError,'landing_label'):
                controller.verify_selected(None,target,None,[row])
        row=dict(observation=dict(instances=[dict(label='metal',zone='bin_metal',official_label='Food Can',semantic_gate_passed=True)]))
        with patch('src.sorting_prompted_preview.PromptedPreviewController.verify_selected',return_value={'status':'delegated'}):
            self.assertEqual(controller.verify_selected(None,target,None,[row])['status'],'delegated')

    def test_prewrite_raw_class_change_is_accepted_but_material_change_is_not(self):
        controller=object.__new__(PromptedMotionController)
        controller.c={'min_scene_iou':.85}
        mask=np.ones((3,4),bool)
        before=dict(masks=[mask],observation=dict(instances=[dict(label='metal',official_label='Drink can',zone='source',semantic_gate_passed=True)]))
        after=copy.deepcopy(before);after['observation']['instances'][0]['official_label']='Food Can'
        with patch('src.sorting_prompted_preview.PromptedPreviewController.unchanged'):
            controller.unchanged(before,after)
            after['observation']['instances'][0]['label']='plastic'
            with self.assertRaisesRegex(ValueError,'material_unstable'):controller.unchanged(before,after)

    def test_existing_other_label_in_bin_does_not_reinstate_empty_requirement(self):
        controller=object.__new__(PromptedMotionController);controller.c={'min_scene_iou':.85}
        controller.placement_verification={'mode':'semantic_rgb'}
        old_mask=np.zeros((6,6),bool);old_mask[:2,:2]=True
        new_mask=np.zeros((6,6),bool);new_mask[3:5,3:5]=True
        old=dict(label='metal',official_label='Food Can',zone='bin_metal',semantic_gate_passed=False)
        placed=dict(label='metal',official_label='Drink can',zone='bin_metal',semantic_gate_passed=True)
        before=[(old,old_mask)];after=before+[(placed,new_mask)]
        controller.outside_objects=lambda row:row
        with patch('src.sorting_prompted_preview.PromptedPreviewController.verify_selected',return_value={'status':'delegated'}) as original:
            result=controller.verify_selected(before,({'label':'metal','official_label':'Drink can'},new_mask),{},[after])
            self.assertEqual(result['status'],'delegated')
            original.assert_called_once()

    def test_weak_background_instability_does_not_block_stable_target_or_drop_obstacles(self):
        controller=object.__new__(PromptedMotionController)
        controller.motion_options={'full_scene_instance_stability_check_enabled':False}
        controller.trial_labels=['Drink can']
        controller.c=dict(min_scene_iou=.85,image_size=[10,10],source_roi=[0,0,10,10],profiles={},max_scene_changed_fraction=.05)
        controller.changed_pixels=lambda a,b:np.zeros((10,10),bool)
        mask=np.zeros((10,10),bool);mask[2:6,2:6]=True
        weak=~mask
        target=dict(label='metal',official_label='Drink can',zone='source',semantic_gate_passed=True)
        noise=dict(label='paper',official_label='Corrugated carton',zone='source',semantic_gate_passed=False)
        before=dict(masks=[mask,weak],observation=dict(instances=[target,noise]))
        after=dict(masks=[mask],observation=dict(instances=[target]))
        controller.unchanged(before,after)
        self.assertEqual(len(after['masks']),1)
        self.assertEqual(after['target_stability']['raw_before_count'],2)
        self.assertTrue(after['target_stability']['all_latest_masks_retained_for_obstacle_checks'])
        # New weak obstacles remain present too; this does not filter source_objects.
        later=dict(masks=[mask,weak],observation=dict(instances=[target,noise]))
        controller.unchanged(after,later)
        self.assertEqual(len(later['masks']),2)
        lost=dict(masks=[weak],observation=dict(instances=[noise]))
        with self.assertRaises(ValueError):controller.unchanged(after,lost)

    def test_single_item_backend_reports_shared_unresolved_batch_before_start(self):
        from src.monitor_prompted_operation import PromptedBackend
        backend=object.__new__(PromptedBackend)
        with tempfile.TemporaryDirectory() as folder:
            backend.journal_dir=Path(folder)
            pending=backend.journal_dir/'batch-pending.json';pending.write_text('{"status":"needs_review"}')
            with patch('src.monitor_operation.StationBackend.check',return_value=[]),patch(
                    'src.monitor_prompted_operation.calibrated_settings',return_value=({}, {'eligible_materials':['metal']})):
                self.assertIn('unified_batch_needs_review',backend.check())
                self.assertTrue(pending.exists())


if __name__=='__main__':unittest.main()
