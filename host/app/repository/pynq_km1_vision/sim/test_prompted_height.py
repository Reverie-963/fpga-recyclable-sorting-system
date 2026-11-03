import threading
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from src.prompted_height import object_height,effective_heights
from src.monitor_prompted_operation import PromptedOperations
from src.sorting_prompted_motion import run
from src.board_prompted_console import BoardPreviewBackend


class HeightTests(unittest.TestCase):
    def test_invalid_input_rejected_and_selected_material_only_changes(self):
        settings={'heights_mm':dict(paper=65,plastic=53,metal=56)}
        for value in (True,False,'60',0,121,float('nan'),float('inf')):
            with self.subTest(value=value),self.assertRaises(ValueError):object_height(value)
        self.assertEqual(effective_heights(settings,60,['metal']),dict(paper=65,plastic=53,metal=60))
        self.assertEqual(settings['heights_mm']['metal'],56)
        self.assertIsNone(object_height(None))

    def test_height_changes_calibrated_ray_intersection(self):
        from sim.test_sorting_relocated import RelocatedGeometryTests
        from src.sorting_relocated_geometry import pixels_to_card
        fixture=RelocatedGeometryTests();fixture.setUp()
        try:
            args=(fixture.pose,fixture.camera,fixture.station,fixture.reg)
            low=pixels_to_card([[360,260]],40,*args)
            high=pixels_to_card([[360,260]],80,*args)
            self.assertTrue(np.isfinite(high).all())
            self.assertGreater(float(np.linalg.norm(high-low)),.1)
        finally:fixture.doCleanups()

    def test_new_height_can_enter_physical_preflight_without_old_height_rejection(self):
        settings={'heights_mm':dict(paper=65,plastic=53,metal=56)}
        config={'profiles':{m:{'object_height_mm':h} for m,h in settings['heights_mm'].items()},'journal_dir':'synthetic-ledger'}
        bundle=({},config,Path('.'),None,None,None,None,None)
        with patch('src.sorting_prompted_motion.calibrated_settings',return_value=(settings,{'eligible_materials':['metal'],'eligible_official_labels':['Drink can']})),\
             patch('src.sorting_prompted_motion.current_binding',return_value='synthetic'),\
             patch('src.sorting_prompted_motion.execution_scope',return_value={'materials':['metal']}),\
             patch('src.sorting_prompted_motion.load_unified',return_value=(bundle,{})),\
             patch('src.sorting_prompted_motion.integration_blockers',return_value=['synthetic_device_blocker']),\
             patch('src.sorting_prompted_motion.BatchMotion') as motion:
            with self.assertRaisesRegex(ValueError,'prompted_station_blocked:synthetic_device_blocker'):run('METAL',True,60)
            motion.assert_not_called()

    def operations(self):
        operations=object.__new__(PromptedOperations)
        operations.lock=threading.RLock();operations.token='page'
        operations.state='idle';operations.worker=None
        operations.backend=SimpleNamespace(object_height_mm=56,mechanical_heights_mm={'metal':56})
        return operations

    def test_preview_freezes_input_and_rejection_restores_previous_value(self):
        operations=self.operations()
        payload=dict(action='preview',token='page',object_height_mm=60)
        with patch('src.monitor_operation.Operations.command',side_effect=lambda data:operations.backend.object_height_mm):
            self.assertEqual(operations.command(payload),60)
        with patch('src.monitor_operation.Operations.command',side_effect=ValueError('invalid_request')):
            with self.assertRaises(ValueError):operations.command(dict(payload,object_height_mm=75))
        self.assertEqual(operations.backend.object_height_mm,60)

    def test_busy_or_expired_page_cannot_change_height(self):
        operations=self.operations()
        for state,token in [('executing','page'),('idle','expired')]:
            operations.state=state
            with self.assertRaises(ValueError):operations.command(dict(action='preview',token=token,object_height_mm=60))
            self.assertEqual(operations.backend.object_height_mm,56)

    def test_start_freezes_new_height_for_next_selected_attempt(self):
        operations=self.operations()
        with patch('src.monitor_operation.Operations.command',side_effect=lambda data:(data['material'],operations.backend.object_height_mm)):
            self.assertEqual(operations.command(dict(action='start',token='page',material='METAL',object_height_mm=60)),('METAL',60))

    def test_new_height_compiles_grasp_route_and_keeps_explicit_grip_depth(self):
        from sim.test_sorting_unified import UnifiedBatchTests
        from src.sorting_prompted_motion import PromptedMotionController,observation_timing
        from src.prompted_recyclables import read_policy,settings_for
        from src.sorting_io import Statistics
        fixture=UnifiedBatchTests();fixture.setUp()
        try:
            fixture.source.items=[i for i in fixture.source.items if i['label']=='metal']
            fixture.geometry.runtime=observation_timing(fixture.c)
            fixture.geometry.runtime['profiles']['metal']['object_height_mm']=60
            original_depth=fixture.c['profiles']['metal']['grasp_z_mm']
            from src.prompted_motion_policy import read_motion_options
            options=read_motion_options()
            fixture.policy.update(verification_timeout_s=options['frame_timeout_s'],verification_max_age_ms=options['frame_max_age_ms'])
            with fixture.journal:
                controller=PromptedMotionController(fixture.c,fixture.source,Statistics(),None,
                    fixture.journal,fixture.geometry,fixture.source.reference,fixture.root,
                    policy=fixture.policy,tracker=fixture.tracker,mode='preview',
                    prompt_settings=settings_for(read_policy(),{}),trial_labels=['Drink can'],
                    clock=fixture.clock.monotonic,sleep=fixture.clock.sleep)
                controller.policy['models']['metal']['object_height_mm']=60
                row=controller.acquire()
                row['observation']['instances'][0].update(official_label='Drink can',semantic_gate_passed=True)
                choice,plan=controller.candidate(row,(row['observation']['instances'][0],row['masks'][0]),'height-test')
                self.assertEqual(choice['object_height_mm'],60)
                self.assertEqual(controller.c['profiles']['metal']['grasp_z_mm'],original_depth)
                self.assertEqual(fixture.c['profiles']['metal']['object_height_mm'],56)
                self.assertGreater(len(plan),5)
        finally:fixture.doCleanups()

    def test_migration_adapter_forwards_height_for_board_preview_and_pc_execution(self):
        original=SimpleNamespace(object_height_mm=None)
        original.cycle=lambda material,execute:original.object_height_mm
        board=SimpleNamespace(plan=lambda payload:dict(status='preview_ready',received=payload))
        backend=BoardPreviewBackend(original,board);backend.object_height_mm=60
        self.assertEqual(backend.cycle('METAL',False)['received']['object_height_mm'],60)
        self.assertEqual(backend.cycle('METAL',True),60)


if __name__=='__main__':unittest.main()
