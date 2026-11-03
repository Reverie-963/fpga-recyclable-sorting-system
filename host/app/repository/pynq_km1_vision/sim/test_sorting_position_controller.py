import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from sim.test_sorting_known import config
from src.sorting_demo import DemoClock,DemoSource,DemoMotion
from src.sorting_io import Statistics
from src.sorting_loop import Journal,mapped
from src.sorting_scene import check_motion_layout
from src.sorting_position_controller import (PositionController,PositionWindowedCamera,
    validate_position_policy,apply_position_geometry)


def policy():
    return dict(kind='operator-supervised-position-policy-v1',operator_authorized=True,
        operator_statement='复用现有坐标系，接入统一位置规划并执行受监督试验',
        label='paper',object_height_mm=65,axis_tolerance_deg=20,
        workspace_physically_qualified=False,unattended_accepted=False,
        pick_center_card_polygon_mm=[[-10,0],[10,0],[10,10],[-10,10]],
        source_card_polygon_mm=[[-15,-5],[15,-5],[15,20],[-15,20]],
        model=dict(kind='carton-position-model-preview-v1',base_center_pwm=1500,
            base_pwm_per_degree=7.66,radial_bias_mm=10,wrist_pwm_per_degree=600/90))


class PositionControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.c=config();self.c['profiles']['paper']['object_height_mm']=65
        self.c['statistics']['regions']=dict(source=self.c['source_roi'],
            static_a=self.c['layout']['static_rois'][0],static_b=self.c['layout']['static_rois'][1],
            **{p['bin_id']:p['bin_roi'] for p in self.c['profiles'].values()})
        self.clock=DemoClock();self.source=DemoSource(self.c,self.clock)
        self.source.items=[i for i in self.source.items if i['label']=='paper']
        get=self.source.next
        def next_row(timeout):
            row=get(timeout);row['observation']['source_frame_id']=str(row['observation']['frame_id']);return row
        self.source.next=next_row
        layout=dict(self.c['layout'],image_size=self.c['image_size'])
        self.geometry=SimpleNamespace(runtime=copy.deepcopy(self.c),reference=self.source.reference,signature='f'*64,
            map=lambda label,pts:mapped(self.c['profiles'][label]['image_to_card_mm'],pts),
            check=lambda image:check_motion_layout(self.source.reference,image,layout))
        self.motion=DemoMotion(self.source,self.clock)
    def controller(self,p=None):
        journal=Journal(Path(self.temp.name)/'ledger')
        controller=PositionController(self.c,self.source,Statistics(),self.motion,journal,self.geometry,
            self.source.reference,self.temp.name,mode='simulation',position_policy=p or policy(),
            clock=self.clock.monotonic,sleep=self.clock.sleep)
        return controller,journal
    def test_one_cycle_compiles_and_executes_complete_route_and_verifies(self):
        controller,journal=self.controller()
        with journal:r=controller.cycle()
        self.assertEqual(r['status'],'simulated_visual_pass',r)
        expected=13+len(self.c['profiles']['paper']['transfer_pwm'])+len(self.c['profiles']['paper']['retreat_pwm'])
        self.assertEqual(len(r['plan']),expected);self.assertEqual(len(self.motion.sent),expected)
        self.assertTrue(r['choice']['complete_route_compiled_before_motion'])
        self.assertFalse(r['workspace_physically_qualified']);self.assertIsNone(journal.pending())
    def test_source_outside_policy_rejected_before_transport(self):
        p=policy();p['pick_center_card_polygon_mm']=[[8,0],[10,0],[10,2],[8,2]]
        controller,journal=self.controller(p)
        with journal:r=controller.cycle()
        self.assertEqual(r['status'],'rejected');self.assertIn('outside_operator',r['reason'])
        self.assertEqual(self.motion.sent,[])
    def test_low_speed_release_wait_and_home_recheck_preserved(self):
        controller,journal=self.controller()
        with journal:r=controller.cycle()
        self.assertEqual(controller.movement_wait('release',2000),3)
        self.assertEqual(controller.movement_wait('source_high',5000),5.5)
        self.assertEqual(r['plan'][0]['step'],'home')
        self.assertTrue(any('guard_home' in e['image'] for e in r['evidence']))
    def test_unknown_category_or_occupied_destination_rejected(self):
        controller,journal=self.controller();self.source.items[0]['label']='metal'
        with journal:r=controller.cycle()
        self.assertEqual(r['status'],'rejected');self.assertEqual(self.motion.sent,[])
    def test_other_material_keeps_original_checked_route(self):
        self.source.items=[i for i in DemoSource(self.c,self.clock).items if i['label']=='metal']
        controller,journal=self.controller()
        with journal:r=controller.cycle()
        self.assertEqual(r['status'],'simulated_visual_pass',r)
        self.assertEqual(r['choice']['route_mode'],'legacy_checked_points')
        self.assertFalse(controller.position_route_active)
    def test_explicit_policy_cannot_claim_workspace_or_unattended_acceptance(self):
        for change in (dict(operator_authorized=False),dict(workspace_physically_qualified=True),
                       dict(unattended_accepted=True),dict(axis_tolerance_deg=31),dict(object_height_mm=56)):
            with self.assertRaises(ValueError):validate_position_policy(dict(policy(),**change),self.c)
        self.assertEqual(validate_position_policy(dict(policy(),axis_tolerance_deg=30),self.c)['axis_tolerance_deg'],30)
    def test_source_geometry_projects_existing_registration_and_binds_identity(self):
        g=SimpleNamespace(original=self.c,runtime=copy.deepcopy(self.c),
            registration=dict(world_to_card_xy=np.eye(3).tolist()),pose={},camera={},station={},signature='f'*64)
        with patch('src.sorting_position_controller.world_to_pixels',lambda pts,*args:np.asarray(pts)[:,:2]+[40,50]):
            apply_position_geometry(g,policy())
        self.assertEqual(g.runtime['source_roi'],[25,45,56,71])
        self.assertNotEqual(g.signature,'f'*64);self.assertEqual(self.c['source_roi'],[5,30,80,110])
    def test_trial_source_cannot_overlap_bins(self):
        g=SimpleNamespace(original=self.c,runtime=copy.deepcopy(self.c),
            registration=dict(world_to_card_xy=np.eye(3).tolist()),pose={},camera={},station={},signature='f'*64)
        with patch('src.sorting_position_controller.world_to_pixels',lambda pts,*args:np.asarray(pts)[:,:2]+[110,50]):
            with self.assertRaisesRegex(ValueError,'regions_overlap'):apply_position_geometry(g,policy())
    def test_camera_retreat_barrier_survives_without_forced_window_phase(self):
        class Source:
            def after(self,stamp):self.barrier=stamp;self.post_retreat_count=0
        raw=Source();source=PositionWindowedCamera(raw);source.queue.append('old')
        source.after(123)
        self.assertEqual(raw.barrier,123);self.assertIsNone(raw.post_retreat_count);self.assertFalse(source.queue)

if __name__=='__main__':unittest.main()
