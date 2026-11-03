"""Synthetic measurements and routes; these tests never open a device."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from src.sorting_item_heights import height_plan, ItemHeights, rank_candidates
from src.yoloe_inference import encode_mask
from src.monitor_prompted_operation import PromptedOperations
from src.board_prompted_console import BoardPreviewBackend


def plan_for(pairs, size, heights=None):
    return dict(kind='instance-heights-v1',model_sha256='a'*64,configuration_sha256='b'*64,
        source_stream_id='stream',image_size=size,items=[dict(item_id=n+1,label=obj['label'],
        official_label=obj['official_label'],mask_rle=encode_mask(mask),
        object_height_mm=(heights or [40,80])[n]) for n,(obj,mask) in enumerate(pairs)])


class ItemHeightTests(unittest.TestCase):
    def setUp(self):
        a=np.zeros((30,40),bool);b=a.copy();a[2:8,2:8]=1;b[17:24,26:32]=1
        self.pairs=[(dict(label='metal',official_label='Drink can',confidence=.9),a),
                    (dict(label='metal',official_label='Drink can',confidence=.8),b)]
        self.plan=plan_for(self.pairs,[40,30])

    def resolver(self,plan=None):
        return ItemHeights(plan or self.plan,model='a'*64,configuration='b'*64,image_size=[40,30])

    def test_same_material_reverse_order_keeps_each_measurement(self):
        resolver=self.resolver();fresh=[(dict(o),m.copy()) for o,m in reversed(self.pairs)]
        resolver.bind(fresh,'stream',.85)
        self.assertEqual([resolver.measurement(o)['object_height_mm'] for o,_ in fresh],[80,40])
        self.plan['items'][0]['object_height_mm']=99
        self.assertEqual(resolver.measurement(fresh[1][0])['object_height_mm'],40)

    def test_optional_width_bound_to_own_mask_and_invalid_values_rejected(self):
        self.plan['items'][0]['grasp_width_mm']=10
        self.plan['items'][0]['measurement_profile']=dict(id='c'*32,revision=2,name='红色吸管',pose='平放')
        resolver=self.resolver();resolver.bind(list(reversed(self.pairs)),'stream',.85)
        self.assertEqual(resolver.measurement(self.pairs[0][0])['grasp_width_mm'],10)
        self.assertNotIn('grasp_width_mm',resolver.measurement(self.pairs[1][0]))
        for width in (True,None,0,151,float('nan'),'10'):
            p=copy.deepcopy(self.plan);p['items'][0]['grasp_width_mm']=width
            with self.subTest(width=width),self.assertRaisesRegex(ValueError,'measured_width'):height_plan(p)
        p=copy.deepcopy(self.plan);p['items'][0]['measurement_profile']['revision']=True
        with self.assertRaisesRegex(ValueError,'measurement_profile'):height_plan(p)

    def test_raw_class_change_keeps_height_but_material_change_does_not(self):
        resolver=self.resolver();fresh=[(dict(o,official_label='Food Can'),m.copy()) for o,m in self.pairs]
        resolver.bind(fresh,'stream',.85)
        self.assertEqual([resolver.measurement(o)['object_height_mm'] for o,_ in fresh],[40,80])
        self.assertEqual(resolver.measurement(fresh[0][0])['official_label'],'Drink can')
        fresh[0][0]['label']='plastic'
        with self.assertRaisesRegex(ValueError,'missing_or_moved'):resolver.bind(fresh,'stream',.85)

    def test_missing_new_moved_or_ambiguous_objects_never_borrow_height(self):
        moved=self.pairs[0][1].copy();moved[:]=0;moved[8:14,15:21]=1
        for pairs in ([self.pairs[0],(dict(self.pairs[1][0]),moved)],
                      [self.pairs[0],self.pairs[0]]):
            with self.assertRaises(ValueError):self.resolver().bind(pairs,'stream',.85)
        duplicate=copy.deepcopy(self.plan);duplicate['items'][1]['mask_rle']=duplicate['items'][0]['mask_rle']
        with self.assertRaisesRegex(ValueError,'ambiguous'):self.resolver(duplicate).bind(self.pairs,'stream',.85)

    def test_all_source_obstacles_need_their_own_height(self):
        p=copy.deepcopy(self.plan);p['items']=p['items'][:1]
        with self.assertRaisesRegex(ValueError,'missing_or_moved'):self.resolver(p).bind(self.pairs,'stream',.85)

    def test_fresh_scene_check_keeps_frozen_candidate_bindings(self):
        resolver=self.resolver();resolver.bind(self.pairs,'stream',.85)
        fresh=[(dict(o),m.copy()) for o,m in self.pairs]
        resolver.check_scene(fresh,'stream',.85)
        self.assertEqual(resolver.measurement(self.pairs[0][0])['object_height_mm'],40)

    def test_identity_image_and_stream_changes_rejected(self):
        for key,value in [('model_sha256','c'*64),('configuration_sha256','c'*64),('image_size',[30,40])]:
            p=copy.deepcopy(self.plan);p[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.resolver(p)
        with self.assertRaisesRegex(ValueError,'stream_changed'):self.resolver().bind(self.pairs,'new',.85)

    def test_invalid_height_and_rle_and_duplicate_ids_rejected(self):
        for value in (None,True,'40',0,121,float('nan'),float('inf')):
            p=copy.deepcopy(self.plan);p['items'][0]['object_height_mm']=value
            with self.subTest(height=value),self.assertRaises(ValueError):height_plan(p)
        for key,value in [('mask_rle',[1201]),('mask_rle',[1200]),('item_id',2)]:
            p=copy.deepcopy(self.plan);p['items'][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):height_plan(p)

    def test_operations_freeze_and_rollback_the_entire_plan(self):
        from sim.test_prompted_height import HeightTests
        operations=HeightTests().operations()
        data=dict(action='preview',token='page',height_plan=self.plan)
        with patch('src.monitor_operation.Operations.command',return_value='accepted'):
            self.assertEqual(operations.command(data),'accepted')
        self.plan['items'][0]['object_height_mm']=90
        self.assertEqual(operations.backend.height_plan['items'][0]['object_height_mm'],40)
        previous=copy.deepcopy(operations.backend.height_plan)
        with patch('src.monitor_operation.Operations.command',side_effect=ValueError('rejected')):
            with self.assertRaises(ValueError):operations.command(data)
        self.assertEqual(operations.backend.height_plan,previous)
        with self.assertRaisesRegex(ValueError,'conflict'):operations.command(dict(data,object_height_mm=56))

    def test_board_adapter_forwards_plan_without_scalar(self):
        received=[];original=SimpleNamespace(object_height_mm=None,height_plan=None)
        backend=BoardPreviewBackend(original,SimpleNamespace(plan=lambda p:received.append(p) or {}))
        backend.height_plan=self.plan;backend.cycle('AUTO',False)
        self.assertEqual(received[0]['height_plan'],self.plan)
        self.assertNotIn('object_height_mm',received[0])
        self.assertIs(backend.height_plan,original.height_plan)


class SelectionTests(unittest.TestCase):
    def test_new_height_mode_rechecks_all_obstacles_even_with_legacy_scene_matching_disabled(self):
        from src.sorting_prompted_motion import PromptedMotionController
        from src.sorting_unified import UnifiedController
        controller=object.__new__(PromptedMotionController)
        controller.c={'min_scene_iou':.85};controller.motion_options={'full_scene_instance_stability_check_enabled':False}
        a=np.zeros((30,40),bool);a[2:8,2:8]=1;b=a.copy();b[:]=0;b[20:26,20:26]=1
        obj=dict(label='metal',official_label='Drink can')
        resolver=ItemHeights(plan_for([(obj,a)],[40,30],[56]),model='a'*64,configuration='b'*64,image_size=[40,30])
        controller.item_heights=resolver
        controller.source_objects=lambda r:r['pairs']
        before={'pairs':[(obj,a)],'observation':{'source_stream_id':'stream'}}
        after={'pairs':[(dict(obj),b)],'observation':{'source_stream_id':'stream'}}
        with self.assertRaisesRegex(ValueError,'missing_split_or_moved'):controller.unchanged(before,after)

    def test_lexicographic_clearance_duration_confidence_and_stable_tie(self):
        def candidate(track,clearance,seconds,confidence):
            return (track,({'confidence':confidence},None),{'nominal_neighbor_clearance_mm':clearance},[('move',b'',seconds)])
        candidates=[candidate(1,3,1,.99),candidate(4,10,8,.8),candidate(3,10,8,.9),candidate(2,10,4,.8)]
        rank_candidates(candidates,lambda name,ms:ms)
        self.assertEqual([v[0] for v in candidates],[2,3,4,1])
        self.assertEqual([v[2]['selection']['rank'] for v in candidates],[1,2,3,4])
        rank_candidates([candidate(1,None,3,.8)],lambda name,ms:ms)

    def test_actual_route_compilation_uses_per_object_height_and_keeps_grip_z(self):
        from sim.test_sorting_unified import UnifiedBatchTests
        from src.sorting_loop import mapped
        fixture=UnifiedBatchTests();fixture.setUp()
        try:
            with fixture.journal:
                controller=fixture.factory(0);controller.mode='preview'
                row=controller.acquire();row['observation']['source_stream_id']='stream'
                for obj in row['observation']['instances']:obj['official_label']=obj['label']
                pairs=controller.source_objects(row);heights=[60+i*5 for i in range(len(pairs))]
                plan=plan_for(pairs,fixture.c['image_size'],heights)
                controller.item_heights=ItemHeights(plan,model='a'*64,configuration='b'*64,image_size=fixture.c['image_size'])
                controller.geometry.pose=controller.geometry.camera=controller.geometry.station=controller.geometry.registration={}
                calls=[]
                def project(points,height,*args):
                    calls.append(height)
                    return mapped(fixture.c['profiles']['metal']['image_to_card_mm'],points)+[height*.001,0]
                with patch('src.sorting_relocated_geometry.pixels_to_card',side_effect=project):
                    selected,rejected=controller.select(row)
                self.assertFalse(rejected);self.assertIsNotNone(selected)
                self.assertEqual(set(calls),set(heights))
                self.assertEqual(len(controller.candidate_order),3)
                for choice in controller.candidate_order:
                    item=plan['items'][choice['height_item_id']-1]
                    self.assertEqual(choice['object_height_mm'],item['object_height_mm'])
                    self.assertEqual(choice['grasp_z_mm'],fixture.c['profiles'][choice['label']]['grasp_z_mm'])
                    self.assertTrue(choice['complete_route_compiled_before_motion'])
                self.assertFalse(any(m.sent for m in fixture.motions))
        finally:fixture.doCleanups()

    def test_height_below_existing_grasp_z_rejects_without_uart(self):
        from sim.test_sorting_unified import UnifiedBatchTests
        fixture=UnifiedBatchTests();fixture.setUp()
        try:
            controller=fixture.factory(0);row=controller.acquire();pairs=controller.source_objects(row)
            target=next(p for p in pairs if p[0]['label']=='paper')
            target[0]['official_label']='paper'
            plan=plan_for([target],fixture.c['image_size'],[20])
            resolver=ItemHeights(plan,model='a'*64,configuration='b'*64,image_size=fixture.c['image_size'])
            resolver.bind([target],'stream',.85);controller.item_heights=resolver
            previous=controller.c['profiles']['paper']['object_height_mm']
            with self.assertRaisesRegex(ValueError,'below_existing_grasp_z'):controller.candidate(row,target,1)
            self.assertEqual(controller.c['profiles']['paper']['object_height_mm'],previous)
            self.assertFalse(any(m.sent for m in fixture.motions))
        finally:fixture.doCleanups()


if __name__=='__main__':unittest.main()
