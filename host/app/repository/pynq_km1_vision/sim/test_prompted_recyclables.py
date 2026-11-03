import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from src.prompted_recyclables import read_policy, settings_for, classify, adapt
from src.sorting_prompted_preview import project_row, stable_official_labels, PromptedPreviewController
from src.yoloe_inference import encode_mask
from src.yoloe_recyclable_monitor import RecyclablePlayground


class PromptedPolicyTests(unittest.TestCase):
    def setUp(self):
        self.p=read_policy();self.s=settings_for(self.p,{})

    def obj(self,label='Clear plastic bottle',score=.8):
        return dict(label=label,confidence=score)

    def test_preserves_exact_official_label_and_separate_material(self):
        row=classify(self.p,self.s,self.obj('Food Can'))
        self.assertEqual(row['label'],'Food Can')
        self.assertEqual(row['official_label'],'Food Can')
        self.assertEqual((row['material'],row['sorting_bin']),('metal','bin_metal'))
        self.assertFalse(row['automatic_motion_allowed'])

    def test_glass_and_composite_never_forced_into_material(self):
        for name in ['Glass bottle','Battery','Drink carton','Carded blister pack','not a TACO label']:
            row=classify(self.p,self.s,self.obj(name,.99))
            self.assertIsNone(row['material']);self.assertFalse(row['semantic_gate_passed'])
            self.assertTrue(row['obstacle'])

    def test_stricter_class_or_zone_gate_wins(self):
        s=settings_for(self.p,dict(material_thresholds={'plastic':{'source_confidence':.85,'bin_confidence':.9}}))
        self.assertFalse(classify(self.p,s,self.obj(score=.8))['semantic_gate_passed'])
        self.assertTrue(classify(self.p,s,self.obj(score=.87))['semantic_gate_passed'])
        self.assertFalse(classify(self.p,s,self.obj(score=.87),'bin_plastic')['semantic_gate_passed'])
        self.assertFalse(classify(self.p,s,self.obj(score=.99),'bin_paper')['semantic_gate_passed'])

    def test_display_threshold_never_discards_obstacle(self):
        row=classify(self.p,self.s,self.obj(score=.15))
        self.assertFalse(row['visible']);self.assertTrue(row['obstacle'])
        self.assertFalse(row['semantic_gate_passed'])

    def test_target_subset_only_prompts_selected_labels(self):
        s=settings_for(self.p,dict(classes=['Clear plastic bottle']))
        self.assertEqual(s['classes'],['Clear plastic bottle'])
        self.assertNotIn('Drink can',s['classes'])
        with self.assertRaises(ValueError):settings_for(self.p,dict(classes=['plastic bottle']))

    def test_height_and_threshold_validation(self):
        for value in [0,121,float('nan'),True,'53']:
            with self.assertRaises(ValueError):settings_for(self.p,dict(heights_mm={'plastic':value}))
        for value in [-1,1.1,float('inf'),True]:
            with self.assertRaises(ValueError):settings_for(self.p,dict(class_thresholds={'Drink can':value}))
        with self.assertRaises(ValueError):settings_for(self.p,dict(material_thresholds={'metal':{'bin_id':'bin_paper'}}))

    def test_settings_replay_and_height_hash(self):
        self.assertEqual(settings_for(self.p,self.s),self.s)
        updated=settings_for(self.p,dict(heights_mm={'metal':60}))
        self.assertNotEqual(updated['settings_sha256'],self.s['settings_sha256'])
        self.assertEqual(updated['policy_sha256'],self.s['policy_sha256'])

    def test_wrong_weights_rejected(self):
        with self.assertRaises(ValueError):adapt(self.p,self.s,dict(model_sha256='bad',instances=[]))

    def result(self,label='Clear plastic bottle',score=.8):
        image=np.zeros((20,30,3),np.uint8);mask=np.zeros((20,30),bool);mask[2:10,3:8]=True
        obj=dict(self.obj(label,score),bbox_xyxy=[3,2,8,10],mask_rle=encode_mask(mask))
        return image,dict(instances=[obj],image_size=[30,20],model_sha256=self.p['model']['sha256'],
            settings_sha256=self.s['settings_sha256'],source_bgr_sha256=hashlib.sha256(image.tobytes()).hexdigest())

    def test_projection_preserves_unknown_and_weak_obstacles(self):
        for label,score in [('Clear plastic bottle',.15),('Glass bottle',.99)]:
            image,result=self.result(label,score)
            row=project_row(self.p,self.s,result,image,{})
            if label=='Glass bottle':
                self.assertEqual(len(row['excluded']),1)
                with self.assertRaisesRegex(ValueError,'excluded'):stable_official_labels([row]*3)
            else:self.assertEqual(len(row['masks']),1)

    def test_changed_pixels_and_masks_fail_binding(self):
        image,result=self.result();image[0,0,0]=1
        with self.assertRaisesRegex(ValueError,'binding'):project_row(self.p,self.s,result,image,{})
        image,result=self.result();result['instances'][0]['bbox_xyxy']=[0,0,1,1]
        with self.assertRaisesRegex(ValueError,'bbox'):project_row(self.p,self.s,result,image,{})

    def test_material_and_mask_stability_survives_raw_class_change(self):
        image,result=self.result();first=project_row(self.p,self.s,result,image,{})
        second=copy.deepcopy(first);second['observation']['instances'][0]['official_label']='Other plastic bottle'
        self.assertIs(stable_official_labels([first,second,first]),first)
        second['observation']['instances'][0]['label']='metal'
        with self.assertRaisesRegex(ValueError,'material_or_mask'):stable_official_labels([first,second,first])

    def test_weakest_frame_blocks_candidate(self):
        image,result=self.result();first=project_row(self.p,self.s,result,image,{})
        image,result=self.result(score=.55);second=project_row(self.p,self.s,result,image,{})
        merged=stable_official_labels([first,second,copy.deepcopy(first)])
        self.assertFalse(merged['observation']['instances'][0]['semantic_gate_passed'])
        controller=object.__new__(PromptedPreviewController)
        with self.assertRaisesRegex(ValueError,'confidence'):
            controller.candidate(merged,(merged['observation']['instances'][0],merged['masks'][0]),'0')

    def test_policy_source_exact_name_fixture(self):
        labels={r['official_label'] for r in self.p['classes']}
        self.assertEqual(len(labels),24);self.assertNotIn('Plastic glooves',labels)
        self.assertEqual(sum(r['material'] is not None for r in self.p['classes']),24)

    def test_global_point_six_covers_detection_display_classes_and_zones(self):
        self.assertEqual(self.s['confidence'],.6)
        self.assertEqual(self.s['display_confidence'],.6)
        self.assertEqual(set(self.s['class_thresholds'].values()),{.6})
        for values in self.s['material_thresholds'].values():
            self.assertEqual(values,dict(source_confidence=.6,bin_confidence=.6))
        self.assertTrue(classify(self.p,self.s,self.obj('Drink can',.6))['semantic_gate_passed'])
        self.assertFalse(classify(self.p,self.s,self.obj('Drink can',.599))['semantic_gate_passed'])


class SavedSettingsTests(unittest.TestCase):
    def test_pause_restart_preserves_heights_prompts_and_thresholds(self):
        class Detector:
            policy=read_policy();execution={};hashes={};embeddings={};vocabulary=[]
            def prepare(self,payload):return settings_for(self.policy,payload)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'settings.json'
            first=RecyclablePlayground(Detector(),'http://127.0.0.1:8780/frame.jpg',path,8811)
            self.assertFalse(first.enabled)
            first.configure(dict(heights_mm={'paper':42},class_thresholds={'Normal paper':.88},enabled=True))
            first.configure(dict(heights_mm={'plastic':40}))
            first.configure(dict(enabled=False))
            second=RecyclablePlayground(Detector(),'http://127.0.0.1:8780/frame.jpg',path,8811)
            self.assertFalse(second.enabled)
            self.assertEqual(second.settings['heights_mm']['paper'],42)
            self.assertEqual(second.settings['class_thresholds']['Normal paper'],.88)
            self.assertEqual(second.settings['heights_mm']['plastic'],40)


class MotionAdapterTests(unittest.TestCase):
    def test_three_materials_compile_existing_routes_without_motion(self):
        from sim.test_sorting_unified import UnifiedBatchTests
        from src.sorting_io import Statistics
        originals={'plastic':'Clear plastic bottle','metal':'Drink can','paper':'Corrugated carton'}
        for material,official in originals.items():
            with self.subTest(material=material):
                fixture=UnifiedBatchTests()
                fixture.setUp()
                try:
                    fixture.source.items=[i for i in fixture.source.items if i['label']==material]
                    with fixture.journal:
                        controller=PromptedPreviewController(fixture.c,fixture.source,Statistics(),None,
                            fixture.journal,fixture.geometry,fixture.source.reference,fixture.root,
                            policy=fixture.policy,tracker=fixture.tracker,mode='preview',
                            clock=fixture.clock.monotonic,sleep=fixture.clock.sleep)
                        row=controller.acquire()
                        obj=row['observation']['instances'][0]
                        obj.update(official_label=official,semantic_gate_passed=True,zone='source',decision_threshold=.75)
                        original_z=controller.c['profiles'][material]['grasp_z_mm']
                        controller.c['profiles'][material]['object_height_mm']+=2
                        controller.policy['models'][material]['object_height_mm']+=2
                        choice,plan=controller.candidate(row,(obj,row['masks'][0]),'test')
                        self.assertEqual(choice['official_label'],official)
                        self.assertEqual(choice['bin_id'],'bin_'+material)
                        self.assertGreater(len(plan),5)
                        self.assertFalse(choice['automatic_motion_allowed'])
                        self.assertEqual(controller.c['profiles'][material]['grasp_z_mm'],original_z)
                        self.assertIsNone(controller.motion)
                finally:fixture.doCleanups()


if __name__=='__main__':unittest.main()
