import copy
import json
from pathlib import Path
import tempfile
import unittest
from src.execution_models import build_profiles,ExecutionModels,digest
from src.execution_model_context import model_context
from src.prompted_recyclables import settings_for,read_policy,classify
from src.prompted_trial_scope import execution_scope


def fixture():
    materials={m:dict(bin_id='bin_'+m,source_confidence=.3,bin_confidence=.3) for m in ('paper','plastic','metal')}
    mapping={'Drink can':'metal','Normal paper':'paper','Clear plastic bottle':'plastic'}
    policy=dict(schema='prompted-recyclables-policy-v1',model=dict(name='old.pt',sha256='a'*64,fixed_classes=True),
        classes=[dict(official_label=n,material=m,grasp_confidence=.3) for n,m in mapping.items()],
        materials=materials,detection_floor=.3,display_confidence=.3,heights_mm=dict(paper=65,plastic=53,metal=56),height_range_mm=[1,120])
    settings=settings_for(policy,{})
    receipt=dict(eligible_materials=['metal'],eligible_official_labels=['Drink can'],evaluation_sha256='old-evidence')
    catalog=dict(bound_model='old',device='cuda:0',worker_ports=[8831,8832],models=[
        dict(id='old',name='Original',kind='service',sha256='a'*64,url='http://127.0.0.1:8824'),
        dict(id='new',name='New .10',kind='checkpoint',sha256='b'*64,classes=list(mapping),mapping=mapping,
             family='yoloe',confidence=.1,configuration_sha256='c'*64),
        dict(id='three',name='Three classes',kind='checkpoint',sha256='d'*64,classes=['can','plastic','cardboard'],
             mapping=dict(can='metal',plastic='plastic',cardboard='paper'),family='yolov8s',configuration_sha256='e'*64)])
    return policy,settings,receipt,catalog


class ProfileTests(unittest.TestCase):
    def test_selected_weights_taxonomy_thresholds_and_unchanged_heights(self):
        p,s,r,c=fixture();result=build_profiles(c,p,s,r);new=result['profiles'][1]
        self.assertEqual(new['policy']['model']['sha256'],'b'*64)
        self.assertEqual(new['settings']['confidence'],.1)
        self.assertEqual(set(new['settings']['class_thresholds'].values()),{.1})
        self.assertEqual(new['settings']['heights_mm'],s['heights_mm'])
        self.assertEqual(new['receipt']['eligible_official_labels'],[])
        with model_context(new):
            active=read_policy();scope=execution_scope(new['receipt'],active,{})
            self.assertEqual(scope['official_labels'],list(c['models'][1]['mapping']))
            self.assertEqual(scope['validated_scope']['official_labels'],[])

    def test_three_class_model_uses_its_own_labels(self):
        p,s,r,c=fixture();three=build_profiles(c,p,s,r)['profiles'][2]
        with model_context(three):
            decision=classify(read_policy(),three['settings'],dict(label='can',confidence=.7))
        self.assertEqual(decision['material'],'metal');self.assertEqual(decision['official_label'],'can')

    def test_reference_assistance_is_explicit_and_original_label_is_retained(self):
        p,s,r,c=fixture();c['models'][1]['reference_bank_sha256']='f'*64
        profile=build_profiles(c,p,s,r)['profiles'][1]
        obj=dict(label='Drink can',confidence=.5,reference_assisted=True,reference_material='paper',model_material='metal')
        decision=classify(profile['policy'],profile['settings'],obj)
        self.assertEqual(decision['material'],'paper');self.assertEqual(decision['official_label'],'Drink can')
        self.assertEqual(decision['model_material'],'metal')
        with self.assertRaisesRegex(ValueError,'unregistered_reference'):classify(p,s,obj)

    def test_incomplete_mapping_is_not_silently_accepted(self):
        p,s,r,c=fixture();c['models'][1]['mapping'].pop('Drink can')
        with self.assertRaisesRegex(ValueError,'taxonomy'):build_profiles(c,p,s,r)


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory();self.addCleanup(self.folder.cleanup)
        self.path=Path(self.folder.name)/'model.json'
        p,s,r,c=fixture();self.catalog=build_profiles(c,p,s,r);self.models=ExecutionModels(self.catalog,self.path)
    def bind(self,name='new',revision=0,request='request-new-1',idle=lambda:None):
        return self.models.activate(name,revision,request,idle,'1'*32)
    def test_commit_is_persistent_and_does_not_touch_pending(self):
        pending=self.path.with_name('pending.json');pending.write_text('original unresolved')
        value=self.bind();again=ExecutionModels(self.catalog,self.path)
        self.assertEqual(again.public(),value);self.assertEqual(pending.read_text(),'original unresolved')
    def test_busy_cas_and_unknown_profile_preserve_old_binding(self):
        def busy():raise ValueError('busy')
        for args in [dict(idle=busy),dict(revision=99),dict(name='missing')]:
            with self.assertRaises(ValueError):self.bind(**args)
            self.assertEqual(self.models.public()['profile_id'],'old');self.assertFalse(self.path.exists())
    def test_operation_context_stays_frozen_across_external_selection(self):
        old=self.models.snapshot()
        with model_context(old):
            self.bind()
            self.assertEqual(read_policy()['model']['sha256'],'a'*64)
        with self.models.context():self.assertEqual(read_policy()['model']['sha256'],'b'*64)
    def test_profile_hash_change_cannot_be_restored_as_old_identity(self):
        self.bind();changed=copy.deepcopy(self.catalog);changed['profiles'][1]['settings']['confidence']=.01
        with self.assertRaisesRegex(ValueError,'profile_changed'):ExecutionModels(changed,self.path)
