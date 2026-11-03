"""Registered perception profiles may share existing supervised mechanics.

Binding a profile does not execute motion, clear pending state, or promote its
recognition/physical validation. Only registered identities can be selected.
"""
import copy
import hashlib
import json
from pathlib import Path
import threading
from .execution_model_context import model_context


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def build_profiles(catalog,base_policy,base_settings,base_receipt,service_profiles=None):
    from .prompted_recyclables import settings_for
    from .prompted_trial_scope import taxonomy_sha256
    service_profiles=service_profiles or {}
    profiles=[]
    for row in catalog['models']:
        if row['kind']=='unavailable' or row.get('disabled_reason'):continue
        if row['id']==catalog['bound_model']:
            policy,settings=copy.deepcopy(base_policy),copy.deepcopy(base_settings)
            if policy['model']['sha256']!=row['sha256']:raise ValueError('original_model_profile_mismatch')
        elif row['kind']=='service':
            source=service_profiles[row['id']]
            policy,settings=copy.deepcopy(source['policy']),copy.deepcopy(source['settings'])
            if policy['model']['sha256']!=row['sha256'] or digest(policy)!=settings['policy_sha256']:
                raise ValueError('service_model_profile_mismatch')
        else:
            classes,mapping=row['classes'],row['mapping']
            if not classes or len(set(classes))!=len(classes) or set(mapping)!=set(classes):
                raise ValueError('incomplete_execution_taxonomy')
            if not set(mapping.values())<=set(('paper','plastic','metal')):raise ValueError('unsupported_execution_material')
            confidence=row.get('confidence',.30)
            if type(confidence) not in (int,float) or not .01<=confidence<=.95:raise ValueError('invalid_execution_confidence')
            policy=copy.deepcopy(base_policy)
            policy['model']=dict(name=row['id']+'.pt',sha256=row['sha256'],fixed_classes=True,
                                 family=row['family'],physical_validation_completed=False)
            old={r['official_label']:r for r in policy['classes']}
            policy['classes']=[dict(old.get(n,{}),official_label=n,material=mapping[n],grasp_confidence=confidence) for n in classes]
            policy['detection_floor']=policy['display_confidence']=confidence
            for material in policy['materials'].values():
                material['source_confidence']=material['bin_confidence']=confidence
            if row.get('reference_bank_sha256'):policy['reference_bank_sha256']=row['reference_bank_sha256']
            else:policy.pop('reference_bank_sha256',None)
            settings=settings_for(policy,{})
        if settings_for(policy,settings)!=settings:raise ValueError('execution_settings_not_canonical')
        if policy['model']['sha256']!=row['sha256']:raise ValueError('execution_weight_mismatch')
        labels=[r['official_label'] for r in policy['classes'] if r['material'] in ('paper','plastic','metal')]
        if not labels:raise ValueError('execution_profile_has_no_supported_classes')
        receipt=copy.deepcopy(base_receipt)
        if settings!=base_settings or policy!=base_policy:
            receipt.update(eligible_materials=[],eligible_official_labels=[])
        receipt.update(model_sha256=row['sha256'],original_validation_receipt_sha256=digest(base_receipt),
            evaluation_scope='original_model_only; selected_profile_is_operator_supervised_trial',
            physical_validation_completed=False)
        scope=dict(kind='operator-supervised-trial-scope-v1',
            operator_statement='2026-10-31: operator requested that execution follow the selected registered model and configuration.',
            physical_validation_completed=False,single_manual_attempt_only=True,model_sha256=row['sha256'],
            taxonomy_sha256=taxonomy_sha256(policy['classes']),official_labels=labels)
        profile=dict(id=row['id'],name=row['name'],model_sha256=row['sha256'],
            upstream_configuration_sha256=row.get('configuration_sha256'),protocol=row.get('protocol','candidate'),
            family=row.get('family','yoloe'),policy=policy,settings=settings,receipt=receipt,scope=scope,
            device=row.get('device',catalog.get('device')),reference_bank_sha256=row.get('reference_bank_sha256'),
            service_url=row.get('url'),allowed_worker_ports=list(catalog['worker_ports']))
        profile['profile_sha256']=digest(profile)
        profiles.append(profile)
    return dict(schema='execution-model-profiles-v1',default_model=catalog['bound_model'],profiles=profiles)


class ExecutionModels:
    def __init__(self,catalog,state_path):
        if catalog.get('schema')!='execution-model-profiles-v1':raise ValueError('invalid_execution_model_catalog')
        self.profiles={p['id']:copy.deepcopy(p) for p in catalog['profiles']}
        if len(self.profiles)!=len(catalog['profiles']):raise ValueError('duplicate_execution_profiles')
        for p in self.profiles.values():
            body={k:v for k,v in p.items() if k!='profile_sha256'}
            if digest(body)!=p['profile_sha256']:raise ValueError('execution_profile_changed')
        self.path=Path(state_path);self.lock=threading.RLock()
        self.active=catalog['default_model'];self.revision=0;self.request_id=None;self.route_id=None
        if self.active not in self.profiles:raise ValueError('missing_default_execution_profile')
        if self.path.exists():
            previous=json.loads(self.path.read_text())
            if previous['profile_id'] not in self.profiles or self.profiles[previous['profile_id']]['profile_sha256']!=previous['profile_sha256']:
                raise ValueError('saved_execution_binding_identity_changed')
            if (type(previous.get('revision')) is not int or previous['revision']<0 or
                not isinstance(previous.get('route_id'),str) or len(previous['route_id'])!=32 or
                any(c not in '0123456789abcdef' for c in previous['route_id'])):
                raise ValueError('saved_execution_binding_state_invalid')
            self.active=previous['profile_id'];self.revision=previous['revision'];self.request_id=previous['request_id'];self.route_id=previous.get('route_id')

    def snapshot(self):
        with self.lock:
            profile=copy.deepcopy(self.profiles[self.active])
            profile['binding_revision']=self.revision
            profile['route_id']=self.route_id
            return profile

    def public(self):
        with self.lock:
            p=self.snapshot()
            return dict(profile_id=p['id'],name=p['name'],model_sha256=p['model_sha256'],
                profile_sha256=p['profile_sha256'],configuration_sha256=p['upstream_configuration_sha256'],
                settings_sha256=p['settings']['settings_sha256'],policy_sha256=p['settings']['policy_sha256'],
                revision=p['binding_revision'],request_id=self.request_id,route_id=p['route_id'],physical_validation_completed=False)

    def context(self):
        return model_context(self.snapshot())

    def activate(self,profile_id,expected_revision,request_id,check_idle,route_id=None):
        with self.lock:
            if profile_id not in self.profiles:raise ValueError('unregistered_execution_profile')
            if type(expected_revision) is not int or expected_revision!=self.revision:raise ValueError('execution_binding_revision_changed')
            if not isinstance(request_id,str) or not 8<=len(request_id)<=80 or request_id==self.request_id:
                raise ValueError('invalid_or_duplicate_execution_binding_request')
            if not isinstance(route_id,str) or len(route_id)!=32 or any(c not in '0123456789abcdef' for c in route_id):
                raise ValueError('invalid_execution_route_identity')
            check_idle()
            p=self.profiles[profile_id]
            record=dict(profile_id=profile_id,profile_sha256=p['profile_sha256'],revision=self.revision+1,request_id=request_id,route_id=route_id)
            self.path.parent.mkdir(parents=True,exist_ok=True)
            temporary=self.path.with_suffix('.tmp')
            with temporary.open('w',encoding='utf-8') as f:
                import os
                json.dump(record,f);f.flush();os.fsync(f.fileno())
            temporary.replace(self.path)
            self.active=profile_id;self.revision+=1;self.request_id=request_id;self.route_id=route_id
            return self.public()
