"""Commit a verified model selection to PS; never submit an arm operation."""
import json
from pathlib import Path
import urllib.request
import uuid


class BindingOutcomeUnknown(RuntimeError):
    preserve_candidate=True


class ModelMotionBridge:
    def __init__(self,board,catalog,relay_url,relay_token):
        self.board=board;self.profiles={p['id']:p for p in catalog['profiles']}
        self.relay_url=relay_url;self.relay_token=relay_token;self.binding=None;self.known=False
        self.status()

    def status(self):
        code,raw,_=self.board.request('/execution-model')
        if code!=200:self.known=False;raise ValueError('execution_binding_status_unavailable')
        value=json.loads(raw)
        p=self.profiles.get(value.get('profile_id'))
        if p is None or value.get('profile_sha256')!=p['profile_sha256']:
            self.known=False;raise ValueError('execution_binding_catalog_mismatch')
        self.binding=value;self.known=True
        return value

    def matches(self,model_id):
        return self.known and self.binding['profile_id']==model_id and self.binding['profile_sha256']==self.profiles[model_id]['profile_sha256']

    def observe(self,value):
        if not isinstance(value,dict):return
        p=self.profiles.get(value.get('profile_id'))
        if p is None or value.get('profile_sha256')!=p['profile_sha256']:self.known=False;return
        if self.binding and value.get('revision',-1)<self.binding['revision']:return
        self.binding=dict(value);self.known=True

    def bind(self,row,url):
        p=self.profiles[row['id']]
        if row['sha256']!=p['model_sha256'] or row.get('configuration_sha256')!=p['upstream_configuration_sha256']:
            raise ValueError('selected_execution_profile_mismatch')
        payload=dict(profile_id=p['id'],profile_sha256=p['profile_sha256'],url=url)
        req=urllib.request.Request(self.relay_url+'/execution-model/register',data=json.dumps(payload).encode(),
            headers={'Content-Type':'application/json','Authorization':'Bearer '+self.relay_token})
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req,timeout=15) as response:registered=json.load(response)
        if registered.get('profile_sha256')!=p['profile_sha256']:raise ValueError('execution_visual_registration_mismatch')
        previous=self.status();request_id=uuid.uuid4().hex
        data=dict(profile_id=p['id'],profile_sha256=p['profile_sha256'],expected_revision=previous['revision'],request_id=request_id,route_id=registered['route_id'])
        try:
            code,raw,_=self.board.request('/execution-model/bind',json.dumps(data).encode())
            if code!=200:raise ValueError('execution_model_binding_rejected:'+raw.decode()[:240])
            value=json.loads(raw)
            if value.get('profile_sha256')!=p['profile_sha256'] or value.get('request_id')!=request_id:
                raise ValueError('execution_model_binding_acknowledgement_changed')
            self.binding=value;self.known=True
            return value
        except (OSError,ValueError) as original:
            # Read-only resolution of one uncertain commit; never repeat the POST.
            try:value=self.status()
            except (OSError,ValueError) as exc:
                self.known=False
                raise BindingOutcomeUnknown('执行模型绑定回执未确认；保留候选服务，请核对连接。') from exc
            if value.get('request_id')==request_id and value.get('profile_sha256')==p['profile_sha256']:
                return value
            raise original
