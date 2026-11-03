import copy
import hashlib
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

sys.path.insert(0,str(Path(__file__).parents[1]))
from switchable_console import ModelSelection


class RestoreTests(unittest.TestCase):
    def setUp(self):
        self.binding=dict(profile_id='current',profile_sha256='profile',route_id='1'*32,revision=7)
        self.profile=dict(model_sha256='weight',profile_sha256='profile',upstream_configuration_sha256='configuration',settings=dict(threshold=.1))
        self.row=dict(id='current',kind='checkpoint',sha256='weight',configuration_sha256='configuration',device='cuda:0')
        self.health=dict(model_sha256='weight',configuration_sha256='configuration',execution=dict(device='cuda:0'),generation=3,ready=True)
        self.live=dict(self.health,model_version='weight',source_stream_id='stream',result_age_ms=20)
        self.relay=dict(generation=hashlib.sha256(json.dumps(['profile','1'*32,3]).encode()).hexdigest(),settings=dict(threshold=.1),execution_profile=dict(profile_sha256='profile'))
        self.selection=ModelSelection.__new__(ModelSelection);s=self.selection
        s.motion_bridge=SimpleNamespace(status=lambda:copy.deepcopy(self.binding),profiles={'current':self.profile},binding=self.binding)
        s.models={'current':self.row};s.worker_pool=SimpleNamespace(find=Mock(return_value=(object(),'http://127.0.0.1:8831')))
        s.request_json=Mock(side_effect=lambda base,path:self.health if path=='/health' else self.live)
        s.bound_route_health=Mock(side_effect=lambda binding:self.relay);s.lock=threading.RLock();s.selected='default';s.current_url='old'
        s.binding_unknown=True;s.error='old';s.config={'initial_model':'current'};s.select=Mock(side_effect=AssertionError('must not load/rebind'))

    def test_proxy_restart_restores_existing_identity_without_post_load_or_bind(self):
        self.selection.start_initial()
        self.assertEqual(self.selection.selected,'current');self.assertEqual(self.selection.current_url,'http://127.0.0.1:8831')
        self.assertEqual({call.args[1] for call in self.selection.request_json.call_args_list},{'/health','/live-result'})
        self.selection.select.assert_not_called()

    def test_wrong_weight_configuration_device_stale_frame_or_relay_generation_reject(self):
        for value,key,changed in ((self.live,'model_version','wrong'),(self.health,'configuration_sha256','wrong'),
            (self.live,'execution',dict(device='cpu')),(self.live,'result_age_ms',6000),(self.relay,'generation','wrong')):
            old=value[key];value[key]=changed
            with self.subTest(key=key),self.assertRaises(ValueError):self.selection.restore_running_binding()
            self.assertEqual(self.selection.selected,'default');value[key]=old

    def test_missing_exact_worker_lease_is_never_adopted(self):
        self.selection.worker_pool.find.return_value=None
        self.assertFalse(self.selection.restore_running_binding());self.selection.request_json.assert_not_called()

if __name__=='__main__':unittest.main()
