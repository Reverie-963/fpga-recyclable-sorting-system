import copy
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
sys.path.insert(0,str(Path(__file__).parents[1]))
from execution_visual_relay import ProfileRouter
from model_motion_bridge import ModelMotionBridge,BindingOutcomeUnknown
from test_switchable_console import SwitchTests


def profile():
    return dict(id='new',profile_sha256='profile',model_sha256='weight',upstream_configuration_sha256='configuration',
        device='cuda:0',policy={'model':{'name':'new.pt'},'classes':[]},settings={'settings_sha256':'settings','policy_sha256':'policy'},
        allowed_worker_ports=[8831],service_url=None)


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.value=dict(model_sha256='weight',configuration_sha256='configuration',runtime_id='runtime',
                        model_loaded=True,execution={'device':'cuda:0'},source_bgr_sha256='pixels',instances=[])
        self.router=ProfileRouter(dict(default_model='new',profiles=[profile()]),lambda *a:copy.deepcopy(self.value))
    def register(self,url='http://127.0.0.1:8831'):
        return self.router.register(dict(profile_id='new',profile_sha256='profile',url=url))
    def test_new_routes_do_not_replace_an_operation_snapshot(self):
        one=self.register();before=self.router.health('new',one['route_id'])
        two=self.register();after=self.router.health('new',one['route_id'])
        self.assertEqual(before,after);self.assertNotEqual(one['route_id'],two['route_id'])
        result=self.router.predict('new',{},one['route_id'])
        self.assertEqual(result['generation'],before['generation']);self.assertEqual(result['settings_sha256'],'settings')
    def test_weight_configuration_device_and_port_are_enforced(self):
        for field,value in [('model_sha256','wrong'),('configuration_sha256','wrong'),('execution',{'device':'cpu'})]:
            old=self.value[field];self.value[field]=value
            with self.assertRaises(ValueError):self.register()
            self.value[field]=old
        for url in ('https://example.com:8831','http://127.0.0.1:9999','http://user@127.0.0.1:8831'):
            with self.assertRaises(ValueError):self.register(url)


class BridgeTests(unittest.TestCase):
    def setUp(self):
        class Register(BaseHTTPRequestHandler):
            def log_message(self,*a):pass
            def do_POST(self):
                value=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                raw=json.dumps(dict(profile_sha256=value['profile_sha256'],route_id='1'*32)).encode()
                self.send_response(200);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Register)
        threading.Thread(target=self.server.serve_forever,daemon=True).start()
        self.addCleanup(self.server.server_close);self.addCleanup(self.server.shutdown)
        self.posts=0;self.reads_fail=False;self.commit=True;self.lost=False
        self.current=dict(profile_id='old',profile_sha256='old-profile',revision=0)
        outer=self
        class Board:
            def request(self,path,body=None):
                if body is None:
                    if outer.reads_fail:raise ConnectionError('lost readback')
                    return 200,json.dumps(outer.current).encode(),{}
                outer.posts+=1;data=json.loads(body)
                if outer.commit:outer.current=dict(profile_id='new',profile_sha256='profile',revision=1,request_id=data['request_id'],route_id=data['route_id'])
                if outer.lost:
                    if outer.lost=='all':outer.reads_fail=True
                    raise ConnectionError('lost acknowledgement')
                return 200,json.dumps(outer.current).encode(),{}
        old=dict(profile(),id='old',profile_sha256='old-profile')
        self.bridge=ModelMotionBridge(Board(),dict(profiles=[old,profile()]),'http://127.0.0.1:'+str(self.server.server_port),'private')
        self.row=dict(id='new',sha256='weight',configuration_sha256='configuration')
    def test_one_commit_binds_the_selected_model(self):
        self.bridge.bind(self.row,'http://127.0.0.1:8831');self.assertTrue(self.bridge.matches('new'));self.assertEqual(self.posts,1)
    def test_lost_ack_is_resolved_by_readback_without_second_post(self):
        self.lost=True;self.bridge.bind(self.row,'http://127.0.0.1:8831')
        self.assertTrue(self.bridge.matches('new'));self.assertEqual(self.posts,1)
    def test_unknown_outcome_preserves_candidate_and_blocks_admission(self):
        self.lost='all'
        with self.assertRaises(BindingOutcomeUnknown) as e:self.bridge.bind(self.row,'http://127.0.0.1:8831')
        self.assertTrue(e.exception.preserve_candidate);self.assertFalse(self.bridge.known);self.assertEqual(self.posts,1)


class FollowSwitchTests(SwitchTests):
    # Only the additional tests are collected below; inherited legacy tests run
    # separately with their unchanged recognition-only configuration.
    def enable_follow(self):
        outer=self
        class Bridge:
            binding=dict(profile_id='bound',profile_sha256='bound-profile',revision=0)
            failed=False
            def matches(self,name):return self.binding['profile_id']==name
            def observe(self,value):pass
            def bind(self,row,url):
                if self.failed:raise ValueError('model bind failed')
                self.binding=dict(profile_id=row['id'],profile_sha256='new-profile',revision=1)
        self.bridge=Bridge();self.selection.motion_bridge=self.bridge
    def test_follow_commits_before_allowing_actions_and_preserves_old_pending(self):
        self.enable_follow();state=self.choose()
        self.assertTrue(state['motion_model_matches']);self.assertEqual(state['bound_model'],'candidate')
        operation=self.call('/operation');self.assertEqual(operation['blockers'],['old_failure'])
        self.call('/operation',{'action':'preview'})
        sent=self.calls[-1][1];self.assertEqual(sent['execution_profile_sha256'],'new-profile')
        self.assertEqual(sent['execution_binding_revision'],1)
    def test_failed_follow_keeps_old_display_and_execution(self):
        self.enable_follow();self.bridge.failed=True;state=self.choose()
        self.assertEqual(state['selected'],'bound');self.assertEqual(state['bound_model'],'bound')
        self.assertIn('model bind failed',state['error'])


def load_tests(loader,tests,pattern):
    suite=unittest.TestSuite()
    for cls in (RoutingTests,BridgeTests):suite.addTests(loader.loadTestsFromTestCase(cls))
    for name in ('test_follow_commits_before_allowing_actions_and_preserves_old_pending','test_failed_follow_keeps_old_display_and_execution'):
        suite.addTest(FollowSwitchTests(name))
    return suite
