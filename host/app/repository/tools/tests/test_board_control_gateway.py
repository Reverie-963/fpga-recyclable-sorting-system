import json
from pathlib import Path
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import unittest
from contextlib import contextmanager

sys.path.insert(0,str(Path(__file__).parents[1]))
sys.path.insert(0,str(Path(__file__).parents[2]/'pynq_km1_vision'))
from board_control_gateway import make_gateway
from src.board_control_gateway_client import BoardPreviewClient


class RPC:
    def __init__(self):self.calls=[];self.fail=False
    def request(self,path,data=None):
        self.calls.append((path,data))
        if self.fail:raise ValueError('board_control_active')
        if path=='/tasks':return dict(task_id='test-task',lease={'token':'not-shown'})
        if path=='/health':return dict(compute_location='pynq_ps',execution_enabled=False)
        if path=='/tasks/test-task':return dict(status='finished',report=dict(status='no_pickable_instance',compute_location='pynq_ps',physical_motion_attempted=False))
        raise AssertionError('unexpected path '+path)


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        class Upstream(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_GET(self):
                body=json.dumps(dict(settings={'settings_sha256':'snapshot'},generation=3,model_hashes={'weights':'hash'},execution={'device':'cuda:0'})).encode()
                self.send_response(200);self.end_headers();self.wfile.write(body)
            def do_POST(self):raise AssertionError('PC must not plan or run inference in gateway tests')
        self.upstream=ThreadingHTTPServer(('127.0.0.1',0),Upstream)
        self.rpc=RPC();self.idle_checks=[]
        self.gateway=make_gateway(self.rpc,'t'*40,'http://127.0.0.1:'+str(self.upstream.server_port),0,
            idle_check=lambda:self.idle_checks.append(True),timing_provider=lambda:{'max_age_ms':6000},
            motion_options_provider=lambda:{'stable_frames':1,'destination_empty_check_enabled':False})
        self.threads=[]
        for server in (self.upstream,self.gateway):
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();self.threads.append(thread)
        self.addCleanup(self.close)
        root=Path(self.temp.name);(root/'token.txt').write_text('t'*40)
        self.config=root/'active.json'
        self.record=dict(mode='preview_only',url='http://127.0.0.1:'+str(self.gateway.server_port),token_file='token.txt')
        self.config.write_text(json.dumps(self.record))
        self.client=BoardPreviewClient(self.config,root)
    def close(self):
        for server in (self.gateway,self.upstream):server.shutdown();server.server_close()
        for thread in self.threads:thread.join(2)
    def test_plan_uses_same_model_snapshot_and_only_board_task_protocol(self):
        result=self.client.plan({})
        self.assertEqual(result['compute_location'],'pynq_ps')
        self.assertEqual([path for path,data in self.rpc.calls],['/tasks','/tasks/test-task'])
        request=self.rpc.calls[0][1]
        self.assertEqual(request['expected']['settings']['settings_sha256'],'snapshot')
        self.assertEqual(request['expected']['observation_timing'],{'max_age_ms':6000})
        self.assertEqual(request['expected']['motion_options'],{'stable_frames':1,'destination_empty_check_enabled':False})
        self.assertEqual(self.idle_checks,[True])
    def test_rejection_has_no_pc_or_board_retry(self):
        self.rpc.fail=True
        with self.assertRaisesRegex(ValueError,'board_control_active'):self.client.plan({})
        self.assertEqual([p for p,d in self.rpc.calls],['/tasks'])
    def test_height_is_frozen_in_task_without_changing_detector_identity(self):
        self.client.plan({'material':'METAL','object_height_mm':60})
        request=self.rpc.calls[0][1]
        self.assertEqual(request['expected']['object_height_mm'],60)
        self.assertEqual(request['expected']['settings'],{'settings_sha256':'snapshot'})
        self.assertEqual(set(request),{'owner','request_id','material','expected'})
        self.rpc.calls.clear()
        for value in (True,0,121,'60'):
            with self.assertRaises(ValueError):self.client.plan({'object_height_mm':value})
        self.assertFalse(self.rpc.calls)
    def test_unknown_or_motion_arguments_rejected_before_task_admission(self):
        with self.assertRaises(ValueError):self.client.plan({'execute':True})
        with self.assertRaises(ValueError):self.client.request('/execute',{})
        self.assertFalse(self.rpc.calls)
    def test_remote_redirect_and_token_path_escape_rejected(self):
        self.config.write_text(json.dumps(dict(self.record,url='http://192.168.2.99:18813')))
        with self.assertRaises(ValueError):BoardPreviewClient(self.config,self.temp.name)

    def test_existing_client_revalidates_published_target_without_retry(self):
        self.config.write_text(json.dumps(dict(self.record,mode='execute')))
        with self.assertRaisesRegex(ValueError,'preview_only'):self.client.health()
        self.assertFalse(self.rpc.calls)

    def test_transition_lease_is_held_across_board_admission_and_result(self):
        held=[]
        class Owner:
            def check(self):
                if held!=['held']:raise ValueError('not owned')
        @contextmanager
        def lease():
            held.append('held')
            try:yield Owner()
            finally:held.clear()
        original=self.rpc.request
        def request(path,data=None):
            self.assertEqual(held,['held'])
            return original(path,data)
        self.rpc.request=request
        extra=make_gateway(self.rpc,'t'*40,'http://127.0.0.1:'+str(self.upstream.server_port),0,lease_factory=lease)
        thread=threading.Thread(target=extra.serve_forever,daemon=True);thread.start()
        try:
            config=Path(self.temp.name)/'lease-config.json'
            config.write_text(json.dumps(dict(self.record,url='http://127.0.0.1:'+str(extra.server_port))))
            result=BoardPreviewClient(config,self.temp.name).plan({})
            self.assertEqual(result['compute_location'],'pynq_ps')
            self.assertFalse(held)
        finally:extra.shutdown();extra.server_close();thread.join(2)
        self.config.write_text(json.dumps(dict(self.record,token_file='../outside.txt')))
        with self.assertRaises(ValueError):BoardPreviewClient(self.config,self.temp.name)


if __name__=='__main__':unittest.main()
