import importlib.util
from pathlib import Path
import tempfile
import unittest
import urllib.error
from concurrent.futures import Future
import time
import threading
import json
import sys

spec=importlib.util.spec_from_file_location('cloud_bridge',Path(__file__).parents[1]/'cloud_console_bridge.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class BridgeTests(unittest.TestCase):
    def make(self):
        bridge=module.Bridge(dict(site_url='https://fixture.chatgpt.site',device_secret='fixture'))
        for pool in (bridge.pool,bridge.command_pool,bridge.urgent_pool):
            self.addCleanup(lambda p=pool:p.shutdown(wait=False,cancel_futures=True))
        return bridge

    def cloud_reply(self,bridge,reply,on_open=None):
        import io
        class Opener:
            def open(self,request,timeout):
                if on_open:on_open(json.loads(request.data))
                return io.BytesIO(json.dumps(reply).encode())
        bridge.cloud_opener=Opener()

    def test_cloud_stop_and_heartbeat_do_not_wait_for_a_slow_normal_command(self):
        bridge=self.make();bridge.snapshot_at=time.monotonic()
        release=threading.Event();started=threading.Event();urgent=threading.Event()
        def execute(command):
            if command['kind']=='detect':started.set();release.wait(3)
            else:urgent.set()
            return {'id':command['id'],'ok':True,'data':{}}
        bridge.execute=execute
        self.cloud_reply(bridge,{'server_time':0,'command':{'id':'slow','kind':'detect','payload':{},'expires_at':6500}})
        bridge.exchange();self.assertTrue(started.wait(1))
        try:
            self.cloud_reply(bridge,{'server_time':0,'command':{'id':'stop','kind':'operation','payload':{'action':'stop'},'expires_at':6500}})
            bridge.exchange();self.assertTrue(urgent.wait(1));self.assertFalse(release.is_set())
        finally:release.set()

    def test_results_completed_during_upload_are_not_discarded(self):
        bridge=self.make();bridge.snapshot_at=time.monotonic();bridge.results=[{'id':'first','ok':True}]
        def append_during_upload(data):
            self.assertEqual([r['id'] for r in data['results']],['first'])
            with bridge.results_lock:bridge.results.append({'id':'late','ok':True})
        self.cloud_reply(bridge,{'command':None},append_during_upload)
        bridge.exchange();self.assertEqual([r['id'] for r in bridge.results],['late'])

    def test_live_cloud_connection_does_not_refresh_stale_local_telemetry(self):
        bridge=self.make();bridge.snapshot={'operation':{'busy':False},'models':{'revision':1}}
        bridge.snapshot_at=time.monotonic()-9;sent=[]
        self.cloud_reply(bridge,{'command':None},sent.append);bridge.exchange()
        self.assertGreaterEqual(sent[0]['snapshot']['telemetry_age_ms'],9000)
        self.assertIsNone(sent[0]['snapshot']['operation'])

    def test_remote_operation_read_drops_tokens_and_never_posts(self):
        bridge=self.make();calls=[]
        def request(path,payload=None,timeout=3):
            calls.append((path,payload));return {'busy':False,'token':'private'}
        bridge.local_request=request
        result=bridge.execute({'id':'read','kind':'operation_state','payload':{}})
        self.assertEqual(result['data'],{'busy':False});self.assertEqual(calls,[('/operation',None)])

    def test_removes_local_tokens_recursively(self):
        self.assertEqual(module.clean({'token':'secret','nested':[{'token':'secret','safe':1}]}),{'nested':[{'safe':1}]})

    def test_timeout_after_submission_is_unknown_and_never_replayed(self):
        bridge=self.make();calls=[]
        def request(path,payload=None,timeout=3):
            calls.append((path,payload))
            if payload is None:return {'token':'local-token','operation_id':'old','owner':'original'}
            raise urllib.error.URLError('response lost')
        bridge.local_request=request
        cmd={'id':'command','kind':'operation','remaining_ms':6500,'payload':{'action':'preview','owner':'cloud-owner','request_id':'new','expected_operation_id':'old'}}
        self.assertTrue(bridge.execute(cmd)['unknown']);self.assertTrue(bridge.execute(cmd)['unknown'])
        writes=[data for path,data in calls if data is not None];self.assertEqual(len(writes),1);self.assertEqual(writes[0]['token'],'local-token')

    def test_rejects_changed_task_and_foreign_heartbeat_before_post(self):
        bridge=self.make();calls=[]
        def request(path,payload=None,timeout=3):calls.append(payload);return {'token':'local','operation_id':'actual','owner':'original'}
        bridge.local_request=request
        for action in ('start','heartbeat'):
            result=bridge.execute({'id':action,'kind':'operation','payload':{'action':action,'owner':'foreign','operation_id':'actual','expected_operation_id':'stale'}})
            self.assertFalse(result['ok'])
        self.assertEqual(calls,[None,None])

    def test_model_revision_cannot_be_rebased_silently(self):
        bridge=self.make();bridge.local_request=lambda *a,**k:{'revision':4,'token':'local'}
        result=bridge.execute({'id':'one','kind':'select_model','payload':{'model_id':'candidate','revision':3}})
        self.assertFalse(result['ok'])

    def review_fixture(self):
        project=str(Path(__file__).resolve().parents[2]/'pynq_km1_vision')
        if project not in sys.path:sys.path.insert(0,project)
        from src.board_control_ownership import BoardOwnership
        from src.monitor_operation import Operations
        from src.sorting_manual_review import acknowledge,review_status
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        folder=Path(temporary.name)/'journal';folder.mkdir()
        lease=Path(temporary.name)/'device.lease'
        owner=BoardOwnership(Path(temporary.name)/'owner.db');self.addCleanup(owner.close)
        owner.finish(owner.claim('fixture','failed-request-001'),'needs_review')
        raw=b'{"status":"needs_review","task_id":"fixture-child"}'
        for name in ('pending.json','batch-pending.json','child.json'):(folder/name).write_bytes(raw)
        case=self
        class Backend:
            def check(self):return ['needs_review'] if owner.status()['status']=='blocked' else []
            def review_status(self):return review_status(folder,lease,owner)
            def review(self,**kw):return acknowledge(folder,lease,owner,**kw)
            def cycle(self,*a):case.fail('review must not move')
            def reset(self,*a):case.fail('review must not reset')
            def home(self,*a):case.fail('review must not home')
            def stop(self,*a):case.fail('review must not send STOP')
        op=Operations(Backend(),audit=Path(temporary.name)/'operation.jsonl');self.addCleanup(op.close)
        op.state='finished';op.operation_id='finished-operation'
        op.result={'status':'needs_review'}
        command=dict(id='fixture-review-transport',kind='operation',payload=dict(action='review',owner='fixture-page',
            request_id='review-request-001',expected_operation_id=op.operation_id,review_epoch=owner.epoch,
            review_request_id='failed-request-001',ready=True,review_confirmed=True))
        return op,owner,folder,raw,command

    def test_review_reaches_real_dispatcher_with_original_task_identity(self):
        op,owner,folder,raw,command=self.review_fixture();bridge=self.make();posts=[]
        def request(path,payload=None,timeout=3):
            self.assertEqual(path,'/operation')
            if payload is None:return op.status()
            posts.append(dict(payload));return op.command(payload)
        bridge.local_request=request
        result=bridge.execute(command)
        self.assertTrue(result['ok'],result)
        self.assertEqual(result['data']['result']['status'],'operator_reviewed')
        self.assertEqual(posts[0]['expected_operation_id'],'finished-operation')
        self.assertEqual(owner.status()['status'],'idle')
        self.assertFalse((folder/'pending.json').exists())
        self.assertEqual((folder/'child.json').read_bytes(),raw)
        self.assertEqual(command['payload']['expected_operation_id'],'finished-operation')
        self.assertTrue(bridge.execute(command)['unknown']);self.assertEqual(len(posts),1)

    def test_review_rechecks_identity_at_dispatch_after_bridge_precheck(self):
        op,owner,folder,raw,command=self.review_fixture();bridge=self.make();posts=[]
        def request(path,payload=None,timeout=3):
            if payload is None:return op.status()
            posts.append(dict(payload));op.operation_id='changed-after-precheck'
            return op.command(payload)
        bridge.local_request=request
        result=bridge.execute(command)
        self.assertFalse(result['ok']);self.assertIn('manual_review_context_changed',result['error'])
        self.assertEqual(len(posts),1);self.assertEqual(owner.status()['status'],'blocked')
        self.assertEqual((folder/'pending.json').read_bytes(),raw)

    def test_expired_command_never_reads_or_posts(self):
        bridge=self.make();bridge.local_request=lambda *a,**k:self.fail('expired request reached backend')
        self.assertFalse(bridge.execute({'id':'old','remaining_ms':0,'kind':'operation','payload':{}})['ok'])

    def test_protected_configuration_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'config.bin';value={'device_secret':'fixture','site_url':'https://fixture.chatgpt.site'}
            module.save_config(path,value);self.assertEqual(module.load_config(path),value)
            if module.os.name=='nt':self.assertNotIn(b'fixture',path.read_bytes())

    def test_slow_history_does_not_hold_up_fresh_telemetry(self):
        bridge=self.make();bridge.tasks_future=Future();bridge.last_task_at=time.monotonic()
        def get(path):bridge.stop.set();return None
        bridge.maybe_get=get
        bridge.collect()
        self.assertGreater(bridge.snapshot_at,0)
        self.assertFalse(bridge.tasks_future.done())

    def test_history_error_retains_last_actual_records(self):
        bridge=self.make();bridge.last_tasks={'records':[{'task_id':'actual'}]};bridge.last_task_at=time.monotonic()
        bridge.tasks_future=Future();bridge.tasks_future.set_exception(urllib.error.URLError('temporary'))
        def get(path):bridge.stop.set();return None
        bridge.maybe_get=get
        bridge.collect()
        self.assertEqual(bridge.snapshot['tasks'],{'records':[{'task_id':'actual'}]})

    def test_candidate_points_and_operation_timing_survive_public_snapshot(self):
        bridge=self.make();bridge.last_task_at=time.monotonic()
        replies={'/operation':dict(operation_id='one',per_instance_widths_supported=True,result={'status':'needs_review'},
                    progress={'operation_id':'one','phase':'connecting'}),
                 '/live-result':dict(frame_id=2,image_size=[640,480],overlay_base64='YQ==',
                    display_grasp_points=[dict(pixel=[100,200])],
                    display_geometry_timing={'elapsed_ms':1.5},instances=[])}
        def get(path):
            bridge.stop.set()
            return replies.get(path)
        bridge.maybe_get=get;bridge.collect()
        public=bridge.snapshot
        self.assertEqual(public['vision']['display_grasp_points'][0]['pixel'],[100,200])
        self.assertEqual(public['operation']['result']['status'],'needs_review')
        self.assertTrue(public['operation']['per_instance_widths_supported'])
        key=public['vision']['frame_key']
        self.assertEqual(bridge.display_frame(key)['base64'],'YQ==')
        self.assertIsNone(bridge.display_frame('wrong-image'))
        # Repeated upstream image must not reset its original freshness clock.
        stamp=bridge.display_frames[key][1];bridge.collect()
        self.assertEqual(bridge.display_frames[key][1],stamp)


if __name__=='__main__':unittest.main()
