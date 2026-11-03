import tempfile
from pathlib import Path
import unittest
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from types import SimpleNamespace

from src.board_control_ownership import BoardOwnership
from src.board_station_host import Authority,guard_connection_type,translated_arm,ReadinessView,renew_operation_heartbeat,StationAuthorization,initial_decision_thresholds


class SelectedModelBootstrapTests(unittest.TestCase):
    def test_bootstrap_uses_registered_profile_and_restores_context(self):
        from src.execution_model_context import current_profile,model_context
        from src.sorting_prompted_motion import calibrated_settings
        profile={'settings':{'confidence':.10},'receipt':{'original_evidence':'unchanged'}}
        options={'decision_thresholds':{'min_scene_iou':.85}}
        self.assertIsNone(current_profile())
        value=initial_decision_thresholds(options,lambda:model_context(profile),calibrated_settings)
        self.assertEqual(value,{'min_scene_iou':.85,'min_confidence':.10})
        self.assertIsNone(current_profile())
        self.assertEqual(profile['receipt'],{'original_evidence':'unchanged'})

    def test_without_registered_profile_keeps_original_receipt_failure(self):
        from contextlib import nullcontext
        def invalid_receipt():raise ValueError('prompted_trial_implementation_changed')
        with self.assertRaisesRegex(ValueError,'implementation_changed'):
            initial_decision_thresholds({'decision_thresholds':{}},nullcontext,invalid_receipt)


class StationAuthenticationTests(unittest.TestCase):
    def test_instance_opener_authenticates_snapshot_get_without_leaking_to_other_origins(self):
        token='private-fixture-token';seen=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*a):pass
            def do_GET(self):
                seen.append((self.path,self.headers.get('Authorization')))
                self.send_response(200 if self.headers.get('Authorization')=='Bearer '+token else 403)
                self.send_header('Content-Length','4');self.end_headers();self.wfile.write(b'PNG!')
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            url='http://127.0.0.1:'+str(server.server_port)
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            auth=StationAuthorization(token,url);opener.add_handler(auth)
            with opener.open(url+'/snapshot.png?frame_id=12',timeout=2) as reply:self.assertEqual(reply.read(),b'PNG!')
            self.assertEqual(seen,[('/snapshot.png?frame_id=12','Bearer '+token)])
            request=urllib.request.Request('http://other.invalid/snapshot.png')
            self.assertIs(auth.http_request(request),request);self.assertIsNone(request.get_header('Authorization'))
        finally:server.shutdown();server.server_close();thread.join(2)


class ReadinessSnapshotTests(unittest.TestCase):
    def test_display_uses_snapshot_while_each_admission_still_runs_fresh_check(self):
        view=ReadinessView();calls=[]
        def expensive():calls.append(True);return ['original_guard']
        self.assertEqual(view.check(expensive),['original_guard'])
        with view.display():
            self.assertEqual(view.check(lambda:(_ for _ in ()).throw(AssertionError('display blocked'))),['original_guard'])
        view.check(expensive);self.assertEqual(len(calls),2)

    def test_display_context_does_not_skip_another_threads_actual_admission(self):
        view=ReadinessView();called=[]
        with view.display():
            t=threading.Thread(target=lambda:view.check(lambda:called.append(True) or []))
            t.start();t.join(2)
        self.assertEqual(called,[True])


class BoardActuatorBoundaryTests(unittest.TestCase):
    def test_heartbeat_ack_never_calls_readiness_or_status_and_renews_both_leases(self):
        self.authority.begin('web','heartbeat-request')
        operation=SimpleNamespace(lock=threading.RLock(),token='page-token',owner='web',
            operation_id='heartbeat-request',last_heartbeat=0,
            status=lambda:(_ for _ in ()).throw(AssertionError('heavy status called')))
        data=dict(token='page-token',owner='web',operation_id='heartbeat-request')
        self.now=11
        self.assertTrue(renew_operation_heartbeat(operation,data,self.authority,lambda:self.now)['heartbeat_accepted'])
        self.assertEqual(operation.last_heartbeat,11)
        self.now=13;self.authority.check()
        with self.assertRaisesRegex(ValueError,'not_operation_owner'):
            renew_operation_heartbeat(operation,dict(data,owner='different-client'),self.authority)
        self.owner.stop('test stop')
        with self.assertRaisesRegex(ValueError,'blocked'):
            renew_operation_heartbeat(operation,data,self.authority)

    def test_transport_mapping_keeps_qualified_configuration_and_rejects_identity_changes(self):
        arm=dict(port='COM9',usb_identity=dict(vid=0x1a86,pid=0x7523,serial_number=None),move_ms=5000)
        mapping=dict(qualified_port='COM9',arm_device='/dev/ttyUSB0',usb_identity=dict(arm['usb_identity']))
        runtime=translated_arm(arm,mapping)
        self.assertEqual(arm['port'],'COM9');self.assertEqual(runtime['port'],'/dev/ttyUSB0')
        self.assertEqual(runtime['move_ms'],5000)
        runtime['usb_identity']['pid']=0x7522
        self.assertEqual(arm['usb_identity']['pid'],0x7523)
        with self.assertRaisesRegex(ValueError,'mapping_changed'):
            translated_arm(arm,dict(mapping,qualified_port='COM10'))
        with self.assertRaisesRegex(ValueError,'mapping_changed'):
            translated_arm(arm,dict(mapping,usb_identity=dict(vid=0x1a86,pid=0x7522)))

    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.now=0
        self.owner=BoardOwnership(Path(temp.name)/'ownership.db',clock=lambda:self.now)
        self.addCleanup(self.owner.close)
        self.authority=Authority(self.owner)
        self.writes=[];outer=self
        class Connection:
            def open(self):return 'opened'
            def send(self,data,duration,guard=None):
                if guard:guard()
                outer.writes.append(data)
            def request(self,line):outer.writes.append(line)
        self.connection=guard_connection_type(Connection,self.authority)()

    def test_no_serial_open_or_write_without_board_lease(self):
        for action in (self.connection.open,lambda:self.connection.send(b'move',100),
                       lambda:self.connection.request('ARM')):
            with self.assertRaisesRegex(ValueError,'ownership_required'):action()
        self.assertFalse(self.writes)

    def test_repeated_live_checks_do_not_rewrite_state_but_expiry_is_durable(self):
        self.authority.begin('web','request-no-write')
        changes=self.owner.db.total_changes
        for _ in range(20):self.authority.check()
        self.assertEqual(self.owner.db.total_changes,changes)
        self.now=13
        with self.assertRaisesRegex(ValueError,'blocked'):self.authority.check()
        self.assertGreater(self.owner.db.total_changes,changes)
        self.assertEqual(self.owner.status()['reason'],'board_owner_heartbeat_lost')

    def test_expiry_during_geometric_guard_prevents_uart_write(self):
        self.authority.begin('web','request1')
        self.connection.open()
        def late_guard():self.now=13
        with self.assertRaisesRegex(ValueError,'board_control_blocked'):
            self.connection.send(b'move',100,late_guard)
        self.assertFalse(self.writes)
        self.authority.finish(failed=True)
        self.assertEqual(self.owner.status()['status'],'blocked')

    def test_stop_after_observation_cannot_be_cleared_by_finished_worker(self):
        self.authority.begin('cli','request2')
        self.owner.stop('operator')
        with self.assertRaisesRegex(ValueError,'blocked'):self.connection.send(b'move',100)
        self.authority.finish({'status':'completed_visual_evidence'})
        self.assertEqual(self.owner.status()['status'],'blocked')
        self.assertFalse(self.writes)

    def test_guard_failure_after_serial_open_requires_review(self):
        self.authority.begin('web','request3');self.connection.open()
        def bad_guard():raise ValueError('marker moved')
        with self.assertRaisesRegex(ValueError,'marker moved'):self.connection.send(b'move',100,bad_guard)
        self.authority.finish(failed=True)
        self.assertEqual(self.owner.status()['status'],'blocked')
        self.assertFalse(self.writes)

    def test_readonly_rejection_releases_owner_without_mechanical_success(self):
        self.authority.begin('web','request4')
        self.authority.finish({'status':'blocked'})
        self.assertEqual(self.owner.status()['status'],'idle')
        with self.assertRaisesRegex(ValueError,'already_used'):self.authority.begin('web','request4')

    def test_success_keeps_original_write_guard_and_releases_verified_result_only(self):
        self.authority.begin('web','request5');self.connection.open()
        checks=[]
        self.connection.send(b'original-bytes',100,lambda:checks.append('original guard'))
        self.assertEqual(checks,['original guard']);self.assertEqual(self.writes,[b'original-bytes'])
        self.authority.finish({'status':'completed_visual_evidence'})
        self.assertEqual(self.owner.status()['status'],'idle')

    def test_source_change_prevents_open_and_new_stale_frame_prevents_write(self):
        self.authority.begin('web','request6')
        self.authority.source_check=lambda:(_ for _ in ()).throw(ValueError('source changed'))
        with self.assertRaisesRegex(ValueError,'source changed'):self.connection.open()
        self.assertFalse(self.authority.port_attempted)
        self.authority.source_check=lambda:None
        self.authority.max_frame_age_ms=6000;self.authority.clock=lambda:10
        self.authority.last_capture_ms=0
        with self.assertRaisesRegex(ValueError,'prewrite_capture_expired'):self.connection.send(b'move',100)
        self.assertFalse(self.writes)


if __name__=='__main__':unittest.main()
