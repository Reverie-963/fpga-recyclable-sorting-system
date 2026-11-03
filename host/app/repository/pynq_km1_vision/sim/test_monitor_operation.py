"""Dashboard requests must never bypass a single supervised physical attempt."""
import json
import hashlib
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import urllib.request
import urllib.error
from src.monitor_operation import Operations, StationBackend,scoped_request,OPERATION_ROUTES
from src.sorting_monitor_server import make_server


class Backend:
    def __init__(self, reports=None):
        self.reports = iter(reports or [])
        self.calls, self.stops = [], 0
        self.entered, self.release = threading.Event(), threading.Event()
        self.block = False
        self.blockers = []

    def check(self):
        return self.blockers

    def cycle(self, material, execute):
        self.calls.append((material, execute))
        self.entered.set()
        if self.block:
            self.release.wait(3)
        if execute:
            return dict(status='completed_visual_evidence', task_id='real', choice={'label': 'paper'})
        return next(self.reports, dict(status='preview_only'))

    def stop(self):
        self.stops += 1

    def reset(self,confirmed=False):
        self.reset_confirmed=confirmed
        return dict(status='state_reset',physical_motion=False,original_reports_preserved=True)

    def home(self):
        self.calls.append(('HOME',True))
        return dict(status='neutral_return_command_completed',task_id='home',physical_motion_attempted=True)


class OperationTests(unittest.TestCase):
    def setUp(self):
        self.sessions = []

    def tearDown(self):
        for session in self.sessions:
            session.backend.release.set()
            session.close()

    def session(self, backend=None, **kwargs):
        session = Operations(backend or Backend(), poll_s=.01, **kwargs)
        self.sessions.append(session)
        return session

    def command(self, s, action, **values):
        body = dict(token=s.token, owner='page-owner', request_id='request-one', ready=True, action=action)
        body.update(values)
        return s.command(body)

    def wait(self, predicate):
        deadline = time.monotonic()+2
        while not predicate():
            if time.monotonic() > deadline:
                self.fail('condition timeout')
            time.sleep(.005)

    def test_preview_never_moves_and_replay_is_rejected_after_finish(self):
        s = self.session()
        self.command(s, 'preview')
        self.wait(lambda: s.state == 'finished')
        self.assertEqual(s.backend.calls, [('AUTO', False)])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.command(s, 'start')

    def test_reset_returns_to_idle_without_rearming_or_replaying_motion(self):
        s=self.session();self.command(s,'preview');self.wait(lambda:s.state=='finished')
        old_calls=list(s.backend.calls)
        self.command(s,'reset',request_id='reset-unique',reset_confirmed=False)
        self.assertEqual(s.state,'idle');self.assertIsNone(s.owner)
        self.assertEqual(s.result['status'],'state_reset');self.assertEqual(s.backend.calls,old_calls)
        with self.assertRaisesRegex(ValueError,'duplicate'):
            self.command(s,'preview')

    def test_reset_is_rejected_while_worker_is_active(self):
        s=self.session();s.backend.block=True;self.command(s,'start')
        self.assertTrue(s.backend.entered.wait(1))
        with self.assertRaisesRegex(ValueError,'busy'):
            self.command(s,'reset',request_id='reset-busy',reset_confirmed=True)
        self.assertEqual(s.backend.calls,[('AUTO',True)])

    def test_home_is_an_explicit_action_and_preview_or_reset_do_not_call_it(self):
        s=self.session()
        with self.assertRaisesRegex(ValueError,'neutral_return_scene_confirmation'):
            self.command(s,'home',request_id='home-no-confirmation',ready=False)
        self.assertEqual(s.backend.calls,[])
        self.command(s,'preview');self.wait(lambda:s.state=='finished')
        self.assertEqual(s.backend.calls,[('AUTO',False)])
        self.command(s,'home',request_id='home-explicit',ready=False,home_confirmed=True)
        self.wait(lambda:s.state=='finished')
        self.assertEqual(s.backend.calls,[('AUTO',False),('HOME',True)])
        self.assertEqual(s.result['status'],'neutral_return_command_completed')

    def test_jaw_calibration_is_single_physical_action_and_never_sorting(self):
        backend=Backend();calls=[]
        backend.jaw_calibrate=lambda:calls.append('jaw') or dict(status='jaw_calibration_command_completed',
            physical_motion_attempted=True,target_pwm=[1500]*5+[1100],jaw_calibration={'measurement_required':True})
        s=self.session(backend)
        with self.assertRaisesRegex(ValueError,'scene_ready'):
            self.command(s,'jaw_calibrate',ready=False)
        self.command(s,'jaw_calibrate');self.wait(lambda:s.state=='finished')
        self.assertEqual(calls,['jaw']);self.assertEqual(backend.calls,[])
        self.assertFalse(s.result['preview']);self.assertEqual(s.result['target_pwm'][-1],1100)
        with self.assertRaisesRegex(ValueError,'duplicate'):self.command(s,'jaw_calibrate')

    def test_old_backend_and_blocked_station_cannot_calibrate(self):
        s=self.session()
        with self.assertRaisesRegex(ValueError,'not_supported'):self.command(s,'jaw_calibrate')
        s.backend.jaw_calibrate=lambda:self.fail('must not reach hardware')
        s.backend.blockers=['operator_stop_requested']
        with self.assertRaisesRegex(ValueError,'station_not_ready'):self.command(s,'jaw_calibrate')

    def test_executor_exception_has_visible_reason_without_claiming_no_motion(self):
        s=self.session()
        with patch.object(s.backend,'cycle',side_effect=PermissionError('[WinError 32] task file in use')) as cycle:
            self.command(s,'start');self.wait(lambda:s.state=='rejected')
        result=s.status()['result']
        self.assertIn('WinError 32',result['reason']);self.assertEqual(result['status'],'rejected')
        self.assertNotIn('physical_motion_attempted',result)
        self.assertEqual(cycle.call_count,1)

    def test_unified_backend_dispatches_bound_batch_and_exposes_batch_result(self):
        backend=StationBackend.__new__(StationBackend)
        backend.binding=Path('bound-station.json');backend.unified=True;backend._verify=lambda:None
        with patch('src.sorting_unified_run.run',return_value=dict(execution_blockers=[])) as run:
            self.assertEqual(backend.check(),[])
            run.assert_called_once_with(backend.binding,'preflight')
        with patch('src.sorting_unified_run.run',return_value=dict(status='completed',batch_id='batch',
                   completed_count=2,remaining=[],physical_motion_attempted=True)) as run:
            backend.cycle('AUTO',True)
            run.assert_called_once_with(backend.binding,'execute',ready=True,material='AUTO')
        fake=Backend();fake.unified=True
        fake.cycle=lambda material,execute:dict(status='completed',batch_id='batch',completed_count=2,
                                               remaining=[],physical_motion_attempted=True)
        s=self.session(fake);self.command(s,'start');self.wait(lambda:s.state=='finished')
        self.assertEqual(s.status()['control_version'],3)
        self.assertTrue(s.status()['batch_mode']);self.assertEqual(s.result['completed_count'],2)
        self.assertEqual(s.result['remaining_count'],0)

    def test_ready_token_material_and_existing_protections(self):
        s = self.session()
        for change in ({'ready': False}, {'ready': 'true'}, {'material': 'UNKNOWN'}, {'token': 'old'}):
            body = dict(token=s.token, owner='page-owner', request_id='request-one',
                        action='start', ready=True)
            body.update(change)
            with self.assertRaises(ValueError):
                s.command(body)
        s.backend.blockers = ['existing_physical_trial_unresolved']
        with self.assertRaisesRegex(ValueError, 'station_not_ready'):
            self.command(s, 'arm')
        self.assertEqual(s.backend.calls, [])

    def test_standby_must_see_empty_before_new_target_then_one_attempt(self):
        s = self.session(Backend([dict(status='preview_only'), dict(status='empty_scene'),
                                 dict(status='preview_only')]))
        self.command(s, 'arm')
        self.wait(lambda: s.state == 'finished')
        self.assertEqual([execute for _, execute in s.backend.calls], [False, False, False, True])
        self.assertEqual(s.result['task_id'], 'real')
        time.sleep(.04)
        self.assertEqual(len(s.backend.calls), 4)

    def test_material_standby_can_wait_with_other_materials_present(self):
        s=self.session(Backend([dict(status='no_matching_material'),dict(status='preview_only')]))
        self.command(s,'arm',material='PAPER')
        self.wait(lambda:s.state=='finished')
        self.assertEqual(s.backend.calls,[('PAPER',False),('PAPER',False),('PAPER',True)])

    def test_busy_rejects_other_requests_including_preview(self):
        backend = Backend(); backend.block = True
        s = self.session(backend)
        self.command(s, 'preview')
        self.assertTrue(backend.entered.wait(1))
        with self.assertRaisesRegex(ValueError, 'busy'):
            self.command(s, 'start', **{'request_id': 'request-two'})

    def test_cancel_inflight_observation_never_executes_and_does_not_latch_global_stop(self):
        backend = Backend(); backend.block = True
        s = self.session(backend)
        self.command(s, 'arm')
        self.assertTrue(backend.entered.wait(1))
        self.command(s, 'cancel')
        backend.release.set()
        self.wait(lambda: s.state == 'stopped')
        self.assertFalse(any(execute for _, execute in backend.calls))
        self.assertEqual(backend.stops, 0)

    def test_disconnect_during_execution_persists_stop_and_never_retries(self):
        backend = Backend(); backend.block = True
        s = self.session(backend, lease_s=.03)
        self.command(s, 'start')
        self.assertTrue(backend.entered.wait(1))
        self.wait(lambda: backend.stops == 1)
        self.assertEqual(s.reason, 'page_disconnected')
        backend.release.set()
        self.wait(lambda: s.state == 'stopped')
        self.assertEqual(len(backend.calls), 1)

    def test_standby_timeout_never_moves(self):
        s = self.session(Backend([dict(status='empty_scene')]*100), standby_s=.03)
        self.command(s, 'arm')
        self.wait(lambda: s.state == 'stopped')
        self.assertEqual(s.reason, 'standby_timeout')
        self.assertFalse(any(execute for _, execute in s.backend.calls))

    def test_rejected_observation_or_blockers_disarm_without_motion(self):
        for report in (dict(status='rejected'), dict(status='preview_only', execution_blockers=['occupied'])):
            s = self.session(Backend([report]))
            self.command(s, 'arm')
            self.wait(lambda: s.state == 'rejected')
            self.assertEqual(s.backend.calls, [('AUTO', False)])

    def test_foreign_tab_cannot_renew_or_cancel_owner_lease(self):
        backend = Backend(); backend.block = True
        s = self.session(backend)
        self.command(s, 'arm')
        for action in ('heartbeat', 'cancel'):
            with self.assertRaisesRegex(ValueError, 'not_operation_owner'):
                s.command(dict(token=s.token, owner='other-page', action=action,
                    operation_id=s.operation_id))
        self.command(s, 'heartbeat', operation_id=s.operation_id)
        self.command(s, 'stop')
        self.assertEqual(backend.stops, 1)

    def test_close_stops_running_motion_and_restart_does_not_arm(self):
        backend = Backend(); backend.block = True
        s = self.session(backend)
        self.command(s, 'start')
        self.assertTrue(backend.entered.wait(1))
        backend.release.set()
        s.close()
        fresh = self.session()
        self.assertEqual(fresh.state, 'idle')
        self.assertEqual(fresh.backend.calls, [])
        self.assertNotEqual(fresh.token, s.token)


class OperationHttpTests(unittest.TestCase):
    def test_all_and_three_material_interfaces_share_ownership_and_fix_material(self):
        backend=Backend();operations=Operations(backend)
        server=make_server(0,'http://127.0.0.1:1',operations=operations)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base='http://127.0.0.1:'+str(server.server_port)
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            for index,(route,material) in enumerate(OPERATION_ROUTES.items()):
                with opener.open(base+route) as reply:state=json.load(reply)
                self.assertEqual(state['token'],operations.token)
                if material:self.assertEqual(state['interface_material'],material)
                body=dict(token=operations.token,action='preview',owner='test-page',request_id='scope-test-'+str(index))
                with opener.open(urllib.request.Request(base+route,data=json.dumps(body).encode(),
                        headers={'Content-Type':'application/json'})) as reply:self.assertEqual(reply.status,200)
                operations.worker.join(2)
                self.assertEqual(backend.calls[-1],(material or 'AUTO',False))
            body=dict(token=operations.token,action='start',owner='test-page',request_id='wrong-scope',ready=True,material='METAL')
            previous=len(backend.calls)
            with self.assertRaises(urllib.error.HTTPError) as raised:
                opener.open(urllib.request.Request(base+'/operation/paper',data=json.dumps(body).encode(),
                            headers={'Content-Type':'application/json'}))
            self.assertEqual(raised.exception.code,409);raised.exception.close()
            self.assertEqual(len(backend.calls),previous)
        finally:server.shutdown();thread.join();server.server_close()

    def test_origin_content_type_token_and_real_endpoint(self):
        backend = Backend()
        operations = Operations(backend)
        server = make_server(0, 'http://127.0.0.1:1', operations=operations)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        base = 'http://127.0.0.1:'+str(server.server_port)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(base+'/operation') as reply:
                initial = json.load(reply)
                self.assertEqual(initial['control_version'], 1)
                token = initial['token']
            body = dict(token=token, action='preview', owner='test-page', request_id='test-request')
            for headers, expected in (({'Content-Type': 'application/json', 'Origin': 'https://evil.test'}, 403),
                                      ({'Content-Type': 'text/plain'}, 409)):
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    opener.open(urllib.request.Request(base+'/operation', data=json.dumps(body).encode(), headers=headers))
                self.assertEqual(raised.exception.code, expected); raised.exception.close()
            with opener.open(urllib.request.Request(base+'/operation', data=json.dumps(body).encode(),
                    headers={'Content-Type': 'application/json'})) as reply:
                self.assertEqual(reply.status, 200)
            operations.worker.join(1)
            self.assertEqual(backend.calls, [('AUTO', False)])
        finally:
            server.shutdown(); thread.join(); server.server_close()


class BindingStopTests(unittest.TestCase):
    def test_retired_pointer_or_changed_binding_blocks_but_stop_path_remains_usable(self):
        with tempfile.TemporaryDirectory() as folder:
            backend = StationBackend.__new__(StationBackend)
            backend.pointer, backend.binding = Path(folder)/'active.json', Path(folder)/'binding.json'
            for path in (backend.pointer, backend.binding):
                path.write_text('{}')
            backend.pointer_sha = backend.sha = hashlib.sha256(b'{}').hexdigest()
            backend.stops = [Path(folder)/'STOP']
            backend._verify()
            backend.pointer.write_text('{"path":"replacement"}')
            with self.assertRaisesRegex(ValueError, 'station_binding_changed'):
                backend._verify()
            backend.stop()
            self.assertTrue(backend.stops[0].exists())
            previous = backend.stops[0].read_text()
            backend.stop()
            self.assertTrue(backend.stops[0].read_text().startswith(previous))


if __name__ == '__main__':
    unittest.main()
