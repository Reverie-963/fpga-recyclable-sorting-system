"""Telemetry must never promote plans, echoes or simulations to physical success."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
import urllib.request
import urllib.error
from src.monitor_control import ControlTelemetry, project_record, geometry_summary
from src.sorting_monitor_server import make_server


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data),encoding='utf-8')
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ControlRecordTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.trial=self.root/'can'
        self.command='{#005P1300T2000!}'
        self.plan=dict(task_id='can',class_from_yoloe='metal',bin_id='bin_metal',pick_pixel=[2,3],
            grip_xyz_mm=[1,2,10],model_version='model',
            source_geometry_jobs=[dict(m00=4,m10=8,m01=12,m20=20,m02=40,m11=24,
                min_col=1,max_col=3,min_row=2,max_row=4,status=1,job_id=3)],
            statistics_provenance=dict(backend='pl_http',each_result_cpu_compared=True),
            stages=[dict(name='can_close_initial',steps=[dict(step='close',command=self.command,duration_ms=2000)])])
        self.plan_hash=save(self.trial/'plan.json',self.plan)
        self.record=dict(task_id='can',stage='can_close_initial',plan_sha256=self.plan_hash,
            status='sent_pending_observation',serial_opened=True,serial_closed=True,finished_at='done',
            steps=[dict(step='close',command=self.command,duration_ms=2000,attempted=True,
                        bytes_written=len(self.command),echo_matched=True)])
        self.path=self.trial/'journal/close.json'
        save(self.path,self.record)

    def tearDown(self):self.temp.cleanup()

    def result(self):return project_record(self.path,self.record)

    def test_real_fpga_moments_and_bound_coordinates(self):
        d=self.result()
        self.assertEqual(d['geometry']['centroid_px'],[2,3])
        self.assertEqual(d['geometry']['backend'],'pl_http')
        self.assertTrue(d['geometry']['cpu_compared'])
        self.assertIsNone(d['geometry']['axis_deg'])
        self.assertEqual(d['pick_pixel'],[2,3])

    def test_complete_echo_is_not_grasp_or_placement_success(self):
        d=self.result()
        self.assertEqual(d['echoed_commands'],1)
        self.assertTrue(d['close_command_sent'])
        self.assertFalse(d['grasp_success_verified'])
        self.assertFalse(d['physical_success_verified'])
        self.assertEqual(d['visual_verification']['state'],'not_confirmed')

    def test_attempt_before_uart_write_is_not_counted_as_sent(self):
        entry=self.record['steps'][0]
        entry.pop('bytes_written')
        entry['echo_matched']=False
        d=self.result()
        self.assertEqual(d['written_commands'],0)
        self.assertFalse(d['close_command_sent'])
        self.assertEqual(d['commands'][0]['state'],'intent_only')

    def test_partial_write_and_wrong_command_never_get_echo_credit(self):
        self.record['steps'][0]['bytes_written']=2
        d=self.result()
        self.assertEqual(d['commands'][0]['state'],'partial_write')
        self.assertEqual(d['echoed_commands'],0)
        self.record['steps'][0]['command']='{#005P1400T2000!}'
        self.record['steps'][0]['bytes_written']=len(self.command)
        self.assertEqual(self.result()['commands'][0]['state'],'identity_mismatch')

    def test_wrong_plan_hash_hides_geometry_and_grasp(self):
        self.record['plan_sha256']='wrong'
        d=self.result()
        self.assertFalse(d['plan_bound'])
        self.assertIsNone(d['pick_pixel'])
        self.assertIsNone(d['geometry'])

    def test_simulation_is_never_counted_as_real_writes_or_grasps(self):
        self.record['mode']='simulation'
        d=self.result()
        self.assertEqual(d['record_kind'],'simulation')
        self.assertEqual(d['written_commands'],0)
        self.assertFalse(d['close_command_sent'])

    def test_hashed_predecessor_supplies_original_point_not_unrelated_trial(self):
        second=self.root/'lift'
        parent_execution=hashlib.sha256(self.path.read_bytes()).hexdigest()
        plan=dict(task_id='lift',model_version='model',prior_plan_path='can/plan.json',
            prior_plan_sha256=self.plan_hash,prior_execution_path='can/journal/close.json',
            prior_execution_sha256=parent_execution,
            stages=[dict(name='lift',steps=[dict(step='lift',command=self.command,duration_ms=2000)])])
        ph=save(second/'plan.json',plan)
        record=copy.deepcopy(self.record);record.update(task_id='lift',stage='lift',plan_sha256=ph)
        record['steps'][0]['step']='lift'
        path=second/'journal/lift.json';save(path,record)
        d=project_record(path,record)
        self.assertEqual(d['linked_segments'],2)
        self.assertEqual(d['grasp_source_task_id'],'can')
        self.assertEqual(d['echoed_commands'],2)
        plan['prior_execution_sha256']='wrong';record['plan_sha256']=save(second/'plan.json',plan)
        d=project_record(path,record)
        self.assertEqual(d['linked_segments'],1)
        self.assertIsNone(d['pick_pixel'])

    def test_feedback_requires_exact_execution_identity(self):
        value=dict(task_id='can',plan_sha256=self.plan_hash,execution_sha256='wrong',
                   text='held',can_left_table=True,no_drop_reported=True)
        save(self.path.with_suffix('.observation.json'),value)
        self.assertIsNone(self.result()['operator_feedback'])
        value['execution_sha256']=hashlib.sha256(self.path.read_bytes()).hexdigest()
        save(self.path.with_suffix('.observation.json'),value)
        d=self.result()
        self.assertTrue(d['operator_lift_reported'])
        self.assertFalse(d['physical_success_verified'])

    def test_visual_pass_is_explicit_and_wrong_task_review_is_rejected(self):
        review=dict(status='supervised_trial_visual_evidence_pass',steps=1,
            visual_evidence=[dict(source=dict(erased=4),bin=dict(beyond=10))])
        save(self.trial/'final-review.json',review)
        d=self.result()
        self.assertEqual(d['visual_verification']['state'],'passed')
        self.assertEqual(d['visual_verification']['correct_bin_gain_min'],10)
        self.assertFalse(d['physical_success_verified'])
        review['task_id']='unrelated';save(self.trial/'final-review.json',review)
        self.assertEqual(self.result()['visual_verification']['state'],'not_confirmed')

    def test_broken_newer_record_does_not_silently_show_old_success(self):
        reader=ControlTelemetry([self.root],refresh_s=0)
        self.assertTrue(reader.status(dict(model_sha256='model'))['available'])
        bad=self.root/'new/journal/new.json';bad.parent.mkdir(parents=True);bad.write_text('{')
        self.assertFalse(reader.status({})['available'])

    def test_current_model_mismatch_and_record_age_are_explicit(self):
        d=ControlTelemetry([self.root],refresh_s=0).status(dict(model_sha256='other'))
        self.assertFalse(d['model_matches_monitor'])
        self.assertGreaterEqual(d['record_age_s'],0)
        self.assertFalse(d['live_camera_overlay'])

    def test_other_profile_uses_predecessor_only_for_pose_not_old_object(self):
        plan=copy.deepcopy(self.plan);plan['profile']='can-profile';self.plan_hash=save(self.trial/'plan.json',plan)
        self.record['plan_sha256']=self.plan_hash;save(self.path,self.record)
        child=self.root/'empty'
        next_plan=dict(task_id='empty',profile='empty_claw_plastic_bin',model_version='model',
            prior_plan_path='can/plan.json',prior_plan_sha256=self.plan_hash,
            prior_execution_path='can/journal/close.json',prior_execution_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
            stages=[dict(name='empty',steps=[])])
        ph=save(child/'plan.json',next_plan)
        record=dict(task_id='empty',stage='empty',plan_sha256=ph,status='sent_pending_observation',serial_opened=True,steps=[])
        path=child/'journal/empty.json';save(path,record)
        result=project_record(path,record)
        self.assertTrue(result['scope_boundary'])
        self.assertIsNone(result['pick_pixel'])
        self.assertIsNone(result['geometry'])
        self.assertFalse(result['close_command_sent'])
        history=ControlTelemetry([self.root],refresh_s=0).status({},task_id='can')
        self.assertEqual(history['task_id'],'can')
        self.assertFalse(history['following_latest'])
        self.assertEqual(history['pick_pixel'],[2,3])

    def test_missing_records_is_not_a_hardware_failure_or_success(self):
        d=ControlTelemetry([self.root/'missing'],refresh_s=0).status({})
        self.assertFalse(d['available'])


class DashboardHttpTests(unittest.TestCase):
    def test_read_only_status_and_no_execution_routes(self):
        telemetry=SimpleNamespace(status=lambda h,task_id=None:dict(available=False,readonly=True))
        server=make_server(0,'http://127.0.0.1:1',telemetry)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base='http://127.0.0.1:'+str(server.server_port)
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(base+'/control-status',timeout=4) as reply:
                self.assertTrue(json.load(reply)['readonly'])
            for path in ('/execute','/grasp','/serial','/control-status'):
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    opener.open(urllib.request.Request(base+path,data=b'{}'),timeout=2)
                self.assertEqual(raised.exception.code,404);raised.exception.close()
            with self.assertRaises(urllib.error.HTTPError) as raised:
                opener.open(urllib.request.Request(base+'/control-status',headers={'Origin':'https://example.com'}),timeout=2)
            self.assertEqual(raised.exception.code,403);raised.exception.close()
        finally:server.shutdown();thread.join();server.server_close()


if __name__=='__main__':unittest.main()
