"""Neutral return only from completed receipts; tests never open a serial port."""
import hashlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
import time
from unittest.mock import patch
import numpy as np
from src.arm import packet
from src.sorting_neutral_return import confirmed_start,neutral_plan,calibration_return_plan,source_return_plan,FrozenReturnGuard,return_neutral
from src.monitor_tasks import recorded_frame as read_evidence


class NeutralReturnTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.neutral=[1500,1500,1500,1500,1500,1000]
        self.prepare=[1500,1632,1987,928,1500,1000]
        self.c=dict(known=dict(initial_pwm=self.neutral,prepare_pwm=self.prepare[:4],device_lease='arm.lock',
            stop_files=['STOP']),journal_dir='ledger',profiles={k:dict(retreat_pwm=[self.prepare[:4],self.neutral[:4]]) for k in ['paper','metal','plastic']})
        self.folder=self.root/'ledger';self.folder.mkdir()

    def record(self,state=None,uncertain=False):
        state=self.prepare if state is None else state;command=packet(list(enumerate(state)),1000).decode()
        encoded=command.encode().hex()
        r=dict(task_id='task',mode='known_physical',physical_motion_attempted=True,
            reason='ValueError:motion_requires_three_fixed_markers',
            plan=[dict(step='home',command=command,duration_ms=1000)],
            steps=[dict(step='home',transport_receipt=dict(sent_hex=encoded,received_hex=encoded,
                echo_matched=True,completed_movement_wait=True))])
        if uncertain:r['steps'].append(dict(step='source_high',status='may_be_in_flight'))
        path=self.folder/'task.json';path.write_text(json.dumps(r),encoding='utf-8')
        proof=dict(kind='reviewed-prewrite-neutral-return-start-v1',report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            task_id='task',unwritten_step='source_high',reason=r['reason'],motion_command_written=False)
        return r,path,proof

    def test_unknown_write_needs_bound_exact_prewrite_proof(self):
        r,path,proof=self.record(uncertain=True)
        with self.assertRaisesRegex(ValueError,'last_command_uncertain'):confirmed_start(self.c,self.folder)
        state,actual,_=confirmed_start(self.c,self.folder,proof)
        self.assertEqual(state,self.prepare);self.assertEqual(actual,path)
        path.write_text(path.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'last_command_uncertain'):confirmed_start(self.c,self.folder,proof)

    def test_host_attested_no_uart_parent_and_hash_linked_child_do_not_replace_known_pose(self):
        _,known,_=self.record(self.neutral)
        child=self.folder/'no-motion-child.json'
        body=dict(task_id=child.stem,batch_id='no-motion-batch',mode='known_physical',physical_motion_attempted=True,steps=[],plan=[])
        child.write_text(json.dumps(body));parent=self.folder/'no-motion-batch.json'
        manifest=dict(task_id=parent.stem,batch_id=parent.stem,mode='known_physical',physical_motion_attempted=True,
            board_uart_write_attempted=False,steps=[],items=[dict(task_id=child.stem,report_sha256=hashlib.sha256(child.read_bytes()).hexdigest())])
        parent.write_text(json.dumps(manifest))
        import os
        stamp=time.time_ns()+10**9;os.utime(child,ns=(stamp,stamp));os.utime(parent,ns=(stamp+1,stamp+1))
        state,path,report=confirmed_start(self.c,self.folder)
        self.assertEqual(state,self.neutral);self.assertEqual(path,known)
        self.assertEqual(len(report['pose_history_skipped_no_uart']),2)
        child.write_text(child.read_text()+' ')
        with self.assertRaises(ValueError):confirmed_start(self.c,self.folder)

    def test_zero_steps_or_missing_no_uart_attestation_never_proves_no_motion(self):
        self.record(self.neutral)
        path=self.folder/'uncertain.json';path.write_text(json.dumps(dict(task_id='uncertain',mode='known_physical',physical_motion_attempted=True,steps=[])))
        import os
        stamp=time.time_ns()+10**9;os.utime(path,ns=(stamp,stamp))
        with self.assertRaisesRegex(ValueError,'known_empty_prepare_or_neutral'):confirmed_start(self.c,self.folder)

    def test_other_pose_or_held_gripper_is_rejected(self):
        for state in ([1400,*self.prepare[1:]],self.prepare[:5]+[1300]):
            self.record(state)
            with self.assertRaisesRegex(ValueError,'known_empty_prepare_or_neutral'):
                confirmed_start(self.c,self.folder)

    def source_record(self):
        high=[1674,1645,2108,962];hover=[1674,1394,1973,857];mid=[1677,1406,2135,1006];grip=[1680,1377,2249,1149]
        names=['home','source_high','align_wrist','source_hover','source_mid','source_grip','close',
               'source_lift_mid','source_lift_hover','source_lift_high','front_high','restore_wrist_high','transfer_0']
        targets=[dict(enumerate(self.prepare)),dict(enumerate(high)),{4:1499},dict(enumerate(hover)),
            dict(enumerate(mid)),dict(enumerate(grip)),{5:1316},dict(enumerate(mid)),dict(enumerate(hover)),
            dict(enumerate(high)),dict(enumerate([1500,1685,2031,993])),{4:1500},dict(enumerate(self.prepare[:4]))]
        plan=[dict(step=name,command=packet(sorted(values.items()),500).decode(),duration_ms=500) for name,values in zip(names,targets)]
        steps=[dict(step=p['step'],transport_receipt=dict(sent_hex=p['command'].encode().hex(),
            received_hex=p['command'].encode().hex(),echo_matched=True,completed_movement_wait=True)) for p in plan[:6]]
        steps.append(dict(step='close',status='may_be_in_flight'))
        report=dict(task_id='source-task',mode='known_physical',status='needs_review',reason='ValueError:board_control_blocked',
            physical_motion_attempted=True,plan=plan,steps=steps)
        path=self.folder/'source-task.json';path.write_text(json.dumps(report))
        return report,path,grip+[1499,None]

    def test_source_recovery_preserves_unknown_jaw_then_opens_and_retraces_original_approach(self):
        report,path,start=self.source_record()
        with self.assertRaisesRegex(ValueError,'last_command_uncertain'):confirmed_start(self.c,self.folder)
        self.assertEqual(confirmed_start(self.c,self.folder,allow_source=True)[0],start)
        plan,height=source_return_plan(self.c,start,report)
        self.assertEqual(plan[0][1],packet([(5,1000)],3000));self.assertEqual(len(plan),8)
        self.assertEqual(plan[-1][1],neutral_plan(self.c,self.prepare)[0][0][1])
        for recovery,index in zip(plan[1:4],(4,3,1)):
            self.assertEqual(recovery[1].decode().replace('T3000','T0500'),report['plan'][index]['command'])
        self.assertEqual(report['steps'][-1]['status'],'may_be_in_flight');self.assertIsNone(start[5])

    def test_source_recovery_rejects_unknown_arm_commands_changed_reverse_route_and_extra_steps(self):
        report,path,start=self.source_record()
        variants=[lambda r:r['plan'][6].update(command=packet([(0,1600),(5,1316)],500).decode()),
                  lambda r:r['plan'][7].update(command=packet([(0,1600),(1,1406),(2,2135),(3,1006)],500).decode()),
                  lambda r:r['steps'][5]['transport_receipt'].update(completed_movement_wait=False),
                  lambda r:r['steps'].append(dict(step='source_lift_mid',status='may_be_in_flight')),
                  lambda r:r.update(reason='missing_serial_echo_execution_unknown')]
        for mutate in variants:
            changed=copy.deepcopy(report);mutate(changed);path.write_text(json.dumps(changed))
            with self.subTest(changed=changed),self.assertRaises(ValueError):confirmed_start(self.c,self.folder,allow_source=True)

    def test_source_recovery_preflight_never_opens_uart_and_requires_fresh_empty_jaw_confirmation(self):
        self.source_record();b=dict(implementation_sources=[dict(path='sorting_neutral_return.py')])
        bundle=(b,self.c,self.root,{}, {}, {}, {}, {})
        with patch('src.sorting_neutral_return.load_unified',return_value=(bundle,{})),patch(
                'src.sorting_neutral_return.integration_blockers',return_value=[]),patch('src.sorting_neutral_return.Motion.open') as opened:
            result=return_neutral(self.root/'binding.json');self.assertEqual(result['command_count'],8)
            self.assertEqual(result['empty_jaw_opening_steps'],1);self.assertIsNone(result['start_pwm'][5])
            with self.assertRaisesRegex(ValueError,'scene_confirmation_required'):return_neutral(self.root/'binding.json',execute=True)
            opened.assert_not_called()

    def test_newer_unknown_motion_is_not_skipped_for_an_older_known_pose(self):
        self.record();p=self.folder/'foreign.json'
        p.write_text(json.dumps({'mode':'other_controller','physical_motion_attempted':True}))
        import os
        later=time.time_ns()+10**9;os.utime(p,ns=(later,later))
        with self.assertRaisesRegex(ValueError,'unrecognized_motion_history'):
            confirmed_start(self.c,self.folder)

    def test_plan_reuses_last_retreat_segment_with_low_speed_and_height_check(self):
        plan,height=neutral_plan(self.c,self.prepare)
        self.assertEqual(len(plan),1);self.assertEqual(plan[0][2],5000)
        self.assertGreater(height,160)
        for index,target in enumerate(self.neutral):self.assertIn(f'#{index:03d}P{target:04d}',plan[0][1].decode())
        for p in self.c['profiles'].values():p['retreat_pwm'][-1]=[1400,1500,1500,1500]
        with self.assertRaisesRegex(ValueError,'original_retreat_segment'):neutral_plan(self.c,self.prepare)

    def test_other_material_can_have_a_different_retreat_approach(self):
        self.c['profiles']['paper']['retreat_pwm'][-2]=[1500,1568,1875,843]
        plan,_=neutral_plan(self.c,self.prepare)
        self.assertEqual(plan[0][2],5000)

    def test_calibration_recovery_opens_only_jaw_in_checked_steps_before_original_home(self):
        plan,_=calibration_return_plan(self.c,self.neutral[:5]+[1600])
        self.assertEqual([d for _,_,d in plan],[3000]*6+[5000])
        self.assertEqual([p for _,p,_ in plan[:-1]],[packet([(5,pwm)],3000) for pwm in (1500,1400,1300,1200,1100,1000)])
        self.assertEqual(plan[-1][1],neutral_plan(self.c,self.neutral)[0][0][1])
        for state in (self.prepare[:5]+[1300],self.neutral[:5]+[1700],self.neutral[:5]+[900]):
            with self.subTest(state=state),self.assertRaisesRegex(ValueError,'outside_checked_range'):
                calibration_return_plan(self.c,state)

    def test_prepare_does_not_connect_and_execute_requires_separate_confirmation(self):
        self.record()
        b=dict(implementation_sources=[dict(path='sorting_neutral_return.py')])
        bundle=(b,self.c,self.root,{}, {}, {}, {}, {})
        with patch('src.sorting_neutral_return.load_unified',return_value=(bundle,{})),patch(
                'src.sorting_neutral_return.integration_blockers',return_value=[]),patch('src.sorting_neutral_return.Motion.open') as opened:
            result=return_neutral(self.root/'binding.json')
            self.assertEqual(result['status'],'neutral_return_ready');opened.assert_not_called()
            with self.assertRaisesRegex(ValueError,'scene_confirmation_required'):
                return_neutral(self.root/'binding.json',execute=True)
            opened.assert_not_called()

    def test_two_marker_guard_retains_frozen_shift_without_relocalizing(self):
        image=np.zeros((80,100,3),np.uint8)
        detections=[dict(id=i,corners_px=p) for i,p in [(10,[[10,10],[30,10],[30,30],[10,30]]),
            (12,[[50,40],[70,40],[70,60],[50,60]])]]
        with patch('src.sorting_neutral_return.detect_markers',return_value=detections):
            g=FrozenReturnGuard(image,{},dict(markers=[dict(id=10),dict(id=12)]),{'search_margin_px':5},2,5)
        with patch('src.sorting_neutral_return.detect_markers',return_value=detections),patch(
                'src.sorting_neutral_return.check_motion_layout',return_value={'passed':True}):
            self.assertFalse(g.check(image)['camera_pose_recomputed'])
        moved=json.loads(json.dumps(detections));moved[0]['corners_px']=(np.array(moved[0]['corners_px'])+[6,0]).tolist()
        with patch('src.sorting_neutral_return.detect_markers',return_value=moved):
            with self.assertRaisesRegex(ValueError,'fixed_board_moved'):g.check(image)

    def test_complete_return_clears_only_own_pending_but_partial_write_latches(self):
        jaw=getattr(self,'jaw_test',False)
        recovering=getattr(self,'recovery_test',False)
        r,path,_=self.record(self.neutral if jaw or recovering else None);r['evidence']=[{'image':'prior.png'}];path.write_text(json.dumps(r))
        if recovering:
            command=packet([(5,1300)],3000)
            child=dict(task_id='calibrated',mode='supervised_jaw_calibration',status='jaw_calibration_command_completed',
                physical_motion_attempted=True,source_task_id=path.stem,source_report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                start_pwm=self.neutral,target_pwm=self.neutral[:5]+[1300],
                plan=[dict(command=command.decode())],steps=[dict(transport_receipt=dict(sent_hex=command.hex(),received_hex=command.hex(),echo_matched=True,completed_movement_wait=True))])
            (self.folder/'calibrated.json').write_text(json.dumps(child))
        for p in self.c['profiles'].values():p['close_pwm']=1500
        kwargs={}
        if jaw:kwargs['jaw_step']=dict(target_pwm=1100,source_task_id='task',
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),gap_mm=61,hands_clear=True)
        self.c.update(arm={},statistics={'url':'http://127.0.0.1:1'},layout={},frame_timeout_s=2,max_age_ms=2000,image_size=[10,10])
        for name,p in self.c['profiles'].items():p['bin_id']='bin_'+name
        b=dict(implementation_sources=[dict(path='sorting_neutral_return.py'),dict(path='jaw_calibration.py')],motion_guard_policy=dict(
            min_fixed_markers=2,marker_count_operator_statement='synthetic two-marker policy',
            kind='operator-approved-station-motion-margin-v1',operator_statement='synthetic five pixels',
            production_point_domains_changed=False,max_displacement_px=5))
        bundle=(b,self.c,self.root,{}, {'markers':[]}, {}, {}, {})
        health={'regions':{k:[0,0,2,2] for k in ['source','static_a','static_b','bin_paper','bin_metal','bin_plastic']}}
        image=np.zeros((10,10,3),np.uint8);digest=hashlib.sha256(image.tobytes()).hexdigest()
        def raw(*args):
            if getattr(self,'recovery_block_second',False) and len(FakeMotion.calls)==1:
                raise ValueError('synthetic scene changed before second opening')
            return dict(image=image,observation=dict(source_stream_id='fixed',source_frame_id='1',
                capture_started_ms=time.monotonic()*1000),bgr_sha256=digest,jpeg_sha256=digest)
        class FakeMotion:
            calls=[];partial=False
            def __init__(self,*args):self.last_receipt=None
            def check_stop(self):pass
            def open(self,plan):pass
            def send(self,data,duration,guard):
                guard();self.calls.append(data)
                if self.partial:raise IOError('synthetic partial write')
                self.last_receipt=dict(sent_hex=data.hex(),received_hex=data.hex(),echo_matched=True,completed_movement_wait=False)
            def wait(self,seconds):self.last_receipt['completed_movement_wait']=True
            def close(self):pass
        with patch('src.sorting_neutral_return.load_unified',return_value=(bundle,{})),patch(
            'src.sorting_neutral_return.integration_blockers',return_value=[]),patch(
            'src.monitor_tasks.recorded_frame',return_value=(b'',{'source_stream_id':'fixed'})),patch(
            'src.sorting_neutral_return.cv2.imread',return_value=image),patch(
            'src.sorting_neutral_return.FrozenReturnGuard') as frozen,patch(
            'src.sorting_neutral_return.StationCamera') as camera,patch(
            'src.sorting_neutral_return.StationStatistics') as statistics,patch(
            'urllib.request.urlopen',side_effect=lambda *a,**k:io.StringIO(json.dumps(health))),patch(
            'src.sorting_neutral_return.Motion',FakeMotion):
            frozen.return_value.check.return_value={'passed':True};camera.return_value.raw.side_effect=raw
            statistics.return_value.stream='fixed'
            if recovering:
                preview=return_neutral(self.root/'binding.json')
                self.assertEqual(preview['empty_jaw_opening_steps'],3)
                self.assertEqual(preview['duration_ms'],14000);self.assertEqual(FakeMotion.calls,[])
                with self.assertRaisesRegex(ValueError,'scene_confirmation_required'):
                    return_neutral(self.root/'binding.json',execute=True)
            result=return_neutral(self.root/'binding.json',execute=True,confirmed=True,**kwargs)
            if getattr(self,'recovery_block_second',False):
                self.assertEqual(result['status'],'needs_review');self.assertTrue(result['physical_motion_attempted'])
                self.assertEqual(len(FakeMotion.calls),1);self.assertTrue((self.folder/'pending.json').exists())
                self.assertEqual(result['steps'][1]['status'],'blocked_before_write')
                return
            self.assertEqual(result['status'],'jaw_calibration_command_completed' if jaw else 'neutral_return_command_completed',result)
            count=4 if recovering else 1
            self.assertFalse((self.folder/'pending.json').exists());self.assertEqual(len(FakeMotion.calls),count)
            if recovering:
                self.assertEqual(FakeMotion.calls[:3],[packet([(5,pwm)],3000) for pwm in (1200,1100,1000)])
                self.assertEqual(FakeMotion.calls[-1],neutral_plan(self.c,self.neutral)[0][0][1])
                self.assertEqual(confirmed_start(self.c,self.folder)[0],self.neutral)
            _,observation=read_evidence(self.folder,result['evidence'][-1],pixels=True)
            self.assertEqual(observation['image_size'],[10,10])
            if jaw:
                self.assertEqual(FakeMotion.calls[0],packet([(5,1100)],3000))
                self.assertIsNone(result['jaw_calibration']['gap_mm'])
                next_path=self.folder/(result['task_id']+'.json')
                kwargs['jaw_step'].update(target_pwm=1200,source_task_id=result['task_id'],
                    source_sha256=hashlib.sha256(next_path.read_bytes()).hexdigest(),gap_mm=50)
            FakeMotion.partial=True
            result=return_neutral(self.root/'binding.json',execute=True,confirmed=True,**kwargs)
            self.assertEqual(result['status'],'needs_review');self.assertTrue(result['physical_motion_attempted'])
            self.assertEqual(len(FakeMotion.calls),count+1);self.assertTrue((self.folder/'pending.json').exists())

    def test_jaw_step_uses_guards_receipts_and_preserves_partial_write(self):
        self.jaw_test=True
        self.test_complete_return_clears_only_own_pending_but_partial_write_latches()

    def test_confirmed_closed_calibration_recovers_in_one_transaction(self):
        self.recovery_test=True
        self.test_complete_return_clears_only_own_pending_but_partial_write_latches()

    def test_mid_recovery_scene_failure_keeps_prior_motion_and_pending(self):
        self.recovery_test=True;self.recovery_block_second=True
        self.test_complete_return_clears_only_own_pending_but_partial_write_latches()
