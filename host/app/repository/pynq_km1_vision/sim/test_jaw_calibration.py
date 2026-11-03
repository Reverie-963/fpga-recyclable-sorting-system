"""No-device tests for one-step empty-jaw measurement and its history chain."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from src.arm import packet
from src.jaw_calibration import request,plan,history_state,reference_record
from src.sorting_neutral_return import confirmed_start
from src.board_station_host import Authority
from src.board_control_ownership import BoardOwnership


class JawCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.folder=Path(self.tmp.name);self.neutral=[1500]*5+[1000]
        self.c=dict(known=dict(initial_pwm=self.neutral,prepare_pwm=[1500]*4),
                    profiles={'plastic':{'close_pwm':1300},'paper':{'close_pwm':1500}})
        self.parent=self.folder/'prior.json'
        self.record(self.parent,'known_physical',packet(list(enumerate(self.neutral)),5000))
        self.value=dict(target_pwm=1100,source_task_id='prior',source_sha256=self.sha(self.parent),gap_mm=61,hands_clear=True)

    def sha(self,p):return hashlib.sha256(p.read_bytes()).hexdigest()
    def record(self,path,mode,command,**extra):
        r=dict(task_id=path.stem,mode=mode,physical_motion_attempted=True,
               plan=[dict(command=command.decode())],steps=[dict(transport_receipt=dict(
                   sent_hex=command.hex(),received_hex=command.hex(),echo_matched=True,completed_movement_wait=True))],**extra)
        path.write_text(json.dumps(r));return r

    def test_first_step_only_moves_jaw_with_slow_duration(self):
        steps,target=plan(self.c,self.neutral,self.value,self.parent)
        self.assertEqual(steps,[('jaw_calibration',packet([(5,1100)],3000),3000)])
        self.assertEqual(target[:5],self.neutral[:5])

    def test_oversized_unknown_or_near_contact_step_rejected(self):
        for change in ({'target_pwm':1200},{'target_pwm':1600},{'target_pwm':1000},
                       {'hands_clear':False},{'gap_mm':0},{'gap_mm':2},{'gap_mm':float('nan')},
                       {'target_pwm':True},{'source_task_id':'../escape'}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                plan(self.c,self.neutral,dict(self.value,**change),self.parent)

    def test_measurement_must_bind_exact_prior_report(self):
        self.parent.write_text(self.parent.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'history_changed'):plan(self.c,self.neutral,self.value,self.parent)

    def test_other_joint_or_unchecked_closure_rejected(self):
        state=self.neutral.copy();state[0]=1400
        with self.assertRaises(ValueError):plan(self.c,state,self.value,self.parent)
        state=self.neutral[:5]+[1500]
        with self.assertRaises(ValueError):plan(self.c,state,dict(self.value,target_pwm=1600),self.parent)

    def child(self):
        path=self.folder/'jaw-next.json'
        self.record(path,'supervised_jaw_calibration',packet([(5,1100)],3000),
            status='jaw_calibration_command_completed',source_task_id='prior',source_report_sha256=self.sha(self.parent),
            start_pwm=self.neutral,target_pwm=self.neutral[:5]+[1100])
        return path

    def test_chain_reconstructs_stationary_joints_and_jaw(self):
        p=self.child();self.assertEqual(history_state(p,self.neutral),self.neutral[:5]+[1100])
        state,_,_=confirmed_start(self.c,self.folder,allow_jaw=True)
        self.assertEqual(state[-1],1100)
        with self.assertRaisesRegex(ValueError,'jaw_not_open'):confirmed_start(self.c,self.folder)

    def test_reference_remains_original_and_parent_tampering_is_rejected(self):
        p=self.child();reference,record=reference_record(p,self.neutral)
        self.assertEqual(reference,self.parent);self.assertEqual(record['task_id'],'prior')
        self.parent.write_text(self.parent.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'parent_changed'):reference_record(p,self.neutral)

    def extension_fixture(self):
        parent=self.parent;state=self.neutral.copy()
        for pwm in range(1100,1501,100):
            path=self.folder/('jaw-'+str(pwm)+'.json');target=state[:5]+[pwm]
            self.record(path,'supervised_jaw_calibration',packet([(5,pwm)],3000),
                status='jaw_calibration_command_completed',source_task_id=parent.stem,
                source_report_sha256=self.sha(parent),start_pwm=state,target_pwm=target)
            parent=path;state=target
        extension=dict(kind='operator-approved-jaw-extension-v1',operator_statement='synthetic approved scope',
            max_pwm=1600,baseline_task_id=parent.stem,baseline_sha256=self.sha(parent))
        value=dict(self.value,target_pwm=1525,source_task_id=parent.stem,source_sha256=self.sha(parent),gap_mm=14)
        return parent,state,extension,value

    def test_extension_requires_bound_authorization_and_matching_baseline(self):
        p,state,ext,v=self.extension_fixture()
        for invalid in (None,dict(ext,max_pwm=1700),dict(ext,operator_statement=''),dict(ext,baseline_sha256='0'*64)):
            with self.subTest(extension=invalid),self.assertRaises(ValueError):plan(self.c,state,v,p,invalid)
        steps,target=plan(self.c,state,v,p,ext)
        self.assertEqual(steps,[('jaw_calibration',packet([(5,1525)],3000),3000)])
        self.assertEqual(target[:5],state[:5])

    def test_extension_reduces_steps_near_contact_and_stops_at_two_mm(self):
        p,state,ext,v=self.extension_fixture()
        for gap,delta,valid in ((14,25,True),(14,26,False),(8,10,True),(8,11,False),
                                (4,5,True),(4,6,False),(2,1,False),(0,1,False)):
            with self.subTest(gap=gap,delta=delta):
                value=dict(v,gap_mm=gap,target_pwm=1500+delta)
                if valid:plan(self.c,state,value,p,ext)
                else:
                    with self.assertRaises(ValueError):plan(self.c,state,value,p,ext)
        with self.assertRaises(ValueError):plan(self.c,state,dict(v,target_pwm=1601),p,ext)

    def test_extension_history_requires_recorded_scope_and_preserves_original_reference(self):
        p,state,ext,v=self.extension_fixture();child=self.folder/'jaw-extended.json'
        row=self.record(child,'supervised_jaw_calibration',packet([(5,1525)],3000),
            status='jaw_calibration_command_completed',source_task_id=p.stem,source_report_sha256=self.sha(p),
            start_pwm=state,target_pwm=state[:5]+[1525],jaw_calibration=dict(previous_measurement=v,extension=ext))
        self.assertEqual(history_state(child,self.neutral)[5],1525)
        self.assertEqual(reference_record(child,self.neutral)[0],self.parent)
        row['jaw_calibration'].pop('extension');child.write_text(json.dumps(row))
        with self.assertRaises(ValueError):history_state(child,self.neutral)

    def test_zero_gap_allows_opening_but_not_further_closing(self):
        p,state,ext,v=self.extension_fixture()
        steps,target=plan(self.c,state,dict(v,gap_mm=0,target_pwm=1400),p,ext)
        self.assertEqual(target[-1],1400)

    def test_history_tampering_unknown_receipt_or_joint_packet_rejected(self):
        for kind in ('parent','receipt','joint','status'):
            with self.subTest(kind=kind):
                self.setUp();p=self.child();r=json.loads(p.read_text())
                if kind=='parent':self.parent.write_text(self.parent.read_text()+' ')
                if kind=='receipt':r['steps'][0]['transport_receipt']['completed_movement_wait']=False
                if kind=='status':r['status']='needs_review'
                if kind=='joint':
                    cmd=packet([(0,1400),(5,1100)],3000)
                    r['plan'][0]['command']=cmd.decode()
                    r['steps'][0]['transport_receipt'].update(sent_hex=cmd.hex(),received_hex=cmd.hex())
                p.write_text(json.dumps(r))
                with self.assertRaises(ValueError):history_state(p,self.neutral)

    def test_complete_command_releases_owner_without_claiming_grasp(self):
        owner=BoardOwnership(self.folder/'owner.db');self.addCleanup(owner.close)
        authority=Authority(owner);authority.begin('test-owner','test-request');authority.port_attempted=True
        r=json.loads(self.child().read_text());authority.finish(r,False)
        self.assertEqual(owner.status()['status'],'idle')
        self.assertEqual(r['status'],'jaw_calibration_command_completed')

    def test_incomplete_command_keeps_owner_blocked(self):
        owner=BoardOwnership(self.folder/'owner.db');self.addCleanup(owner.close)
        authority=Authority(owner);authority.begin('test-owner','test-request');authority.port_attempted=True
        r=json.loads(self.child().read_text());r['steps'][0]['transport_receipt']['echo_matched']=False
        authority.finish(r,False);self.assertEqual(owner.status()['status'],'blocked')


if __name__=='__main__':unittest.main()
