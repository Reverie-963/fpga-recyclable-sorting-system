"""Synthetic evidence checks for opt-in jaw calibration; no device I/O."""
import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from src.arm import packet
from src.contour_grasp import jaw_command
from src.jaw_measurements import validate_measurement_record
from src.sorting_known import profile_fingerprint
from src.sorting_loop import fingerprint
from src.sorting_unified_plan import compile_route,validate_policy
from sim.test_sorting_unified import test_config,setup_policy


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.cal=dict(kind='measured-jaw-gap-v1',measurement_id='synthetic-test-only',compression_mm=0,
            opening_margin_mm=0,samples=[dict(gap_mm=9,pwm=1600),dict(gap_mm=61,pwm=1000)])
        self.rows=[]
        for n,sample in enumerate(self.cal['samples']):
            command=packet([(5,sample['pwm'])],3000);task='sample-'+str(n)
            report=self.root/(task+'.json');report.write_text(json.dumps(dict(task_id=task,
                status='jaw_calibration_command_completed',physical_motion_attempted=True,
                plan=[dict(command=command.decode())],steps=[dict(transport_receipt=dict(sent_hex=command.hex(),
                received_hex=command.hex(),echo_matched=True,completed_movement_wait=True))])))
            measurement=self.root/('measurement-'+str(n)+'.json')
            measurement.write_text(json.dumps(dict(measurement_source='operator_mm_ruler',hands_and_ruler_removed=True,
                source_task_id=task,report_sha256=self.sha(report),jaw_pwm=sample['pwm'],gap_mm=sample['gap_mm'])))
            self.rows.append(dict(measurement=self.ref(measurement),command_report=self.ref(report)))
        self.record=self.root/'record.json';self.write_record()

    def sha(self,p):return hashlib.sha256(p.read_bytes()).hexdigest()
    def ref(self,p):return dict(path=p.name,sha256=self.sha(p))
    def write_record(self):
        self.record.write_text(json.dumps(dict(kind='operator-enabled-jaw-measurements-v1',operator_statement='enable',
            calibration_sha256=fingerprint(self.cal),measurement_id=self.cal['measurement_id'],measurements=self.rows,
            physical_grip_verified=False,unattended_accepted=False)))

    def test_exact_manual_readings_and_receipts_validate(self):
        validate_measurement_record(self.record,self.cal,'enable')

    def test_changed_manual_file_or_table_is_rejected(self):
        with self.assertRaises(ValueError):validate_measurement_record(self.record,dict(self.cal,compression_mm=1),'enable')
        path=self.root/self.rows[0]['measurement']['path'];path.write_text(path.read_text()+' ')
        with self.assertRaises(ValueError):validate_measurement_record(self.record,self.cal,'enable')

    def test_rehashed_incorrect_pwm_or_uncertain_receipt_is_rejected(self):
        measurement=self.root/self.rows[0]['measurement']['path'];value=json.loads(measurement.read_text())
        value['jaw_pwm']=1550;measurement.write_text(json.dumps(value));self.rows[0]['measurement']=self.ref(measurement);self.write_record()
        with self.assertRaisesRegex(ValueError,'pwm_mismatch'):validate_measurement_record(self.record,self.cal,'enable')
        value['jaw_pwm']=1600
        report=self.root/self.rows[0]['command_report']['path'];r=json.loads(report.read_text())
        r['steps'][0]['transport_receipt']['completed_movement_wait']=False;report.write_text(json.dumps(r))
        self.rows[0]['command_report']=self.ref(report);value['report_sha256']=self.sha(report)
        measurement.write_text(json.dumps(value));self.rows[0]['measurement']=self.ref(measurement);self.write_record()
        with self.assertRaisesRegex(ValueError,'uncertain_receipt'):validate_measurement_record(self.record,self.cal,'enable')

    def test_shared_policy_changes_only_closure_and_preserves_base_qualification(self):
        c=test_config();policy=setup_policy(c);policy['jaw_open_span_mm']=110
        base=copy.deepcopy(c);identity=profile_fingerprint(c,'plastic')
        slot=policy['bin_slots']['plastic'][0]
        before=compile_route(c,'plastic',[0,5],90,policy,slot)
        policy.update(jaw_calibration=self.cal,jaw_calibration_operator_statement='enable')
        validate_policy(policy,c)
        after=compile_route(c,'plastic',[0,5],90,policy,slot,grasp_width_mm=9)
        self.assertEqual(c,base);self.assertEqual(profile_fingerprint(c,'plastic'),identity)
        self.assertTrue(after['jaw_closure']['width_controls_closure'])
        for a,b in zip(before['steps'],after['steps']):
            if b['name']=='close':self.assertIn('#005P1600',b['command'])
            else:self.assertEqual(a,b)
        for width in (8.99,61.01):
            with self.assertRaises(ValueError):jaw_command(c['profiles']['plastic'],policy,width)

    def test_shared_policy_needs_explicit_activation_and_rejects_conflict(self):
        profile=dict(open_pwm=1000,close_pwm=1300);policy=dict(jaw_open_span_mm=110,jaw_calibration=self.cal)
        with self.assertRaisesRegex(ValueError,'statement_required'):jaw_command(profile,policy,9)
        policy['jaw_calibration_operator_statement']='enable';profile['jaw_calibration']=dict(self.cal,compression_mm=1)
        with self.assertRaisesRegex(ValueError,'conflicting'):jaw_command(profile,policy,9)

    def test_twenty_percent_compression_and_measured_limits(self):
        profile=dict(open_pwm=1000,close_pwm=1300)
        policy=dict(jaw_open_span_mm=110,jaw_calibration=dict(self.cal,compression_ratio=.2),
                    jaw_calibration_operator_statement='operator trial')
        self.assertEqual(jaw_command(profile,policy,50)['target_gap_mm'],40)
        self.assertEqual(jaw_command(profile,policy,11.25)['target_gap_mm'],9)
        for width in (11.24,61.01):
            with self.assertRaises(ValueError):jaw_command(profile,policy,width)
        for ratio in (-.2,1,float('nan'),True):
            policy['jaw_calibration']['compression_ratio']=ratio
            with self.assertRaises(ValueError):jaw_command(profile,policy,50)


if __name__=='__main__':unittest.main()
