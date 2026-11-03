"""Bind an enabled jaw table to manual readings and unchanged command receipts."""
import json
from pathlib import Path
from .sorting_known import bound_file
from .sorting_loop import need,fingerprint
from openmv_gripper.km1_arm_bridge import validate_packet


def validate_measurement_record(path,calibration,operator_statement):
    path=Path(path);record=json.loads(path.read_text(encoding='utf-8'))
    need(record.get('kind')=='operator-enabled-jaw-measurements-v1' and
         record.get('operator_statement')==operator_statement and bool(operator_statement.strip()) and
         record.get('calibration_sha256')==fingerprint(calibration) and
         record.get('measurement_id')==calibration['measurement_id'] and
         record.get('physical_grip_verified') is False and record.get('unattended_accepted') is False,
         'jaw_measurement_activation_not_bound')
    rows=record.get('measurements')
    need(isinstance(rows,list) and len(rows)==len(calibration['samples']),'jaw_measurement_samples_missing')
    observed=[]
    for row in rows:
        measurement=json.loads(bound_file(path.parent,row['measurement']).read_text(encoding='utf-8'))
        report=json.loads(bound_file(path.parent,row['command_report']).read_text(encoding='utf-8'))
        need(measurement.get('measurement_source')=='operator_mm_ruler' and
             measurement.get('hands_and_ruler_removed') is True and
             measurement.get('source_task_id')==report.get('task_id') and
             measurement.get('report_sha256')==row['command_report']['sha256'],
             'jaw_manual_measurement_not_bound')
        need(report.get('physical_motion_attempted') is True and report.get('status') in
             ('completed_visual_evidence','neutral_return_command_completed','jaw_calibration_command_completed'),
             'jaw_measurement_motion_not_completed')
        need(isinstance(report.get('steps'),list) and report['steps'] and
             len(report['steps'])==len(report['plan']),'jaw_measurement_receipts_missing')
        jaw=None
        for i,step in enumerate(report['steps']):
            receipt=step.get('transport_receipt',{})
            need(receipt.get('echo_matched') is True and receipt.get('completed_movement_wait') is True and
                 receipt.get('sent_hex')==receipt.get('received_hex'),'jaw_measurement_uncertain_receipt')
            command=bytes.fromhex(receipt['sent_hex']);validate_packet(command,[[500,2500]]*6)
            text=command.decode('ascii')
            need(text==report['plan'][i]['command'],'jaw_measurement_command_mismatch')
            for offset in range(1,len(text)-1,15):
                word=text[offset:offset+15]
                if int(word[1:4])==5:jaw=int(word[5:9])
        need(jaw==measurement['jaw_pwm'],'jaw_measurement_pwm_mismatch')
        observed.append(dict(gap_mm=measurement['gap_mm'],pwm=measurement['jaw_pwm']))
    need(sorted(observed,key=lambda s:s['gap_mm'])==calibration['samples'],'jaw_measurement_table_mismatch')
    return record
