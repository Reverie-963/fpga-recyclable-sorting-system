"""Bounded empty-jaw measurement steps, never an arbitrary joint command."""
import hashlib
import json
from pathlib import Path
from .arm import packet
from .sorting_loop import finite, need


def request(value):
    need(isinstance(value,dict) and set(value)=={'target_pwm','source_task_id','source_sha256','gap_mm','hands_clear'},
         'jaw_calibration_request_required')
    need(type(value['target_pwm']) is int and 1000<=value['target_pwm']<=1600,'jaw_calibration_outside_checked_range')
    need(isinstance(value['source_task_id'],str) and value['source_task_id'] and
         all(c.isalnum() or c=='-' for c in value['source_task_id']),'jaw_calibration_source_task_invalid')
    need(isinstance(value['source_sha256'],str) and len(value['source_sha256'])==64 and
         all(c in '0123456789abcdef' for c in value['source_sha256']),'jaw_calibration_source_hash_invalid')
    need(finite(value['gap_mm']) and 0<=value['gap_mm']<=150 and value['hands_clear'] is True,
         'jaw_calibration_measurement_and_clearance_required')
    return dict(value)


def history_state(path,neutral,depth=0):
    """Reconstruct only completed single-jaw records rooted in neutral receipts."""
    need(depth<32,'jaw_calibration_history_too_deep')
    path=Path(path);r=json.loads(path.read_text(encoding='utf-8'))
    if r.get('mode')=='supervised_jaw_calibration':
        need(r.get('status')=='jaw_calibration_command_completed','jaw_calibration_previous_step_uncertain')
        parent_id=r.get('source_task_id')
        need(isinstance(parent_id,str) and parent_id and all(c.isalnum() or c=='-' for c in parent_id),
             'jaw_calibration_parent_invalid')
        parent=path.parent/(parent_id+'.json')
        need(parent!=path and hashlib.sha256(parent.read_bytes()).hexdigest()==r['source_report_sha256'],
             'jaw_calibration_parent_changed')
        state=history_state(parent,neutral,depth+1)
        need(r['start_pwm']==state and len(r['steps'])==1 and len(r['plan'])==1,
             'jaw_calibration_history_invalid')
        if max(state[5],r['target_pwm'][5])>1500:
            measurement=request(r.get('jaw_calibration',{}).get('previous_measurement'))
            need(measurement['target_pwm']==r['target_pwm'][5], 'jaw_calibration_history_target_mismatch')
            extended_step(r.get('jaw_calibration',{}).get('extension'),parent,state,measurement)
    else:
        need(r.get('mode') in ('known_physical','supervised_neutral_return'), 'jaw_calibration_unknown_motion_history')
        state=[None]*6
    need(r.get('physical_motion_attempted') is True and r.get('steps'),'jaw_calibration_no_motion_receipts')
    for i,step in enumerate(r['steps']):
        receipt=step.get('transport_receipt',{})
        need(receipt.get('echo_matched') is True and receipt.get('completed_movement_wait') is True and
             receipt.get('sent_hex')==receipt.get('received_hex'),'jaw_calibration_uncertain_receipt')
        command=bytes.fromhex(receipt['sent_hex']).decode('ascii')
        need(command==r['plan'][i]['command'],'jaw_calibration_plan_receipt_mismatch')
        if r.get('mode')=='supervised_jaw_calibration':
            expected=packet([(5,r['target_pwm'][5])],3000).decode()
            need(command==expected,'jaw_calibration_non_jaw_packet')
        for offset in range(1,len(command)-1,15):
            word=command[offset:offset+15];state[int(word[1:4])]=int(word[5:9])
    need(state[:5]==neutral[:5] and type(state[5]) is int and 1000<=state[5]<=1600,
         'jaw_calibration_not_stationary_neutral')
    if r.get('mode')!='supervised_jaw_calibration':need(state==neutral,'jaw_calibration_initial_jaw_not_open')
    return state


def reference_record(path,neutral):
    """Keep the original validated scene fixed across the measurement chain."""
    path=Path(path);history_state(path,neutral)
    r=json.loads(path.read_text(encoding='utf-8'))
    while r.get('mode')=='supervised_jaw_calibration':
        path=path.parent/(r['source_task_id']+'.json')
        need(hashlib.sha256(path.read_bytes()).hexdigest()==r['source_report_sha256'],
             'jaw_calibration_parent_changed')
        r=json.loads(path.read_text(encoding='utf-8'))
    return path,r


def extended_step(extension,prior_path,start,value):
    """An explicit bound scope permits only small, measured closing steps."""
    need(isinstance(extension,dict) and extension.get('kind')=='operator-approved-jaw-extension-v1'
         and isinstance(extension.get('operator_statement'),str) and bool(extension['operator_statement'].strip())
         and type(extension.get('max_pwm')) is int and extension['max_pwm']==1600,
         'jaw_calibration_extension_authorization_required')
    baseline=extension.get('baseline_task_id');digest=extension.get('baseline_sha256')
    need(isinstance(baseline,str) and baseline and all(c.isalnum() or c=='-' for c in baseline)
         and isinstance(digest,str) and len(digest)==64,'jaw_calibration_extension_baseline_invalid')
    cursor=Path(prior_path)
    for _ in range(32):
        r=json.loads(cursor.read_text(encoding='utf-8'))
        if cursor.stem==baseline:
            need(hashlib.sha256(cursor.read_bytes()).hexdigest()==digest and
                 r.get('status')=='jaw_calibration_command_completed' and r['target_pwm'][5]==1500,
                 'jaw_calibration_extension_baseline_changed')
            break
        parent=r.get('source_task_id')
        need(r.get('mode')=='supervised_jaw_calibration' and isinstance(parent,str) and parent
             and all(c.isalnum() or c=='-' for c in parent),'jaw_calibration_extension_unrelated_history')
        cursor=cursor.parent/(parent+'.json')
        need(hashlib.sha256(cursor.read_bytes()).hexdigest()==r['source_report_sha256'],
             'jaw_calibration_parent_changed')
    else:raise ValueError('jaw_calibration_extension_baseline_missing')
    need(max(start[5],value['target_pwm'])<=extension['max_pwm'],'jaw_calibration_extension_range_exceeded')
    delta=value['target_pwm']-start[5]
    limit=5 if value['gap_mm']<=4 else 10 if value['gap_mm']<=8 else 25
    need(delta<=0 or (value['gap_mm']>2 and delta<=limit),'jaw_calibration_extension_step_or_contact')


def plan(c,start,value,prior_path,extension=None):
    value=request(value)
    need(prior_path.stem==value['source_task_id'] and
         hashlib.sha256(prior_path.read_bytes()).hexdigest()==value['source_sha256'],'jaw_calibration_history_changed')
    need(start[:5]==c['known']['initial_pwm'][:5] and 0<abs(value['target_pwm']-start[5])<=100,
         'jaw_calibration_step_too_large_or_other_joint_changed')
    if max(start[5],value['target_pwm'])>1500:
        extended_step(extension,prior_path,start,value)
    else:
        need(value['target_pwm']<=max(p['close_pwm'] for p in c['profiles'].values()),
             'jaw_calibration_exceeds_existing_checked_closure')
    need(value['target_pwm']<=start[5] or value['gap_mm']>2,'jaw_calibration_close_to_contact')
    target=start[:5]+[value['target_pwm']]
    return [('jaw_calibration',packet([(5,value['target_pwm'])],3000),3000)],target
