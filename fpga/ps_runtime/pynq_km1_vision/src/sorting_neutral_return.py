"""Explicit empty-gripper return from a proven prepare/neutral pose only."""
import copy
import hashlib
import json
from itertools import combinations
from pathlib import Path
import time
import uuid
import cv2
import numpy as np
from .arm import packet
from .arm_route_preview import tip_rz
from .sorting_loop import Journal,need
from .sorting_known import bound_file,file_hash
from .sorting_scene import check_motion_layout
from .station_geometry import detect_markers
from .sorting_io import Motion
from .sorting_station_io import StationCamera,StationStatistics
from .sorting_unified_run import load_unified,BatchLease
from .sorting_relocated_run import motion_marker_limit,motion_shift_limit,integration_blockers
from openmv_gripper.km1_arm_bridge import validate_packet


def packet_targets(command):
    data=command.encode('ascii');validate_packet(data,[[500,2500]]*6)
    words=[command[i:i+15] for i in range(1,len(command)-1,15)]
    targets={int(word[1:4]):int(word[5:9]) for word in words}
    need(len(targets)==len(words),'neutral_return_duplicate_channel')
    return targets


def source_return_plan(c,start,prior):
    """Retrace a completed source approach; the only unknown is the empty jaw.

    A lost close receipt is not declared unwritten. No unknown arm/wrist command
    is accepted. Opening the operator-confirmed empty jaw resolves that single
    channel before any lift, using the original planned reverse approach.
    """
    names=['home','source_high','align_wrist','source_hover','source_mid','source_grip','close',
           'source_lift_mid','source_lift_hover','source_lift_high','front_high','restore_wrist_high','transfer_0']
    plan=prior.get('plan',[]);steps=prior.get('steps',[]);target=c['known']['initial_pwm']
    need(prior.get('mode')=='known_physical' and prior.get('status')=='needs_review' and
         prior.get('reason')=='ValueError:board_control_blocked' and len(steps)==7 and
         len(plan)>=len(names) and [p.get('step') for p in plan[:len(names)]]==names and
         [p.get('step') for p in steps]==names[:7] and steps[-1].get('status')=='may_be_in_flight' and
         not steps[-1].get('transport_receipt') and len(start)==6 and start[5] is None,
         'neutral_return_source_history_not_supported')
    poses=[packet_targets(p['command']) for p in plan[:len(names)]]
    prepare=c['known']['prepare_pwm']+target[4:]
    need(target[5]==1000 and poses[0]==dict(enumerate(prepare)) and set(poses[2])=={4} and
         all(set(poses[i])=={0,1,2,3} for i in (1,3,4,5,7,8,9,10,12)) and
         set(poses[6])=={5} and 1000<=poses[6][5]<=1600 and poses[11]=={4:target[4]} and
         poses[7]==poses[4] and poses[8]==poses[3] and poses[9]==poses[1] and
         poses[12]==dict(enumerate(prepare[:4])) and
         start[:4]==[poses[5][i] for i in range(4)] and start[4]==poses[2][4],
         'neutral_return_source_reverse_route_mismatch')
    result=[('open_confirmed_empty_jaw',packet([(5,1000)],3000),3000)]
    state=start[:5]+[1000];heights=[tip_rz(state)[1]]
    for i in (7,8,9,10,11,12):
        after=list(state)
        for channel,value in poses[i].items():after[channel]=value
        heights.extend(tip_rz([a+(b-a)*j/100 for a,b in zip(state,after)])[1] for j in range(101))
        result.append((names[i],packet(sorted(poses[i].items()),3000),3000));state=after
    need(state==prepare,'neutral_return_source_prepare_not_reached')
    home,height=neutral_plan(c,state)
    return result+home,min([height,*heights])


def no_uart_history(folder):
    """Only positive host no-write attestations may leave the last pose intact.

    Child reports need a matching parent manifest SHA and batch identity; an
    absent receipt or a zero step count alone never proves no command was sent.
    """
    folder=Path(folder);proven={}
    for path in folder.glob('*.json'):
        if '.' in path.stem or path.name in ('pending.json','batch-pending.json'):continue
        try:
            raw=path.read_bytes();report=json.loads(raw)
            if not (report.get('mode')=='known_physical' and report.get('task_id')==path.stem and
                    report.get('board_uart_write_attempted') is False and report.get('steps')==[]):continue
            children=[]
            for item in report.get('items',[]):
                task=item.get('task_id');need(isinstance(task,str) and Path(task).name==task and '.' not in task,'invalid_no_uart_child')
                child=folder/(task+'.json');body=child.read_bytes();record=json.loads(body)
                need(hashlib.sha256(body).hexdigest()==item.get('report_sha256') and
                     record.get('mode')=='known_physical' and record.get('task_id')==task and
                     record.get('batch_id')==report.get('batch_id')==report['task_id'] and
                     record.get('steps')==[] and record.get('board_uart_write_attempted') is not True,
                     'no_uart_child_attestation_mismatch')
                children.append((child,hashlib.sha256(body).hexdigest()))
            proven[path]=hashlib.sha256(raw).hexdigest();proven.update(children)
        except (OSError,ValueError,TypeError,KeyError):continue
    return proven


def confirmed_start(c,folder,legacy_proof=None,allow_jaw=False,allow_source=False):
    candidates=[path for path in Path(folder).glob('*.json') if '.' not in path.stem]
    selected=None;no_writes=no_uart_history(folder);skipped=[]
    for path in sorted(candidates,key=lambda p:p.stat().st_mtime_ns,reverse=True):
        if path.name in ('pending.json','batch-pending.json'):continue
        try:r=json.loads(path.read_text(encoding='utf-8-sig'))
        except (OSError,ValueError):continue
        if r.get('physical_motion_attempted') is True:
            if path in no_writes:
                skipped.append(dict(report=path.name,sha256=no_writes[path],basis='host_attested_no_uart_with_child_hashes'))
                continue
            need(r.get('mode') in ('known_physical','supervised_neutral_return','supervised_jaw_calibration'),
                 'neutral_return_unrecognized_motion_history')
            selected=(path,dict(r,pose_history_skipped_no_uart=skipped));break
    need(selected is not None,'neutral_return_no_known_pose')
    path,r=selected;state=[None]*6;confirmed=0
    if r.get('mode')=='supervised_jaw_calibration':
        from .jaw_calibration import history_state
        state=history_state(path,c['known']['initial_pwm'])
        need(allow_jaw or state==c['known']['initial_pwm'],'neutral_return_calibration_jaw_not_open')
        return state,path,r
    for index,step in enumerate(r['steps']):
        receipt=step.get('transport_receipt',{})
        if not (receipt.get('echo_matched') is True and receipt.get('completed_movement_wait') is True):
            if allow_source and index==6 and confirmed==6:
                uncertain_jaw=list(state);uncertain_jaw[5]=None
                source_return_plan(c,uncertain_jaw,r)
                return uncertain_jaw,path,r
            # Legacy failure has no blocked-frame file: this exact report must
            # have a separately bound source/receipt audit proving no write.
            proven=(legacy_proof is not None and index==1 and confirmed==1 and index==len(r['steps'])-1 and
                legacy_proof.get('kind')=='reviewed-prewrite-neutral-return-start-v1' and
                legacy_proof.get('report_sha256')==file_hash(path) and
                legacy_proof.get('task_id')==r['task_id'] and
                legacy_proof.get('unwritten_step')==step['step']=='source_high' and
                legacy_proof.get('reason')==r.get('reason') and
                r.get('reason') in ('ValueError:motion_requires_three_fixed_markers','ValueError:motion_requires_two_fixed_markers') and
                legacy_proof.get('motion_command_written') is False)
            need(proven,'neutral_return_last_command_uncertain');break
        command=bytes.fromhex(receipt['sent_hex'])
        need(receipt['received_hex']==receipt['sent_hex'] and
             command.decode('ascii')==r['plan'][index]['command'],'neutral_return_receipt_mismatch')
        validate_packet(command,[[500,2500]]*6)
        text=command.decode('ascii')
        for offset in range(1,len(text)-1,15):
            word=text[offset:offset+15];state[int(word[1:4])]=int(word[5:9])
        confirmed+=1
    neutral=c['known']['initial_pwm'];prepare=c['known']['prepare_pwm']+neutral[4:]
    need(confirmed and state in (neutral,prepare),'neutral_return_requires_known_empty_prepare_or_neutral')
    need(state[5]==neutral[5] and state[4]==neutral[4],'neutral_return_gripper_or_wrist_not_neutral')
    return state,path,r


def neutral_plan(c,start):
    target=c['known']['initial_pwm'];prepare=c['known']['prepare_pwm']+target[4:]
    need(start in (target,prepare),'neutral_return_unknown_start')
    need(any(p['retreat_pwm'][-2:]==[prepare[:4],target[:4]] for p in c['profiles'].values()),
         'neutral_return_not_original_retreat_segment')
    heights=[tip_rz([a+(b-a)*i/100 for a,b in zip(start,target)])[1] for i in range(101)]
    need(min(heights)>=min(tip_rz(start)[1],tip_rz(target)[1])-.75,'neutral_return_nominal_height_floor')
    data=packet(list(enumerate(target)),5000)
    need(validate_packet(data,[[500,2500]]*6)==5000,'invalid_neutral_return_packet')
    return [('return_neutral',data,5000)],min(heights)


def calibration_return_plan(c,start):
    """Open an empty calibrated jaw at stationary neutral, then reuse home.

    Caller must first validate the completed, hash-linked calibration history.
    These are the same <=100 PWM / 3000 ms opening steps used by calibration.
    """
    target=c['known']['initial_pwm']
    need(len(start)==6 and start[:5]==target[:5] and target[5]==1000 and
         type(start[5]) is int and 1000<=start[5]<=1600,
         'neutral_return_calibration_start_outside_checked_range')
    result=[];jaw=start[5]
    while jaw>target[5]:
        jaw=max(target[5],jaw-100)
        data=packet([(5,jaw)],3000)
        need(validate_packet(data,[[500,2500]]*6)==3000,'invalid_neutral_return_jaw_packet')
        result.append(('open_empty_jaw',data,3000))
    home,height=neutral_plan(c,target)
    return result+home,height


class FrozenReturnGuard:
    def __init__(self,reference,camera,station,layout,minimum,shift):
        self.reference=reference;self.camera=camera;self.minimum=minimum
        allowed={m['id'] for m in station['markers']}
        self.known={d['id']:np.asarray(d['corners_px'],float) for d in detect_markers(reference,camera) if d['id'] in allowed}
        self.layout=dict(layout,max_layout_shift_px=shift,search_margin_px=max(layout['search_margin_px'],int(shift)))
        need(len(self.known)>=minimum,'neutral_return_reference_markers_missing')
    def check(self,image):
        rows=[d for d in detect_markers(image,self.camera) if d['id'] in self.known]
        need(len({d['id'] for d in rows})==len(rows),'motion_duplicate_marker_ids')
        need(len(rows)>=self.minimum,'neutral_return_fixed_markers_missing')
        shift=max(float(np.linalg.norm(np.asarray(d['corners_px'])-self.known[d['id']],axis=1).max()) for d in rows)
        need(shift<=self.layout['max_layout_shift_px'],'neutral_return_fixed_board_moved')
        height,width=image.shape[:2]
        for pair in combinations(rows,2):
            rectangles=[]
            for d in pair:
                p=self.known[d['id']];low=np.floor(p.min(0)-4).astype(int);high=np.ceil(p.max(0)+5).astype(int)
                if not ((low>=0).all() and (high<=[width,height]).all()):break
                rectangles.append([*map(int,low),*map(int,high)])
            if len(rectangles)!=2:continue
            try:static=check_motion_layout(self.reference,image,dict(self.layout,static_rois=rectangles,image_size=[width,height]))
            except ValueError:continue
            return dict(marker_ids=[d['id'] for d in rows],minimum_fixed_markers=self.minimum,
                max_displacement_px=shift,static=static,scope='known_empty_gripper_joint_return_only',camera_pose_recomputed=False)
        raise ValueError('neutral_return_two_photometric_markers_required')


def return_neutral(binding,execute=False,confirmed=False,jaw_step=None):
    bundle,policy=load_unified(binding);b,c,base,cam,station,reg,scene,files=bundle
    need(any(Path(r['path']).name=='sorting_neutral_return.py' for r in b['implementation_sources']),
         'neutral_return_implementation_not_bound')
    folder=(base/c['journal_dir']).resolve();lease=(base/c['known']['device_lease']).resolve()
    stops=[(base/p).resolve() for p in c['known']['stop_files']]
    need(not any(p.exists() for p in stops),'operator_stop_requested')
    need(not (folder/'pending.json').exists() and not (folder/'batch-pending.json').exists(),
         'neutral_return_resolve_previous_task_first')
    need(not integration_blockers(b,c,base,files),'neutral_return_station_not_ready')
    proof=json.loads(files['neutral_return_start_proof'].read_text(encoding='utf-8-sig')) if 'neutral_return_start_proof' in files else None
    # Closed jaws are accepted only through the fully validated calibration
    # history; arbitrary poses and uncertain receipts still fail above.
    start,prior_path,prior=confirmed_start(c,folder,proof,allow_jaw=True,allow_source=jaw_step is None)
    target=c['known']['initial_pwm'];mode='supervised_neutral_return';success='neutral_return_command_completed'
    recovering_jaw=jaw_step is None and prior.get('mode')=='supervised_jaw_calibration' and start!=target
    recovering_source=jaw_step is None and start[5] is None
    if jaw_step is not None:
        from .jaw_calibration import plan as jaw_plan
        need(any(Path(r['path']).name=='jaw_calibration.py' for r in b['implementation_sources']),
             'jaw_calibration_implementation_not_bound')
        plan,target=jaw_plan(c,start,jaw_step,prior_path,extension=b.get('jaw_calibration_extension'));height=tip_rz(start)[1]
        mode='supervised_jaw_calibration';success='jaw_calibration_command_completed'
    elif recovering_source:plan,height=source_return_plan(c,start,prior)
    elif recovering_jaw:plan,height=calibration_return_plan(c,start)
    else:plan,height=neutral_plan(c,start)
    if not execute:return dict(status='jaw_calibration_ready' if jaw_step is not None else 'neutral_return_ready',
        start_pwm=start,target_pwm=target,duration_ms=sum(row[2] for row in plan),
        command_count=len(plan),empty_jaw_opening_steps=1 if recovering_source else len(plan)-1 if recovering_jaw else 0,
        nominal_min_height_mm=height,physical_motion=False)
    need(confirmed,'neutral_return_scene_confirmation_required')
    # No new target mapping is needed for the previously checked final joint
    # segment. Keep the exact saved camera/marker reference and original guards.
    from .monitor_tasks import recorded_frame
    reference_path,reference_report=prior_path,prior
    if prior.get('mode')=='supervised_jaw_calibration':
        from .jaw_calibration import reference_record
        reference_path,reference_report=reference_record(prior_path,c['known']['initial_pwm'])
    need(reference_report.get('evidence'),'neutral_return_reference_image_missing')
    _,obs=recorded_frame(folder,reference_report['evidence'][-1],pixels=True)
    reference=cv2.imread(str(folder/reference_report['evidence'][-1]['image']))
    frozen=FrozenReturnGuard(reference,cam,station,c['layout'],motion_marker_limit(b),motion_shift_limit(b))
    started=time.monotonic()
    task=('jaw-' if jaw_step is not None else 'neutral-')+uuid.uuid4().hex;report=dict(task_id=task,mode=mode,
        status='observing',steps=[],evidence=[],source_task_id=prior['task_id'],source_report_sha256=file_hash(prior_path),
        start_pwm=start,target_pwm=target,nominal_min_height_mm=height,
        plan=[dict(step=n,command=d.decode(),duration_ms=ms) for n,d,ms in plan],
        physical_motion_attempted=False,physical_pose_verified=False,automatic_retry=False,servo_angle_readback=False)
    report['fixed_reference']=dict(task_id=reference_report['task_id'],report_sha256=file_hash(reference_path))
    report['pose_history_skipped_no_uart']=prior.get('pose_history_skipped_no_uart',[])
    if recovering_jaw:
        report['empty_jaw_recovery']=dict(kind='completed-calibration-opening-v1',
            opening_steps=len(plan)-1,stationary_other_joints=True,requires_confirmed_empty_gripper=True)
    if recovering_source:
        report['empty_jaw_recovery']=dict(kind='completed-source-approach-retrace-v1',
            unknown_channel=5,unknown_close_outcome_preserved=True,opening_steps=1,
            original_reverse_approach=True,requires_confirmed_empty_gripper=True)
    import urllib.request
    lease_budget=40+sum(ms/1000+.5 for _,_,ms in plan[1:])
    with Journal(folder) as journal, BatchLease(lease,stops,time.monotonic()+lease_budget) as owner:
        need(journal.pending() is None and not (folder/'batch-pending.json').exists(),'neutral_return_resolve_previous_task_first')
        need(file_hash(prior_path)==report['source_report_sha256'],'neutral_return_history_changed')
        _,latest,_=confirmed_start(c,folder,proof,allow_jaw=True,allow_source=jaw_step is None)
        need(latest==prior_path,'neutral_return_history_changed')
        health=json.load(urllib.request.urlopen(c['statistics']['url']+'/health',timeout=3))
        capture=copy.deepcopy(c);regions=health['regions'];capture['source_roi']=regions['source']
        capture['layout']['static_rois']=[regions['static_a'],regions['static_b']]
        for p in capture['profiles'].values():p['bin_roi']=regions[p['bin_id']]
        capture['statistics']['regions']=regions
        statistics=StationStatistics(capture)
        camera=StationCamera(capture,None,statistics,lossless=False)
        # Stream IDs identify service lifetimes, not physical camera poses.
        # Fresh station identity checks freeze this attempt's stream; the
        # historical image still supplies all marker/photometric guards.
        report['camera_stream_binding']=dict(historical_stream_id=obs['source_stream_id'],
            active_stream_id=statistics.stream,renewed=statistics.stream!=obs['source_stream_id'],
            basis='current_station_identity_and_frozen_reference_guards')
        class ReturnMotion(Motion):
            def check_stop(self):super().check_stop();owner.check()
        motion=ReturnMotion(c['arm'],stops[0]);attempted=False
        def guard(phase):
            owner.check();row=camera.raw(c['frame_timeout_s'])
            need(row['observation']['source_stream_id']==statistics.stream,'motion_camera_stream_changed')
            need(0<=time.monotonic()*1000-row['observation']['capture_started_ms']<=c['max_age_ms'],'neutral_return_frame_expired')
            size=list(row['image'].shape[1::-1])
            need(size==c['image_size'],'neutral_return_frame_size_changed')
            need(row['observation'].get('image_size',size)==size,'neutral_return_observation_size_mismatch')
            row=dict(row,observation=dict(row['observation'],image_size=size))
            report['evidence'].append(journal.capture(task,phase,row))
            return frozen.check(row['image'])
        try:
            report['before_guard']=guard('before')
            report['status']='motion_intent_reserved';journal.save('pending.json',report);attempted=True
            motion.open(plan)
            for command_index,(name,data,duration) in enumerate(plan):
                entry=dict(step=name,status='may_be_in_flight',duration_ms=duration)
                report['steps'].append(entry);journal.save('pending.json',report)
                earlier_motion=report['physical_motion_attempted']
                report['physical_motion_attempted']=True
                def prewrite():
                    try:return guard('prewrite' if len(plan)==1 else 'prewrite_'+str(command_index+1))
                    except Exception:
                        entry.update(status='blocked_before_write',motion_command_written=False)
                        report['physical_motion_attempted']=earlier_motion
                        raise
                motion.send(data,duration,guard=prewrite)
                motion.wait(duration/1000+.5);entry.update(status='sent_and_waited_unverified',
                    transport_receipt=copy.deepcopy(motion.last_receipt))
                journal.save('pending.json',report)
            motion.close()
            report['after_guard']=guard('after')
            report['status']=success
            if jaw_step is not None:
                report['jaw_calibration']=dict(previous_measurement=jaw_step,gap_mm=None,
                    measurement_required=True,other_joints_unchanged=True,physical_grip_verified=False)
                if max(start[5],target[5])>1500:
                    report['jaw_calibration']['extension']=copy.deepcopy(b['jaw_calibration_extension'])
        except Exception as exc:
            report.update(status='needs_review' if attempted else 'rejected',reason=type(exc).__name__+':'+str(exc))
        finally:
            motion.close();report['elapsed_s']=time.monotonic()-started;journal.save(task+'.json',report)
            if attempted:
                if report['status']==success:journal.clear()
                else:journal.save('pending.json',report)
    return report
