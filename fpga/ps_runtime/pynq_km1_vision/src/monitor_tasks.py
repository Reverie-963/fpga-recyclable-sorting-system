"""Read-only view of the current single-item ledger. Never imports an executor."""
import base64
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import threading
import time

from .monitor_control import ROOT,read,geometry_summary,point,number

TASK_NAME=re.compile(r'[a-zA-Z0-9_-]{1,160}')
TERMINAL={'completed_visual_evidence','needs_review','rejected','preview_only','empty_scene','simulated_visual_pass','neutral_return_command_completed'}


def local_file(folder,name):
    if not isinstance(name,str) or Path(name).name!=name:raise ValueError('invalid_evidence_name')
    path=(folder/name).resolve()
    if not path.is_relative_to(folder.resolve()):raise ValueError('invalid_evidence_path')
    return path


def recorded_frame(folder,item,pixels=False):
    """A picture is usable only with its ledger and observation hashes."""
    image=local_file(folder,item['image']);meta=read(local_file(folder,item['observation']))
    if image.stat().st_size>8_500_000:raise ValueError('oversized_image')
    raw=image.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    if digest!=item['png_sha256'] or digest!=meta.get('png_sha256'):
        raise ValueError('record_image_hash_mismatch')
    observation=meta['observation']
    if pixels:
        import cv2
        import numpy as np
        decoded=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
        if decoded is None or hashlib.sha256(decoded.tobytes()).hexdigest()!=meta.get('decoded_bgr_sha256'):
            raise ValueError('record_pixels_mismatch')
        size=[decoded.shape[1],decoded.shape[0]]
        if 'image_size' not in observation:
            # Early motion-guard records omitted dimensions. Both hashes above
            # bind the original decoded pixels; derive metadata in memory only.
            frame=observation.get('source_frame_id')
            valid_frame=(type(frame) is int and frame>=0) or (
                isinstance(frame,str) and frame.isascii() and frame.isdecimal() and
                len(frame)<=20 and str(int(frame))==frame)
            bound_frame=observation.get('station_vision',{}).get('frame_id',frame)
            if (not 1<=size[0]<=640 or not 1<=size[1]<=480 or not valid_frame or
                    str(bound_frame)!=str(frame) or
                    not isinstance(observation.get('source_stream_id'),str) or
                    not observation['source_stream_id']):
                raise ValueError('record_image_size_mismatch')
            observation=dict(observation,image_size=size,
                image_size_source='hash_verified_decoded_legacy_record')
        if size!=observation.get('image_size'):
            raise ValueError('record_image_size_mismatch')
    return raw,observation


def commands(record):
    planned=record.get('plan',[]);actual=record['steps'];output=[]
    if not all(isinstance(x,dict) for x in planned+actual):raise ValueError('invalid_command_entries')
    for index in range(min(80,max(len(planned),len(actual)))):
        target=planned[index] if index<len(planned) else {}
        entry=actual[index] if index<len(actual) else {}
        intended=target.get('command','');receipt=entry.get('transport_receipt',{})
        if not isinstance(receipt,dict):raise ValueError('invalid_receipt')
        state='intent_only' if entry else 'planned';sent=None;echo=False;wait=False;written=0
        try:
            encoded=receipt.get('sent_hex','')
            if not isinstance(encoded,str) or len(encoded)>1600:raise ValueError('oversized_command')
            raw=bytes.fromhex(encoded);sent=raw.decode('ascii') if raw else None
            if sent:
                written=len(raw);state='written'
                echo=receipt.get('echo_matched') is True and receipt.get('received_hex')==encoded
                wait=receipt.get('completed_movement_wait') is True
                if echo:state='echo_matched'
            if sent and (sent!=intended or entry.get('step')!=target.get('step')):
                state='identity_mismatch';echo=False;wait=False
        except (ValueError,UnicodeError,TypeError):state='identity_mismatch'
        output.append(dict(index=index+1,step=str(entry.get('step',target.get('step','')))[:120],
            planned_command=intended[:800] if isinstance(intended,str) else '',command=(sent or '')[:800],
            state=state,duration_ms=target.get('duration_ms'),bytes_written=written,
            completed_wait=wait,echo_matched=echo))
    return output


def project(path,record):
    choice=record.get('choice',{});provenance=record.get('statistics',{});rows=commands(record)
    if not isinstance(choice,dict) or not isinstance(provenance,dict):raise ValueError('invalid_task_facts')
    physical=record.get('mode') in ('known_physical','supervised_neutral_return')
    observation={};evidence=record.get('evidence',[])
    before=next((x for x in evidence if x.get('image')==record['task_id']+'.before.png'),None)
    if before:
        try:
            _,observation=recorded_frame(path.parent,before)
        except (OSError,ValueError,TypeError,KeyError):pass
    geometry_backend=provenance.get('geometry_backend',provenance.get('backend','unknown'))
    geometry=geometry_summary(choice.get('geometry'),dict(backend=geometry_backend,
        each_result_cpu_compared=provenance.get('each_result_cpu_compared') is True))
    verification=record.get('verification',{});passed=verification.get('status')=='visual_evidence_pass'
    frames=verification.get('frames',[]) if passed else []
    source=[f.get('source',{}).get('erased') for f in frames]
    gains=[f.get('bin',{}).get('beyond') for f in frames]
    review=None;review_path=path.parent/(record['task_id']+'.review.json')
    if review_path.is_file():
        value=read(review_path);snapshot=value.get('pending',{})
        required=('task_id','config_sha256','choice','plan','steps','evidence','statistics','status')
        if (value.get('task_id')==record['task_id'] and all(k in snapshot and snapshot[k]==record.get(k) for k in required)
            and isinstance(value.get('observation'),str)):
            review=dict(text=value['observation'][:1500],outcome=value.get('outcome'),binding='matching_task_snapshot')
    operator_path=path.parent/(record['task_id']+'.operator.json')
    if operator_path.is_file():
        value=read(operator_path)
        if value.get('task_id')==record['task_id'] and value.get('report_sha256')==hashlib.sha256(path.read_bytes()).hexdigest():
            review=dict(text=str(value.get('operator_reply',''))[:1500],binding='exact_report_hash',outcome='operator_observation')
    supplemental=[]
    for candidate in sorted(path.parent.glob('recheck-*.json'),key=lambda p:p.stat().st_mtime,reverse=True)[:20]:
        try:
            value=read(candidate)
            if (value.get('original_task_id')!=record['task_id'] or value.get('physical_motion') is not False or
                value.get('original_outcome_unchanged') is not True):continue
            items=value.get('frames',[])
            if len(items)!=3 or len({x.get('image') for x in items})!=3:continue
            observations=[recorded_frame(path.parent,item)[1] for item in items]
            if (len({x.get('source_frame_id') for x in observations})!=3 or
                any(x.get('source_frame_id') is None or x.get('model_version')!=observation.get('model_version') or
                    x.get('source_stream_id')!=provenance.get('stream_id') for x in observations)):continue
            supplemental.append(dict(status=value.get('status'),frames=len(items),min_confidence=value.get('min_confidence'),
                physical_motion=False,original_outcome_unchanged=True,source_record=candidate.name,
                semantic_state=value.get('semantic',{}).get('status'),rgb_state=value.get('board_rgb',{}).get('status')))
        except (OSError,ValueError,TypeError,KeyError):continue
    rgb=None;rgb_path=path.parent/(record['task_id']+'.board-vision.json')
    if rgb_path.is_file():
        value=read(rgb_path);p=value.get('provenance',{});baseline=value.get('baseline',{})
        required=('baseline_frame_id','baseline_bgr_sha256','stream_id','config_sha256')
        if (all(p.get(k)==provenance.get(k) for k in ('assets','config_sha256','stream_id')) and
            p.get('backend')=='pl_board_local' and all(k in baseline and baseline[k]==record.get('motion_baseline',{}).get(k) for k in required)):
            rgb=dict(status=value.get('status'),frames=len(value.get('frames',[])),backend='pl_board_local')
    return dict(available=True,task_id=record['task_id'],schema_version='sorting-monitor-task-v2',
        raw_status=record['status'],record_kind='physical' if physical else record.get('mode','unknown'),
        finished=record['status'] in TERMINAL,reason=str(record.get('reason',''))[:600],
        label=choice.get('label'),bin_id=choice.get('bin_id'),motion_profile=record.get('motion_profile','qualified'),
        elapsed_s=record.get('elapsed_s') if number(record.get('elapsed_s')) else None,
        recorded_at=datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat(),
        source_record=path.name,record_age_s=max(0,time.time()-path.stat().st_mtime),
        model_version=observation.get('model_version'),runtime_id=observation.get('runtime_id'),
        geometry=geometry,statistics_backend=provenance.get('backend'),
        pick_pixel=point(choice.get('pick_pixel'),2),pick_card_xy_mm=point(choice.get('pick_card_xy_mm'),2),
        grip_xyz_mm=point(choice.get('grip_xyz_mm'),3),grip_xy_mm=point(choice.get('grip_xy_mm'),2),
        hover_xyz_mm=point(choice.get('hover_xyz_mm'),3),hover_xy_mm=point(choice.get('hover_xy_mm'),2),
        grasp_source_frame=observation.get('source_frame_id'),grasp_image_available=bool(observation and before),
        commands=rows,planned_commands=len(record.get('plan',[])),
        written_commands=sum(bool(x['bytes_written']) for x in rows) if physical else 0,
        echoed_commands=sum(x['echo_matched'] for x in rows) if physical else 0,
        waited_commands=sum(x['completed_wait'] for x in rows) if physical else 0,
        visual_verification=dict(state='passed' if passed else 'not_confirmed',frames=len(frames),
            source_erased_min=min(source) if source and all(number(v) for v in source) else None,
            correct_bin_gain_min=min(gains) if gains and all(number(v) for v in gains) else None),
        board_verification=rgb,operator_review=review,supplemental=supplemental,
        physical_success_verified=False,grasp_success_verified=False,readonly=True,automatic_motion_allowed=False)


class TaskTelemetry:
    def __init__(self,folder=None,refresh_s=2):
        self.folder=Path(folder) if folder is not None else ROOT/'local-only/sorting-known-ledger'
        self.refresh_s=refresh_s;self.cache={};self.lock=threading.Lock()

    def status(self,health,task_id=None):
        with self.lock:
            now=time.monotonic();entry=self.cache.get(task_id)
            if entry is None or now-entry[0]>=self.refresh_s:
                try:data=self.scan(task_id)
                except (OSError,ValueError,KeyError,TypeError):data=dict(available=False,reason='record_unavailable')
                self.cache[task_id]=(now,data)
                if len(self.cache)>45:self.cache={task_id:self.cache[task_id]}
            result=dict(self.cache[task_id][1])
        result['device_lease_present']=(ROOT/'local-only/arm-device-COM9.lock').exists()
        if result.get('available'):
            result['model_matches_monitor']=result.get('model_version')==health.get('model_sha256') if result.get('model_version') else None
            result['following_latest']=not bool(task_id)
        return result

    def scan(self,task_id):
        if task_id and not TASK_NAME.fullmatch(task_id):raise ValueError('invalid_task')
        paths=[p for p in self.folder.glob('*.json') if p.name=='pending.json' or TASK_NAME.fullmatch(p.stem)]
        paths=sorted(paths,key=lambda p:p.stat().st_mtime,reverse=True)[:150]
        history=[];seen=set();selected=None;invalid_newest=False
        for path in paths:
            try:
                r=read(path)
                if (not isinstance(r,dict) or not isinstance(r.get('task_id'),str) or not TASK_NAME.fullmatch(r['task_id']) or
                    path.name not in ('pending.json',r['task_id']+'.json') or not isinstance(r.get('steps'),list) or
                    not isinstance(r.get('plan',[]),list) or not isinstance(r.get('status'),str)):continue
                if not isinstance(r.get('statistics',{}),dict) or not isinstance(r.get('choice',{}),dict):continue
                if r.get('statistics',{}).get('backend') not in (None,'pl_board_local'):continue
                if r['task_id'] in seen:continue
                seen.add(r['task_id']);history.append(dict(task_id=r['task_id'],label=r.get('choice',{}).get('label'),
                    raw_status=r['status'],motion_profile=r.get('motion_profile','qualified'),
                    statistics_backend=r.get('statistics',{}).get('backend'),
                    recorded_at=datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat()))
                if selected is None and (not task_id or task_id==r['task_id']):selected=(path,r)
            except (OSError,ValueError,KeyError,TypeError):
                if selected is None and not history:invalid_newest=True
        if selected and (task_id or not invalid_newest):
            return dict(project(*selected),records=history[:40])
        return dict(available=False,reason='record_write_in_progress' if invalid_newest else 'no_execution_record',records=history[:40])

    def grasp_frame(self,health,task_id=None):
        status=self.status(health,task_id)
        if not status.get('available') or not status.get('pick_pixel'):
            return dict(available=False,reason='no_bound_grasp_point')
        try:
            record=read(self.folder/status['source_record'])
            if record.get('task_id')!=status['task_id'] or record.get('choice',{}).get('pick_pixel')!=status['pick_pixel']:
                raise ValueError('selected_record_changed')
            item=next(x for x in record.get('evidence',[]) if x.get('image')==record['task_id']+'.before.png')
            raw,observation=recorded_frame(self.folder,item,pixels=True)
            if observation.get('model_version')!=status.get('model_version'):raise ValueError('model_changed')
            x,y=status['pick_pixel'];size=observation['image_size']
            if not 0<=x<size[0] or not 0<=y<size[1]:raise ValueError('point_outside_image')
            return dict(available=True,image_base64=base64.b64encode(raw).decode('ascii'),image_mime='image/png',
                image_size=size,pixel=status['pick_pixel'],source_task_id=status['task_id'],
                source_frame_id=observation.get('source_frame_id'),current_live_frame=False,control_authorized=False)
        except (OSError,ValueError,KeyError,TypeError,StopIteration):
            return dict(available=False,reason='original_pixels_not_found')


def current_workflow(path=None):
    """Expose a small allowlist; never return machine configuration or tokens."""
    try:
        c=read(Path(path) if path else ROOT/'pynq_km1_vision/config/sorting_known.local.json')
        if c.get('statistics',{}).get('backend')!='pl_board_local':return dict(available=False)
        if not number(c.get('min_confidence')) or not 0<=c['min_confidence']<=1:return dict(available=False)
        return dict(available=True,min_confidence=c['min_confidence'],motion_profile=c['known']['motion_profile'],
            geometry_backend='pc_numpy',rgb_backend='pl_board_local',actuator='KM1 / ESP32',
            model_sha256=c.get('model_version'),runtime_id=c.get('runtime_id'),
            readonly=True,automatic_motion_allowed=False)
    except (OSError,ValueError,KeyError,TypeError):return dict(available=False)
