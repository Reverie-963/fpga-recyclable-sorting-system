"""Read-only projection of existing physical execution journals. No hardware I/O."""
from datetime import datetime, timezone
import base64
import hashlib
import json
import math
from pathlib import Path
import threading
import time

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    if path.stat().st_size > 2_000_000:
        raise ValueError('oversized_record')
    return json.loads(path.read_text(encoding='utf-8-sig'))


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def point(value, length):
    return list(value) if isinstance(value, list) and len(value) == length and all(number(v) for v in value) else None


def grasp_evidence(trial, plan):
    """Support legacy captures and newer hash-bound lossless before frames."""
    prefix=plan.get('evidence_prefix')
    if isinstance(prefix,str) and Path(prefix).name==prefix:
        path=trial/(prefix+'before-2.json')
        return (read(path),None) if path.is_file() else ({},None)
    reference=plan.get('preflight_reference_path')
    if not isinstance(reference,str):return {},None
    base=trial.parent.resolve();image=(base/reference).resolve()
    if (not image.is_relative_to(base) or image.name!='before-2.png' or not image.is_file() or
        image.stat().st_size>8_500_000 or
        hashlib.sha256(image.read_bytes()).hexdigest()!=plan.get('preflight_reference_sha256')):
        return {},None
    path=image.with_suffix('.json')
    if not path.is_file():return {},None
    evidence=read(path)
    if (evidence.get('lossless_png_sha256')!=plan['preflight_reference_sha256'] or
        evidence.get('observation',{}).get('model_version')!=plan.get('model_version')):
        return {},None
    return evidence,image


def geometry_summary(stats, provenance):
    if not isinstance(stats, dict) or stats.get('status') != 1:
        return None
    required = ('m00', 'm10', 'm01', 'min_col', 'max_col', 'min_row', 'max_row')
    if (not all(type(stats.get(k)) is int and 0<=stats[k]<2**64 for k in required) or stats['m00'] <= 0 or
        not 0<=stats['min_col']<=stats['max_col']<640 or not 0<=stats['min_row']<=stats['max_row']<480):
        return None
    area = stats['m00']
    x, y = stats['m10']/area, stats['m01']/area
    angle = None
    if all(number(stats.get(k)) for k in ('m20', 'm02', 'm11')):
        xx, yy, xy = stats['m20']/area-x*x, stats['m02']/area-y*y, stats['m11']/area-x*y
        if math.hypot(2*xy,xx-yy)>1e-9:
            angle = math.degrees(.5*math.atan2(2*xy, xx-yy)) % 180
    return dict(area_px=area, centroid_px=[x,y], axis_deg=angle,
        bbox_inclusive_px=[stats[k] for k in ('min_col','min_row','max_col','max_row')],
        job_id=stats.get('job_id'), backend=provenance.get('backend', 'unknown'),
        cpu_compared=provenance.get('each_result_cpu_compared') is True,
        moments={k:stats[k] for k in ('m00','m10','m01','m20','m02','m11') if number(stats.get(k))})


def project_record(path, record, depth=0):
    """Bind external plans by hash and expose only explicitly recorded facts."""
    trial = path.parent.parent if path.parent.name == 'journal' else path.parent
    plan, bound = None, False
    automatic = isinstance(record.get('choice'), dict) and isinstance(record.get('plan'), list)
    if automatic:
        plan, bound = record, True
    elif isinstance(record.get('plan_sha256'), str):
        candidate = trial / 'plan.json'
        if candidate.is_file() and candidate.stat().st_size <= 2_000_000:
            if hashlib.sha256(candidate.read_bytes()).hexdigest() == record['plan_sha256']:
                plan = read(candidate)
                bound = plan.get('task_id') == record['task_id']
    choice = plan.get('choice', {}) if automatic else (plan or {})
    provenance = (plan or {}).get('statistics' if automatic else 'statistics_provenance', (plan or {}).get('statistics',{}))
    jobs = (plan or {}).get('source_geometry_jobs', (plan or {}).get('geometry_jobs',[]))
    stats = choice.get('geometry') if automatic else (jobs[-1] if isinstance(jobs,list) and jobs else None)
    if not bound:
        choice, stats, provenance = {}, None, {}
    if automatic:
        planned = plan['plan']
    else:
        stages = (plan or {}).get('stages', []) if bound else []
        planned = next((s.get('steps', []) for s in stages if s.get('name') == record.get('stage')), [])
    actual = record.get('steps', [])
    commands = []
    for i in range(min(max(len(planned),len(actual)),80)):
        target = planned[i] if i < len(planned) else {}
        entry = actual[i] if i < len(actual) else {}
        command = entry.get('command', target.get('command', ''))
        command = command[:800] if isinstance(command,str) else ''
        state = 'planned'
        written = entry.get('bytes_written')
        if type(written) is int and written > 0:
            full = command.isascii() and written == len(command.encode('ascii'))
            state = 'echo_matched' if full and entry.get('echo_matched') is True else ('written' if full else 'partial_write')
        elif entry.get('status') == 'sent_and_waited_unverified':
            state = 'sent_waited'
        elif entry.get('attempted') or entry.get('status') == 'may_be_in_flight':
            state = 'intent_only'
        if target and entry and (target.get('step') != entry.get('step') or
            target.get('command') and entry.get('command') and target['command'] != entry['command']):
            state = 'identity_mismatch'
        commands.append(dict(index=i+1, step=str(entry.get('step') or target.get('step') or record.get('stage') or '未命名步骤')[:120],
            command=command, state=state, duration_ms=entry.get('duration_ms',target.get('duration_ms')),
            bytes_written=written if type(written) is int else None))
    model = choice.get('model_version') if not automatic else None
    if automatic:
        # Observation evidence carries the model identity, not an inferred current model.
        evidence = record.get('evidence', [])
        if evidence and isinstance(evidence[0].get('observation'), str):
            name = evidence[0]['observation']
            if Path(name).name == name:
                try:model = read(trial/name)['observation']['model_version']
                except (KeyError,TypeError,ValueError,OSError):pass
    verification = record.get('verification') if automatic else None
    review_path = trial / 'final-review.json'
    if not automatic and bound and review_path.is_file():
        review = read(review_path)
        # The legacy review lacks task IDs: require one execution in this directory,
        # a complete command count and a later mtime; disclose this weaker linkage.
        executions = [p for p in path.parent.glob('*.json') if not p.name.endswith('.observation.json')]
        if (len(executions) == 1 and review_path.stat().st_mtime >= path.stat().st_mtime and
            review.get('steps') == len(actual) == len(planned) and
            review.get('task_id',record['task_id']) == record['task_id'] and
            review.get('plan_sha256',record['plan_sha256']) == record['plan_sha256'] and
            review.get('status') == 'supervised_trial_visual_evidence_pass'):
            verification = dict(status='visual_evidence_pass', frames=review.get('visual_evidence', []),
                                association='same_execution_directory')
    visual_pass = isinstance(verification,dict) and verification.get('status') == 'visual_evidence_pass'
    visual_frames = verification.get('frames',[]) if visual_pass else []
    source_erased = [r['source']['erased'] for r in visual_frames if number(r.get('source',{}).get('erased'))]
    bin_gain = [r['bin']['beyond'] for r in visual_frames if number(r.get('bin',{}).get('beyond'))]
    feedback = None
    feedback_path = trial/'operator-feedback.json'
    if bound and feedback_path.is_file():
        value = read(feedback_path)
        summary_path = trial/'execution-summary.json'
        summary_matches = (not value.get('execution_summary_sha256') or
            summary_path.is_file() and hashlib.sha256(summary_path.read_bytes()).hexdigest() == value['execution_summary_sha256'])
        if summary_matches and value.get('plan_sha256') == record.get('plan_sha256') and isinstance(value.get('text'),str):
            feedback = value['text'][:600]
    observation_path = path.with_suffix('.observation.json')
    operator_lift = False
    if bound and observation_path.is_file():
        value = read(observation_path)
        if (value.get('task_id') == record['task_id'] and value.get('plan_sha256') == record.get('plan_sha256') and
            value.get('execution_sha256') == hashlib.sha256(path.read_bytes()).hexdigest()):
            if isinstance(value.get('text'),str):feedback = value['text'][:600]
            operator_lift = value.get('can_left_table') is True and value.get('no_drop_reported') is True
    point_px = choice.get('pick_pixel') if bound else None
    evidence,_ = grasp_evidence(trial,choice) if bound and not automatic else ({},None)
    evidence_frame = evidence.get('observation',{}).get('source_frame_id')
    evidence_label = None
    instance_index=evidence.get('selected_instance_index')
    instances=evidence.get('observation',{}).get('instances',[])
    if type(instance_index) is int and 0<=instance_index<len(instances):
        evidence_label=instances[instance_index].get('label')
    kind = record.get('mode', 'physical' if record.get('serial_opened') is True else 'execution_intent')
    physical = kind == 'physical'
    closed = [c for c in commands if c['step'] in ('close','grip_close','close_gripper') or c['step'].endswith('_close')]
    close_sent = any(c['state'] in ('written','partial_write','echo_matched','sent_waited') for c in closed)
    if 'close' in str(record.get('stage','')).lower():
        close_sent |= any('#005P' in c['command'] and c['state'] in ('written','echo_matched') for c in commands)
    result = dict(available=True, schema_version='sorting-monitor-record-v1',
        task_id=record['task_id'], stage=record.get('stage'), raw_status=record['status'],
        record_kind=kind, serial_closed=record.get('serial_closed'),
        source_record=path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name,
        record_mtime=path.stat().st_mtime, finished=bool(record.get('finished_at')) or
            record.get('status') in ('completed_visual_evidence','needs_review','rejected','preview_only','simulated_visual_pass'),
        plan_bound=bound, plan_sha256=record.get('plan_sha256'), model_version=model,
        runtime_id=(plan or {}).get('runtime_id'),
        activity_profile=(plan or {}).get('profile'), scope_boundary=False,
        label=choice.get('label',choice.get('class_from_yoloe',evidence_label)) if bound else None,
        bin_id=choice.get('bin_id') if bound else None,
        pick_pixel=point(point_px,2), pick_card_xy_mm=point(choice.get('pick_card_xy_mm'),2),
        grip_xyz_mm=point(choice.get('grip_xyz_mm'),3), grip_xy_mm=point(choice.get('grip_xy_mm'),2),
        hover_xyz_mm=point(choice.get('hover_xyz_mm'),3),
        geometry=geometry_summary(stats,provenance), grasp_source_frame=evidence_frame,
        commands=commands, planned_commands=len(planned),
        written_commands=sum(c['state'] in ('written','partial_write','echo_matched') for c in commands) if physical else 0,
        echoed_commands=sum(c['state']=='echo_matched' for c in commands) if physical else 0,
        close_command_sent=close_sent if physical else False,
        release_command_sent=physical and any(c['step']=='release' and c['state'] in ('written','echo_matched','sent_waited') for c in commands),
        grasp_success_verified=False, physical_success_verified=False,
        visual_verification=dict(state='passed' if visual_pass else 'not_confirmed', frames=len(visual_frames),
            source_erased_min=min(source_erased) if source_erased else None,
            correct_bin_gain_min=min(bin_gain) if bin_gain else None,
            association=verification.get('association','task_record') if visual_pass else None),
        operator_feedback=feedback, reason=str(record.get('reason',record.get('error','')))[:600],
        operator_lift_reported=operator_lift, linked_segments=1,
        readonly=True, automatic_motion_allowed=False)
    result['current_written_commands'] = result['written_commands']
    result['current_echoed_commands'] = result['echoed_commands']
    result['grasp_source_task_id'] = record['task_id'] if result['pick_pixel'] else None
    for command in commands:
        command.update(task_id=record['task_id'], segment=record.get('stage'), current_segment=True)
    # Commissioning is split into separately reviewed segments. Follow only the
    # explicit, hashed predecessor chain; never borrow a nearby trial's grasp.
    if bound and not automatic and depth < 10 and plan.get('prior_execution_path'):
        base = trial.parent.resolve()
        previous = (base/plan['prior_execution_path']).resolve()
        previous_plan = (base/plan.get('prior_plan_path','')).resolve()
        if (previous.is_relative_to(base) and previous_plan.is_relative_to(base) and previous != path.resolve() and
            previous.is_file() and previous_plan.is_file() and
            hashlib.sha256(previous.read_bytes()).hexdigest() == plan.get('prior_execution_sha256') and
            hashlib.sha256(previous_plan.read_bytes()).hexdigest() == plan.get('prior_plan_sha256')):
            parent = project_record(previous, read(previous), depth+1)
            same_profile = not (plan.get('profile') and parent.get('activity_profile') and plan['profile']!=parent['activity_profile'])
            same_label = not (result.get('label') and parent.get('label') and result['label']!=parent['label'])
            same_bin = not (result.get('bin_id') and parent.get('bin_id') and result['bin_id']!=parent['bin_id'])
            result['scope_boundary'] = not (same_profile and same_label and same_bin)
            if parent['plan_bound'] and same_profile and same_label and same_bin and (not parent.get('model_version') or not model or parent['model_version']==model):
                result['linked_segments'] += parent['linked_segments']
                result['written_commands'] += parent['written_commands']
                result['echoed_commands'] += parent['echoed_commands']
                result['close_command_sent'] |= parent['close_command_sent']
                result['release_command_sent'] |= parent['release_command_sent']
                result['operator_lift_reported'] |= parent['operator_lift_reported']
                if not feedback:result['operator_feedback'] = parent['operator_feedback']
                if parent['pick_pixel'] and (not result['pick_pixel'] or result['pick_pixel']==parent['pick_pixel']):
                    for key in ('label','bin_id','pick_pixel','pick_card_xy_mm','grip_xyz_mm','grip_xy_mm',
                                'hover_xyz_mm','geometry','grasp_source_frame'):
                        if result[key] is None:result[key] = parent[key]
                    if result['grasp_source_task_id'] is None or evidence_frame is None and parent['grasp_source_frame'] is not None:
                        result['grasp_source_task_id']=parent['grasp_source_task_id']
                previous_commands = [dict(c,current_segment=False) for c in parent['commands']]
                result['commands'] = (previous_commands + commands)[-160:]
                for i,c in enumerate(result['commands'],1):c['index']=i
    return result


class ControlTelemetry:
    def __init__(self, roots=None, refresh_s=1):
        self.roots = roots
        self.refresh_s = refresh_s
        self.last_scan = 0
        self.cached = dict(available=False, reason='no_execution_record')
        self.lock = threading.Lock()
        self.selected = None
        self.sources = {}
        self.grasp_cache = {}

    def scan(self, requested_task=None):
        roots = self.roots
        if roots is None:
            roots = list((ROOT/'local-only/yoloe-physical-loop').glob('physical-*'))
            roots += [ROOT/'local-only/sorting-station-ledger']
        paths = []
        for root in map(Path,roots):
            journals = list(root.glob('*/journal')) + [root/'journal']
            if root.name.endswith('ledger') or self.roots is not None:
                journals.append(root)
            for journal in journals:
                paths.extend(p for p in journal.glob('*.json') if not p.name.endswith(('.observation.json','.review.json')))
        paths = sorted(set(paths), key=lambda p:p.stat().st_mtime, reverse=True)[:200]
        invalid_time = 0
        records = []
        for path in paths:
            try:
                record = read(path)
                if (not isinstance(record,dict) or not isinstance(record.get('task_id'),str) or
                    not isinstance(record.get('steps'),list) or not isinstance(record.get('status'),str)):
                    continue
                records.append((path,record))
            except (OSError,ValueError,KeyError,TypeError,IndexError):
                invalid_time = max(invalid_time,path.stat().st_mtime if path.exists() else time.time())
        history=[]
        self.sources={r['task_id']:p for p,r in reversed(records)}
        for path,record in records:
            if not any(r['task_id']==record['task_id'] for r in history):
                history.append(dict(task_id=record['task_id'],stage=record.get('stage'),
                    recorded_at=datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat()))
            if len(history)>=40:break
        selected=next(((p,r) for p,r in records if not requested_task or r['task_id']==requested_task),None)
        if selected and (requested_task or invalid_time<=selected[0].stat().st_mtime):
            result=project_record(*selected)
            self.detached_review(result,roots)
            result.update(records=history,following_latest=not bool(requested_task))
            return result
        return dict(available=False, records=history,reason='record_write_in_progress' if invalid_time else 'no_execution_record')

    def detached_review(self,result,roots):
        origin=self.sources.get(result.get('grasp_source_task_id'))
        if not origin or result['visual_verification']['state']=='passed':return
        trial=origin.parent.parent if origin.parent.name=='journal' else origin.parent
        hashes={hashlib.sha256(p.read_bytes()).hexdigest() for p in list(trial.glob('*precommit.json'))[:30]
                if p.stat().st_size<=2_000_000}
        if not hashes:return
        for root in map(Path,roots):
            for path in list(root.glob('*/final-review.json'))[:100]:
                review=read(path)
                if (review.get('status')!='supervised_trial_visual_evidence_pass' or
                    review.get('source_baseline_json_sha256') not in hashes or
                    review.get('model_version')!=result.get('model_version') or
                    result.get('runtime_id') and review.get('runtime_id')!=result['runtime_id'] or
                    path.stat().st_mtime<result['record_mtime']):continue
                frames=review.get('frames',[])
                if not frames:continue
                source=[r.get('source',{}).get('erased') for r in frames]
                gains=[r.get('bin',{}).get('beyond') for r in frames]
                if not all(number(v) for v in source+gains):continue
                result['visual_verification']=dict(state='passed',frames=len(frames),
                    source_erased_min=min(source),correct_bin_gain_min=min(gains),association='hashed_execution_baseline')
                feedback_path=path.parent/'operator-feedback.json'
                if feedback_path.is_file():
                    feedback=read(feedback_path)
                    if feedback.get('final_review_sha256')==hashlib.sha256(path.read_bytes()).hexdigest():
                        result['operator_feedback']=str(feedback.get('original_text',feedback.get('text','')))[:600]
                return

    def grasp_frame(self,health,task_id=None):
        """Recover original uploaded pixels by the execution's decoded-BGR hash."""
        import cv2
        import numpy as np
        record=self.status(health,task_id)
        if not record.get('available') or not record.get('pick_pixel'):
            return dict(available=False,reason='no_bound_grasp_point')
        source=self.sources.get(record.get('grasp_source_task_id'))
        if not source:return dict(available=False,reason='grasp_origin_unavailable')
        trial=source.parent.parent if source.parent.name=='journal' else source.parent
        plan=read(trial/'plan.json')
        evidence,bound_image=grasp_evidence(trial,plan)
        expected=evidence.get('decoded_bgr_sha256')
        if not expected:return dict(available=False,reason='grasp_frame_identity_missing')
        key=(expected,tuple(record['pick_pixel']),record.get('model_version'),record['grasp_source_task_id'])
        if key in self.grasp_cache:return dict(self.grasp_cache[key],selected_task_id=record['task_id'])
        image_candidates=[]
        if bound_image:image_candidates.append(bound_image)
        for folder in (() if bound_image else ('yolov8s-deployment','yolo11s-deployment','yoloe-deployment')):
            base=ROOT/'local-only'/folder/'preview'
            for path in list(base.glob('*/result.json'))[-400:]:
                metadata=read(path)
                if metadata.get('model_version')!=record.get('model_version'):continue
                image_path=path.parent/'input-image'
                if not image_path.is_file() or image_path.stat().st_size>8_500_000:continue
                raw=image_path.read_bytes()
                if hashlib.sha256(raw).hexdigest()!=metadata.get('image_sha256'):continue
                image_candidates.append(image_path)
        for image_path in image_candidates:
                raw=image_path.read_bytes();frame=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
                if frame is None or hashlib.sha256(frame.tobytes()).hexdigest()!=expected:continue
                x,y=record['pick_pixel']
                if not (0<=x<frame.shape[1] and 0<=y<frame.shape[0]):continue
                value=dict(available=True,source='execution_record_original_pixels',image_base64=base64.b64encode(raw).decode('ascii'),
                    image_mime='image/png' if raw.startswith(b'\x89PNG') else 'image/jpeg',
                    image_size=[frame.shape[1],frame.shape[0]],pixel=record['pick_pixel'],
                    source_task_id=record['grasp_source_task_id'],source_frame_id=record['grasp_source_frame'],
                    decoded_bgr_sha256=expected,current_live_frame=False,control_authorized=False)
                if len(self.grasp_cache)>=8:self.grasp_cache.clear()
                self.grasp_cache[key]=value
                return dict(value,selected_task_id=record['task_id'])
        return dict(available=False,reason='original_pixels_not_found')

    def status(self, health, task_id=None):
        with self.lock:
            if time.monotonic()-self.last_scan >= self.refresh_s or task_id!=self.selected:
                try:self.cached = self.scan(task_id)
                except (OSError,ValueError,TypeError):self.cached = dict(available=False,reason='record_unavailable')
                self.last_scan = time.monotonic()
                self.selected = task_id
            result = dict(self.cached)
        if result.get('available'):
            result['record_age_s'] = max(0,time.time()-result['record_mtime'])
            result['recorded_at'] = datetime.fromtimestamp(result['record_mtime'],timezone.utc).isoformat()
            result['model_matches_monitor'] = result.get('model_version') == health.get('model_sha256') if result.get('model_version') else None
            result['live_camera_overlay'] = False
        return result
