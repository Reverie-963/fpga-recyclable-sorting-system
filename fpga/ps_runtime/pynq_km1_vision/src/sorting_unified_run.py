"""Bounded unified single/batch entry; all physical ownership spans the batch."""
import argparse
import copy
import json
import os
from pathlib import Path
import time
import uuid
import hashlib
import cv2
import numpy as np
from .sorting_known import bound_file, file_hash
from .sorting_loop import Journal, LABELS, fingerprint, mapped, need
from .sorting_io import HTTPYoloe, Statistics, Motion
from .sorting_relocated import RelocatedMotion
from .sorting_relocated_run import load_binding, integration_blockers, raw_window, read, motion_shift_limit,motion_marker_limit,initial_marker_shift_limit,initial_window_age_limit
from .sorting_relocated_geometry import RelocatedGeometry, bounded_rectangle
from .sorting_position_controller import PositionWindowedCamera
from .sorting_station_io import StationStatistics, StationCamera, station_policy
from .sorting_instances import InstanceLedger
from .sorting_unified import UnifiedController
from .sorting_unified_plan import validate_policy,apply_decision_thresholds
from .sorting_batch_verification import verify_batch_final
from .station_geometry import world_to_pixels
from .sorting_review_policy import review_factor
from openmv_gripper.km1_arm_bridge import validate_packet

UNIFIED_SOURCES = ('sorting_instances.py','sorting_unified_plan.py','sorting_unified.py','sorting_unified_run.py',
                   'sorting_item_heights.py','prompted_height.py')


def load_unified(path):
    bundle=load_binding(path)
    b,c,base,cam,station,reg,scene,files=bundle
    need('unified_policy' in files, 'unified_policy_not_installed')
    policy=read(files['unified_policy']);validate_policy(policy,c)
    if policy.get('jaw_calibration') is not None:
        from .jaw_measurements import validate_measurement_record
        need('jaw_measurements' in files,'jaw_measurement_evidence_required')
        need(any(Path(r['path']).name=='jaw_measurements.py' for r in b['implementation_sources']),
             'jaw_measurement_validator_not_bound')
        validate_measurement_record(files['jaw_measurements'],policy['jaw_calibration'],
                                    policy['jaw_calibration_operator_statement'])
    need(review_factor(b)==review_factor(policy)==review_factor(c),'review_policy_binding_mismatch')
    sources={Path(v['path']).name for v in b['implementation_sources']}
    need(set(UNIFIED_SOURCES).issubset(sources), 'unified_implementation_not_bound')
    if policy.get('batch_verification_mode')=='end_of_batch':
        need('sorting_batch_verification.py' in sources,'batch_final_verification_implementation_not_bound')
    for label in LABELS:
        record_key,image_key='empty_record_'+label,'empty_image_'+label
        if record_key in files or image_key in files:
            need(record_key in files and image_key in files,'incomplete_per_bin_empty_reference')
            record=read(files[record_key]);image=cv2.imread(str(files[image_key]))
            need(record.get('kind')=='supervised-per-bin-empty-reference-v1' and record.get('label')==label and
                 record.get('bin_id')==c['profiles'][label]['bin_id'] and record.get('bin_empty_confirmed') is True and
                 record.get('review_source')=='saved_unmodified_frames_visual_review' and
                 record.get('other_bins_empty_claimed') is False and image is not None and
                 record['reference_bgr_sha256']==hashlib.sha256(image.tobytes()).hexdigest() and
                 record['camera_sha256']==fingerprint(cam) and record['station_sha256']==fingerprint(station) and
                 record['registration_sha256']==fingerprint(reg) and
                 record['scene_regions_sha256']==fingerprint(scene['scene_regions']) and
                 len(record.get('frames',[]))==3 and len({r['source_frame_id'] for r in record['frames']})==3 and
                 all(r['bin_object_count']==0 and r['changed_fraction']<=c['max_scene_changed_fraction'] for r in record['frames']),
                 'per_bin_empty_reference_not_verified')
            for frame in record['frames']:bound_file(files[record_key].parent,frame['image'])
    if policy.get('motion_timing') is not None:
        need('sorting_unified_motion.py' in sources, 'unified_timing_implementation_not_bound')
    # A custom measured landing slot needs its own immutable observation record.
    for label,slots in policy['bin_slots'].items():
        for slot in slots:
            if slot['origin']=='explicit_slot_record':
                record=read(bound_file(files['unified_policy'].parent,slot['motion_record']))
                need(record.get('kind')=='verified-bin-slot-v1' and record.get('label')==label and
                     record.get('slot_sha256')==fingerprint({k:v for k,v in slot.items() if k!='motion_record'}) and
                     record.get('operator_confirmed') is True and record.get('physical_trial_completed') is True,
                     'bin_slot_not_physically_bound')
    return bundle,policy


def apply_geometry(geometry,policy):
    poly=np.asarray(policy['source_card_polygon_mm'],float)
    world=mapped(np.linalg.inv(np.asarray(geometry.registration['world_to_card_xy'],float)),poly)
    heights=(0,max(v['object_height_mm'] for v in policy['models'].values()))
    xyz=np.vstack([np.c_[world,np.full(len(world),h)] for h in heights])
    rect=bounded_rectangle(world_to_pixels(xyz,geometry.pose,geometry.camera,geometry.station),geometry.runtime['image_size'])
    regions=copy.deepcopy(geometry.runtime['statistics']['regions']);regions['source']=rect
    masks=[];w,h=geometry.runtime['image_size']
    for r in regions.values():
        mask=np.zeros((h,w),bool);mask[r[1]:r[3],r[0]:r[2]]=True
        need(not any((mask&other).any() for other in masks),'unified_projected_regions_overlap');masks.append(mask)
    geometry.runtime['source_roi']=rect;geometry.runtime['statistics']['regions']=regions
    geometry.signature=fingerprint(dict(base_geometry=geometry.signature,unified_policy=fingerprint(policy),regions=regions))
    return geometry


def per_bin_empty_images(geometry,files,stream):
    result={}
    for label in LABELS:
        key='empty_record_'+label
        if key not in files:continue
        record=read(files[key])
        need(record['source_stream_id']==stream,'per_bin_empty_reference_stream_changed')
        saved=copy.copy(geometry)
        saved.scene=dict(geometry.scene,reference_pose=record['reference_pose'])
        result[label]=saved.warp_empty_reference(cv2.imread(str(files['empty_image_'+label])))
    return result


class BatchLease:
    def __init__(self,path,stops,deadline,clock=time.monotonic):
        self.path,self.stops,self.deadline,self.clock=Path(path),[Path(p) for p in stops],deadline,clock
        self.token=json.dumps(dict(kind='sorting-batch-owner-v1',pid=os.getpid(),token=uuid.uuid4().hex));self.owned=False
    def check(self):
        need(not any(p.exists() for p in self.stops),'operator_stop_requested')
        need(self.clock()<self.deadline,'unified_batch_deadline')
        if self.owned:need(self.path.read_text(encoding='utf-8')==self.token,'unified_batch_lease_replaced')
    def __enter__(self):
        self.check();self.path.parent.mkdir(parents=True,exist_ok=True)
        fd=os.open(self.path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        try:os.write(fd,self.token.encode());os.fsync(fd)
        finally:os.close(fd)
        self.owned=True
        try:self.check()
        except BaseException:
            self.__exit__()
            raise
        return self
    def __exit__(self,*args):
        if self.owned and self.path.exists() and self.path.read_text(encoding='utf-8')==self.token:self.path.unlink()
        self.owned=False


class BatchMotion(RelocatedMotion):
    def __init__(self,*args,owner,**kwargs):
        self.batch_owner=owner
        super().__init__(*args,**kwargs)
    def check_stop(self):
        super().check_stop();self.batch_owner.check()
    def open(self,plan):
        self.owned_task_id=self.journal.pending()['task_id'];self.check_stop()
        from serial.tools import list_ports
        identity=self.arm['usb_identity']
        devices=[p.device for p in list_ports.comports() if p.vid==identity['vid'] and p.pid==identity['pid'] and
                 (identity.get('serial_number') is None or p.serial_number==identity['serial_number'])]
        need(devices==[self.arm['port']],'unified_station_usb_not_unique')
        for _,data,duration in plan:need(validate_packet(data,[[500,2500]]*6)==duration,'invalid_unified_packet')
        # The outer lease is held across observation, every item and every gap.
        # Closing UART must not release another item's batch ownership.
        Motion.open(self,plan)


def activate_regions(stats,geometry,owner):
    owner.check();current=stats.checked(stats.request('/health'))
    result=stats.request('/regions',dict(stream_id=current['stream_id'],frame_id=current['frame_id'],
        expected_config_sha256=current['config_sha256'],expected_baseline_frame_id=current['baseline_frame_id'],
        regions=geometry.runtime['statistics']['regions'],calibration_sha256=geometry.signature))
    need(result['baseline_invalidated'] and result['config_sha256']==station_policy(geometry.runtime) and
         result['calibration_sha256']==geometry.signature,'unified_region_activation_mismatch')
    return result


def target_region_runtime(geometry,rect):
    runtime=copy.deepcopy(geometry.runtime);w,h=runtime['image_size']
    need(isinstance(rect,list) and len(rect)==4 and all(type(v) is int for v in rect) and
         0<=rect[0]<rect[2]<=w and 0<=rect[1]<rect[3]<=h,'invalid_target_evidence_rectangle')
    for name,r in runtime['statistics']['regions'].items():
        if name=='source':continue
        need(not (max(rect[0],r[0])<min(rect[2],r[2]) and max(rect[1],r[1])<min(rect[3],r[3])),
             'target_evidence_overlaps_protected_region')
    runtime['source_roi']=list(rect);runtime['statistics']['regions']['source']=list(rect)
    station_policy(runtime)
    return runtime


def configure_target_region(controller,choice,owner):
    """Under the batch lease, bind only this target's RGB source rectangle.

    Detection stays full-frame. All destination/static coordinates and all
    release commands stay unchanged; the precise baseline is bound afterward.
    """
    owner.check();geometry=controller.geometry
    runtime=apply_decision_thresholds(target_region_runtime(geometry,choice['source_evidence_roi']),controller.policy)
    # RGB evidence is per target; registration and first-motion matching still
    # use the complete immutable pickup region from this batch's observation.
    runtime['pickup_source_roi']=copy.deepcopy(controller.c.get('pickup_source_roi',controller.c['source_roi']))
    geometry.runtime=runtime
    geometry.signature=fingerprint(dict(frozen_geometry=geometry.signature,target_regions=runtime['statistics']['regions']))
    activation=activate_regions(controller.stats,geometry,owner)
    statistics=StationStatistics(runtime)
    controller.stats=statistics;controller.c=copy.deepcopy(runtime)
    controller.c['max_objects']=controller.policy['max_objects']
    controller.source.source.c=runtime;controller.source.source.statistics=statistics
    controller.motion.c=runtime
    controller.motion.camera.c=runtime;controller.motion.camera.statistics=statistics
    return activation


def clear_batch_pending(journal):
    """Only retry transient Windows file sharing; never retry any motion."""
    for attempt in range(10):
        try:
            (journal.root/'batch-pending.json').unlink(missing_ok=True)
            return
        except PermissionError as exc:
            if getattr(exc,'winerror',None) not in (32,33) or attempt==9:raise
            time.sleep(.05)


def run_batch(controller_factory,journal,max_items,max_seconds,owner_check=lambda:None,clock=time.monotonic,
              freeze_plan=None):
    """Bounded batch, optionally compiling all routes before the first motion."""
    need(type(max_items) is int and 1<=max_items<=20 and 0<max_seconds<=3600,'invalid_unified_batch_limits')
    batch=uuid.uuid4().hex;started=clock();tracker=None;deferred_controllers=[]
    manifest=dict(batch_id=batch,status='observing',items=[],completed_count=0,remaining=[],
        max_items=max_items,max_seconds=max_seconds,automatic_retry=False,unattended_accepted=False,
        motion_completed_count=0,per_item_visual_failures=[])
    journal.save('batch-pending.json',manifest)
    try:
        prepared=None;first_controller=None
        if freeze_plan is not None:
            owner_check()
            first_controller=controller_factory(0);tracker=first_controller.tracker
            need(first_controller.policy.get('batch_verification_mode')=='end_of_batch' and
                 first_controller.policy.get('skip_per_item_visual_verification') is True,
                 'frozen_batch_requires_final_verification')
            prepared=freeze_plan(first_controller,max_items)
            owner_check();need(clock()-started<max_seconds,'unified_batch_deadline')
            manifest['planning_completed_before_motion']=True
            manifest['planned_count']=len(prepared)
            manifest['frozen_plan']=[dict(track_id=p['selected'][0],choice=p['selected'][2],
                plan=[dict(step=n,command=d.decode(),duration_ms=ms) for n,d,ms in p['selected'][3]])
                for p in prepared]
            manifest['remaining']=[dict(track_id=i,label=o['label']) for i,o,_ in tracker.live]
            journal.save('batch-pending.json',manifest)
            if not prepared:manifest['status']='selection_complete' if tracker.live else 'empty_scene'
        for index in range(max_items):
            if prepared is not None and index>=len(prepared):break
            owner_check();need(clock()-started<max_seconds,'unified_batch_deadline')
            controller=first_controller if index==0 and first_controller is not None else controller_factory(index)
            tracker=controller.tracker
            if prepared is not None:
                controller.prepared_batch_item=prepared[index]
            result=controller.cycle();result['batch_id']=batch;result['batch_index']=index
            journal.save(result['task_id']+'.json',result)
            if result['status']=='needs_review':journal.save('pending.json',result)
            manifest['items'].append(dict(task_id=result['task_id'],status=result['status'],choice=result.get('choice'),
                report_sha256=file_hash(journal.root/(result['task_id']+'.json')),reason=result.get('reason'),
                physical_motion_attempted=result.get('physical_motion_attempted',False)))
            manifest['items'][-1]['slot_checks']=result.get('slot_checks',[])
            manifest['remaining']=[dict(track_id=i,label=o['label']) for i,o,_ in tracker.live]
            if result['status']=='motion_completed_awaiting_batch_verification':
                deferred_controllers.append(controller)
                manifest['motion_completed_count']+=1
                if result['verification']['status']=='per_item_visual_failed':
                    manifest['per_item_visual_failures'].append(dict(task_id=result['task_id'],
                        reason=result['verification']['reason']))
                if (prepared is not None and index+1==len(prepared)) or not tracker.live:
                    manifest['status']='awaiting_batch_verification';break
            elif result['status'] in ('completed_visual_evidence','simulated_visual_pass'):
                manifest['completed_count']+=1
                if not tracker.live:
                    manifest['status']='completed';break
            elif result['status']=='empty_scene':manifest['status']='completed' if manifest['completed_count'] else 'empty_scene';break
            elif result['status']=='no_matching_material':manifest['status']='selection_complete';break
            else:
                manifest['status']='needs_review' if (result['status']=='needs_review' or
                    (freeze_plan is not None and deferred_controllers)) else 'blocked'
                manifest['reason']=result.get('reason');break
            journal.save('batch-pending.json',manifest)
        else:manifest['status']='item_limit_reached'
    except (Exception,KeyboardInterrupt) as exc:
        manifest.update(status='needs_review' if (journal.pending() is not None or
                        (freeze_plan is not None and deferred_controllers)) else 'blocked',
                        reason=type(exc).__name__+':'+str(exc))
    finally:
        if (deferred_controllers and journal.pending() is None and
                (freeze_plan is None or manifest['status']=='awaiting_batch_verification')):
            try:
                owner_check()
                final=verify_batch_final(deferred_controllers,journal,batch)
                manifest['final_verification']=final
                if final['status']=='batch_visual_evidence_pass':
                    manifest['completed_count']=len(deferred_controllers)
                    if manifest['status']=='awaiting_batch_verification':manifest['status']='completed'
                else:
                    manifest.update(status='needs_review',reason=final.get('reason','batch_final_verification_failed'))
            except Exception as exc:
                manifest.update(status='needs_review',reason=type(exc).__name__+':'+str(exc))
        manifest['elapsed_s']=clock()-started
        journal.save(batch+'.batch.json',manifest)
        if manifest['status']=='needs_review':journal.save('batch-pending.json',manifest)
        else:clear_batch_pending(journal)
    return manifest


def run(path,mode,ready=False,max_items=6,max_seconds=1200,material='AUTO'):
    entry_started=time.monotonic()
    need(material in ('AUTO','PAPER','METAL','PLASTIC'),'invalid_unified_material')
    allowed=LABELS if material=='AUTO' else (material.lower(),)
    bundle,policy=load_unified(path)
    max_seconds*=policy.get('wait_time_scale',1)
    b,c,base,cam,station,reg,scene,files=bundle
    ledger=(base/c['journal_dir']).resolve()
    blockers=integration_blockers(b,c,base,files)
    if (ledger/'batch-pending.json').exists():blockers.append('unified_batch_needs_review')
    if mode=='preflight':
        return dict(status='unified_ready' if not blockers else 'unified_waiting',execution_blockers=blockers,
            labels=list(LABELS),planar_orientation_degrees=[0,360],max_objects=policy['max_objects'],
            physical_workspace_qualified=False,physical_motion=False,
            material_selection=material,detection_window_px=policy.get('detection_window_px',c['source_roi']),
            fixed_release_routes=True,
            batch_verification_mode=policy.get('batch_verification_mode','per_item'),
            skip_per_item_visual_verification=policy.get('skip_per_item_visual_verification',False),
            wait_time_scale=policy.get('wait_time_scale',1),
            review_tolerance_factor=review_factor(b),
            decision_thresholds=policy.get('decision_thresholds',{}),
            empty_bin_max_changed_fraction=policy.get('empty_bin_max_changed_fraction',c['max_scene_changed_fraction']),
            min_fixed_markers=motion_marker_limit(b),
            initial_marker_shift_px=initial_marker_shift_limit(b),
            initial_window_max_age_ms=initial_window_age_limit(b,c),
            motion_timing=policy.get('motion_timing'),
            bin_slot_capacity={label:sum(s['origin']!='candidate' for s in slots) for label,slots in policy['bin_slots'].items()})
    need(mode in ('preview','execute'),'invalid_unified_mode')
    if mode=='execute':need(ready and not blockers,'unified_setup_not_ready:'+','.join(blockers))
    raw,pose,old_stats=raw_window(c,cam,station,reg,scene,min_fixed_markers=motion_marker_limit(b),
                               max_marker_shift_px=initial_marker_shift_limit(b),
                               initial_window_max_age_ms=initial_window_age_limit(b,c),review_tolerance_factor=review_factor(b))
    geometry=apply_geometry(RelocatedGeometry(c,cam,station,reg,scene,pose,raw[-1]['image'],
        motion_shift_px=motion_shift_limit(b),min_fixed_markers=motion_marker_limit(b),
        review_tolerance_factor=review_factor(b)),policy)
    geometry.runtime=apply_decision_thresholds(geometry.runtime,policy)
    geometry.runtime['known']['expected_camera_stream_id']=raw[-1]['observation']['source_stream_id']
    empty=geometry.warp_empty_reference(cv2.imread(str(files['empty_image'])))
    empty_by_label=per_bin_empty_images(geometry,files,raw[-1]['observation']['source_stream_id'])
    if mode=='preview':
        source=PositionWindowedCamera(StationCamera(old_stats.c,HTTPYoloe(c,None),old_stats))
        with Journal(Path(path).parent/'unified-preview-ledger') as journal:
            tracker=InstanceLedger(geometry.runtime['min_scene_iou'])
            controller=UnifiedController(c,source,Statistics(),None,journal,geometry,empty,base,
                policy=policy,tracker=tracker,mode='preview',allowed_labels=allowed,empty_by_label=empty_by_label)
            report=controller.cycle();report['execution_blockers']=blockers
            report['projected_regions']=geometry.runtime['statistics']['regions']
            report['entry_elapsed_s']=time.monotonic()-entry_started
            journal.save(report['task_id']+'.json',report)
            return report
    lease=(base/c['known']['device_lease']).resolve()
    stops=[(base/p).resolve() for p in c['known']['stop_files']]
    with Journal(ledger) as journal:
        need(journal.pending() is None and not (ledger/'batch-pending.json').exists(),'unified_previous_transaction_unresolved')
        with BatchLease(lease,stops,time.monotonic()+max_seconds) as owner:
            activation=activate_regions(old_stats,geometry,owner)
            stats=StationStatistics(geometry.runtime)
            source=PositionWindowedCamera(StationCamera(geometry.runtime,HTTPYoloe(c,None),stats))
            tracker=InstanceLedger(geometry.runtime['min_scene_iou'])
            def factory(index):
                owner.check()
                current_stats=StationStatistics(geometry.runtime)
                source.source.c=geometry.runtime;source.source.statistics=current_stats
                source.after(time.monotonic()*1000)
                motion=BatchMotion(geometry.runtime,journal,geometry,lease,stops,ledger/'pending.json',owner=owner)
                return UnifiedController(c,source,current_stats,motion,journal,geometry,empty,base,
                    policy=policy,tracker=tracker,mode='known_physical',authorization_check=owner.check,
                    allowed_labels=allowed,target_region_setup=(
                        (lambda controller,choice:configure_target_region(controller,choice,owner))
                        if 'detection_window_px' in policy else None),empty_by_label=empty_by_label)
            result=run_batch(factory,journal,max_items,max_seconds,owner.check)
            result.update(binding_sha256=file_hash(path),region_activation=activation,
                          unified_policy_sha256=fingerprint(policy),material_selection=material,
                          entry_elapsed_s=time.monotonic()-entry_started,
                          physical_motion_attempted=any(i['physical_motion_attempted'] for i in result['items']))
            journal.save(result['batch_id']+'.batch.json',result)
            return result


def resolve_batch(path,observation):
    bundle,_=load_unified(path);_,c,base,*_=bundle
    ledger=(base/c['journal_dir']).resolve()
    need(observation.strip(),'batch_review_observation_required')
    need(not (base/c['known']['device_lease']).resolve().exists(),'batch_device_still_owned')
    with Journal(ledger) as journal:
        need(journal.pending() is None,'resolve_motion_task_before_batch')
        record=read(ledger/'batch-pending.json')
        journal.save(record['batch_id']+'.batch-review.json',dict(batch=record,observation=observation,
            outcome='operator_resolved_not_automatic_success'))
        (ledger/'batch-pending.json').unlink()
    return dict(status='operator_resolved_not_automatic_success',physical_motion=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding',type=Path,required=True);parser.add_argument('--config',type=Path)
    action=parser.add_mutually_exclusive_group(required=True)
    for name in ('preflight','preview','execute','resolve-batch'):action.add_argument('--'+name,action='store_true')
    parser.add_argument('--ready',action='store_true');parser.add_argument('--max-items',type=int,default=6)
    parser.add_argument('--max-seconds',type=float,default=1200);parser.add_argument('--observation',default='')
    parser.add_argument('--material',choices=('AUTO','PAPER','METAL','PLASTIC'),default='AUTO')
    args=parser.parse_args()
    if args.config:
        b=read(args.binding);need(args.config.resolve()==bound_file(args.binding.parent,b['files']['config']),
                                'unified_explicit_config_mismatch')
    if args.resolve_batch:result=resolve_batch(args.binding,args.observation)
    else:result=run(args.binding,'execute' if args.execute else 'preview' if args.preview else 'preflight',
                    args.ready,args.max_items,args.max_seconds,args.material)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    return 2 if result['status'] in ('rejected','blocked','needs_review','unified_waiting') else 0

if __name__=='__main__':raise SystemExit(main())
