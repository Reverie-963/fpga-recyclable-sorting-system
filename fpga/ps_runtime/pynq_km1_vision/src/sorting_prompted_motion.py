"""Single supervised prompted-model transaction through existing motion guards."""
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request
import cv2
from .prompted_recyclables import read_policy, settings_for, classify, digest
from .sorting_loop import need, Journal, mask_of
from .sorting_prompted_preview import PromptedPreviewController, ROOT
from .sorting_unified_run import (load_unified, apply_geometry, per_bin_empty_images,
    BatchLease, BatchMotion, activate_regions, configure_target_region)
from .sorting_relocated_run import (raw_window, integration_blockers, motion_shift_limit,
    motion_marker_limit, initial_marker_shift_limit, initial_window_age_limit)
from .sorting_relocated_geometry import RelocatedGeometry,pixels_to_card
from .sorting_review_policy import review_factor
from .sorting_station_io import StationCamera, StationStatistics
from .sorting_position_controller import PositionWindowedCamera
from .sorting_instances import InstanceLedger
from .sorting_io import Statistics
from .prompted_motion_policy import read_motion_options
from .prompted_trial_scope import execution_scope
from .prompted_region_geometry import geometry_for_policy,apply_prompted_geometry
from .prompted_observation import frame_count,raw_prompted_window,PromptedWindowedCamera,PromptedStationStatistics
from .prompted_height import object_height,effective_heights
from .motion_sequence import scene_check_mode
from .sorting_item_heights import ItemHeights, height_plan as validate_height_plan

CALIBRATION=Path(os.environ.get('YOLOE_RECYCLABLE_CALIBRATION', ROOT/'local-only/yoloe-recyclables/calibration-20261030/deployment.json'))
FRAME_MAX_AGE_MS=6000
FRAME_TIMEOUT_S=6


def observation_timing(config):
    result=copy.deepcopy(config)
    options=read_motion_options()
    result.update(max_age_ms=options['frame_max_age_ms'],frame_timeout_s=options['frame_timeout_s'],stable_frames=frame_count())
    return result


class PromptedHTTP:
    def __init__(self, settings, url=None):
        self.settings,self.url=settings,url or os.environ.get('YOLOE_RECYCLABLE_URL','http://127.0.0.1:8811')
        self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def predict(self,image,prompt_mode='text'):
        need(prompt_mode=='text','prompted_motion_requires_text')
        ok,png=cv2.imencode('.png',image);need(ok,'cannot_encode_prompted_control_image')
        request=urllib.request.Request(self.url+'/detect',data=json.dumps(dict(
            image_base64=base64.b64encode(png).decode(),control_only=True)).encode(),headers={'Content-Type':'application/json'})
        with self.opener.open(request,timeout=10) as response:result=json.load(response)
        need(result['settings_sha256']==self.settings['settings_sha256'],'prompt_settings_changed')
        need(result['model_sha256']==read_policy()['model']['sha256'],'prompt_model_changed')
        need(result['source_bgr_sha256']==hashlib.sha256(image.tobytes()).hexdigest() and
             result['image_size']==list(image.shape[1::-1]),'prompt_control_image_binding_changed')
        result.pop('overlay_base64',None)
        for obj in result['instances']:
            need(obj['material'] in ('paper','plastic','metal'),'unknown_prompted_motion_material')
            mask_of(obj,result['image_size'])
            obj['label']=obj['material']
        return dict(result,model_version=result['model_sha256'],runtime_id=result['settings_sha256'],
                    prompt_mode='text',image_sha256=hashlib.sha256(png).hexdigest())


class PromptedMotionController(PromptedPreviewController):
    def __init__(self,*args,prompt_settings,trial_labels=None,**kwargs):
        self.prompt_settings=prompt_settings
        self.trial_labels=trial_labels
        self.observation_evidence=[]
        super().__init__(*args,**kwargs)

    def acquire(self,*args,**kwargs):
        row=super().acquire(*args,**kwargs)
        policy=read_policy()
        for obj,mask in zip(row['observation']['instances'],row['masks']):
            zone=self.known_bin_zone(mask)
            if zone is None:
                zone=obj.get('station_zone','source')
                if zone not in ('source','bin_paper','bin_plastic','bin_metal'):zone='source'
            obj.update(classify(policy,self.prompt_settings,obj,zone))
        self.observation_evidence.append(row)
        self.observation_evidence=self.observation_evidence[-3:]
        return row

    def cycle(self):
        report=super().cycle()
        report['motion_sequence']=dict(scene_check_mode=scene_check_mode(getattr(self,'motion_options',{})),
            initial_visual_checks=getattr(self.motion,'sequence_scene_checks',0),
            per_command_visual_checks=scene_check_mode(getattr(self,'motion_options',{}))=='per_command',
            post_placement_verification_required=True,automatic_retry=False)
        report['prompted_observation_frames']=[self.journal.capture(report['task_id'],'prompted_'+str(i),r)
            for i,r in enumerate(self.observation_evidence)]
        report['original_labels_by_frame']=[[dict(label=o['official_label'],material=o['label'],
            confidence=o['confidence'],zone=o['zone']) for o in r['observation']['instances']]
            for r in self.observation_evidence]
        self.journal.save(report['task_id']+'.json',report)
        return report

    def unchanged(self,before,after):
        if getattr(self,'item_heights',None) is not None:self.check_height_scene(before,after)
        if getattr(self,'motion_options',{}).get('full_scene_instance_stability_check_enabled',True):
            super().unchanged(before,after)
        else:
            from .sorting_instances import associate
            from .static_visual_verification import roi_mask
            from .sorting_loop import iou
            selected=getattr(self,'batch_context',{}).get('target')
            def targets(row):
                return [(o,m) for o,m in zip(row['observation']['instances'],row['masks'])
                        if o['official_label'] in self.trial_labels and o['zone']=='source' and o['semantic_gate_passed']]
            if selected is not None:
                # Opening all labels must not silently turn the disabled whole-
                # scene match back on. Track only this frozen selected target;
                # all other raw masks remain available to obstacle checks.
                old=[selected]
                new=[pair for pair in targets(after) if pair[0]['label']==selected[0]['label']
                     and iou(selected[1],pair[1])>=self.c['min_scene_iou']]
                need(len(new)==1,'prompt_selected_target_missing_or_moved' if not new else 'prompt_selected_target_ambiguous')
            else:
                old,new=targets(before),targets(after)
            matches=associate(old,new,self.c['min_scene_iou'])
            need(all(o['label']==new[j][0]['label'] for (o,_),j in zip(old,matches)),
                 'prompt_material_unstable')
            changed=self.changed_pixels(before,after)
            for rect in [self.c['source_roi']]+[p['bin_roi'] for p in self.c['profiles'].values()]:
                need(float(changed[roi_mask(rect,self.c['image_size'])].mean())<=self.c['max_scene_changed_fraction'],
                     'prompt_scene_rgb_changed')
            after['target_stability']=dict(full_scene_instance_matching=False,tracked_targets=len(old),
                raw_before_count=len(before['masks']),raw_after_count=len(after['masks']),
                all_latest_masks_retained_for_obstacle_checks=True)
            return
        from .sorting_loop import iou
        for obj,mask in zip(before['observation']['instances'],before['masks']):
            hits=[o for o,m in zip(after['observation']['instances'],after['masks']) if iou(mask,m)>=self.c['min_scene_iou']]
            need(len(hits)==1 and hits[0]['label']==obj['label'],'prompt_material_unstable')
            if obj['zone']=='source' and obj['semantic_gate_passed']:
                need(hits[0]['semantic_gate_passed'],'prompt_confidence_unstable')

    def candidate(self,row,target,track_id):
        need(self.mode!='known_physical' or target[0]['official_label'] in self.trial_labels,
             'prompted_original_class_not_in_trial_scope')
        choice,plan=super().candidate(row,target,track_id)
        choice.update(execution_basis='operator_authorized_prompted_single_item_trial',
                      perception_model_sha256=read_policy()['model']['sha256'],
                      settings_sha256=self.prompt_settings['settings_sha256'])
        return choice,plan

    def verify_selected(self,before,target,choice,after):
        if self.placement_verification['mode']=='bin_occupied':
            return super().verify_selected(before,target,choice,after)
        # Material controls routing; raw class changes do not invalidate it.
        from .sorting_instances import associate
        old=self.outside_objects(before)
        for row in after:
            outside=self.outside_objects(row)
            matched=associate(old,outside,self.c['min_scene_iou'],allow_extra=True)
            for index,(obj,_) in enumerate(outside):
                if index in matched:continue
                if obj['zone']=='bin_'+target[0]['label']:
                    need(obj['label']==target[0]['label'] and obj['semantic_gate_passed'],
                         'prompt_landing_label_or_zone_confidence')
        return super().verify_selected(before,target,choice,after)


def current_binding():
    pointer=ROOT/'local-only/station-grasp-integration/active.json'
    record=json.loads(pointer.read_text(encoding='utf-8'))
    binding=(pointer.parent/record['path']).resolve()
    need(hashlib.sha256(binding.read_bytes()).hexdigest()==record['sha256'],'active_binding_changed')
    return binding


def calibrated_settings():
    from .execution_model_context import current_profile
    active=current_profile()
    if active is not None:return copy.deepcopy(active['settings']),copy.deepcopy(active['receipt'])
    record=json.loads(CALIBRATION.read_text(encoding='utf-8'))
    settings_path=Path(os.environ.get('YOLOE_RECYCLABLE_SETTINGS',CALIBRATION.parent.parent/'settings.json'))
    settings=json.loads(settings_path.read_text(encoding='utf-8'))
    need(settings_for(read_policy(),settings)==settings,'prompted_settings_not_current_policy')
    need(record['settings_sha256']==settings['settings_sha256'],'prompted_calibration_settings_changed')
    need(hashlib.sha256((CALIBRATION.parent/'summary.json').read_bytes()).hexdigest()==record['evaluation_sha256'],
         'prompted_calibration_evidence_changed')
    if 'model_sha256' in record:
        need(record['model_sha256']==read_policy()['model']['sha256'],'prompted_trial_model_changed')
    for path,expected in record.get('implementation_sha256',{}).items():
        need(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,'prompted_trial_implementation_changed')
    return settings,record


def run(material='AUTO',execute=False,object_height_mm=None,height_plan=None):
    need(material in ('AUTO','PAPER','PLASTIC','METAL'),'invalid_material')
    settings,receipt=calibrated_settings()
    binding=current_binding();bundle,policy=load_unified(binding)
    b,c,base,cam,station,reg,scene,files=bundle
    scope=execution_scope(receipt,read_policy(),read_motion_options())
    ready_labels=scope['materials']
    allowed=(ready_labels if execute else ['paper','plastic','metal']) if material=='AUTO' else [material.lower()]
    need(allowed and (not execute or set(allowed).issubset(ready_labels)),'prompted_material_validation_not_ready')
    supplied_height=object_height(object_height_mm)
    height_plan=validate_height_plan(height_plan)
    need(height_plan is None or supplied_height is None,'instance_and_shared_height_conflict')
    heights=effective_heights(settings,supplied_height,allowed)
    item_heights=None
    if height_plan is not None:
        from .execution_model_context import current_profile
        active=current_profile()
        configuration=(active['upstream_configuration_sha256'] if active is not None
                       else settings['settings_sha256'])
        item_heights=ItemHeights(height_plan,model=read_policy()['model']['sha256'],
                                configuration=configuration,image_size=c['image_size'])
    mechanical_height_match=all(heights[m]==c['profiles'][m]['object_height_mm'] for m in allowed)
    # Operator height applies to this attempt's calibrated ray projection.
    # Robot grip Z and the measured mechanical maps remain explicit parameters.
    blockers=integration_blockers(b,c,base,files)
    ledger=(base/c['journal_dir']).resolve()
    need(not execute or not blockers,'prompted_station_blocked:'+','.join(blockers))
    # One pick, one final check using the explicitly selected verification mode.
    policy=copy.deepcopy(policy)
    policy['selection_policy']='clearance_then_route_v1'
    options=read_motion_options()
    from .prompted_grasp_override import requested_grasp_planes
    c,policy,grasp_override=requested_grasp_planes(c,policy,options,scope)
    from .prompted_batch import batch_settings,PromptedBatchController,run_registered_batch
    batch=batch_settings(options)
    need(not (batch and execute) or height_plan is not None,'prompted_batch_requires_instance_heights')
    policy['placement_verification']=options.get('placement_verification',{'mode':'semantic_rgb'})
    policy['rgb_first_verification']=options.get('rgb_first_verification',False)
    policy['batch_verification_mode']='end_of_batch' if batch else 'per_item'
    policy['skip_per_item_visual_verification']=batch is not None
    if batch:policy['batch_final_operator_statement']=batch['operator_statement']
    policy['decision_thresholds']=dict(options['decision_thresholds'],min_confidence=settings['confidence'])
    policy['threshold_operator_statement']='Current task: calibrate prompted model with existing labelled photographs.'
    policy['empty_bin_max_changed_fraction']=.05;policy['relax_bin_edge_overlap']=False
    policy['min_geometry_component_fraction']=options['min_geometry_component_fraction']
    if scope['basis']=='operator_supervised_trial' and options.get('live_object_admission',False):
        policy['live_object_admission']=dict(kind='operator-live-geometry-v1',
            operator_statement=scope['operator_statement'],physical_validation_completed=False)
    policy['verification_timeout_s']=options['frame_timeout_s']
    policy['verification_max_age_ms']=options['frame_max_age_ms']
    if policy.get('motion_timing'):
        policy['motion_timing'].update(options['motion_timing_overrides'])
    raw,pose,old_stats=raw_prompted_window(observation_timing(c),cam,station,reg,scene,min_fixed_markers=motion_marker_limit(b),
        max_marker_shift_px=initial_marker_shift_limit(b),initial_window_max_age_ms=options['frame_max_age_ms'],
        review_tolerance_factor=review_factor(b))
    projection_policy=copy.deepcopy(policy)
    for label,height in heights.items():projection_policy['models'][label]['object_height_mm']=height
    if item_heights is not None:
        for item in height_plan['items']:
            model=projection_policy['models'][item['label']]
            model['object_height_mm']=max(model['object_height_mm'],item['object_height_mm'])
    geometry=apply_prompted_geometry(geometry_for_policy(c,cam,station,reg,scene,pose,raw[-1]['image'],
        motion_shift_px=motion_shift_limit(b),min_fixed_markers=motion_marker_limit(b),review_tolerance_factor=review_factor(b)),projection_policy)
    geometry.map=lambda label,points:pixels_to_card(points,heights[label],pose,cam,station,reg)
    for label,height in heights.items():geometry.runtime['profiles'][label]['object_height_mm']=height
    geometry.runtime.update(model_family='yoloe',model_version=read_policy()['model']['sha256'],
        runtime_id=settings['settings_sha256'],prompt_mode='text',min_confidence=settings['confidence'])
    geometry.runtime=observation_timing(geometry.runtime)
    geometry.signature=digest(dict(mechanical_geometry=geometry.signature,
        perception_model=geometry.runtime['model_version'],settings=settings['settings_sha256'],heights_mm=heights,
        instance_height_plan=height_plan))
    geometry.runtime['known']['expected_camera_stream_id']=raw[-1]['observation']['source_stream_id']
    motion_options=read_motion_options()
    requires_empty=motion_options['destination_empty_check_enabled'] or policy['placement_verification']['mode']!='bin_occupied'
    empty=geometry.warp_empty_reference(cv2.imread(str(files['empty_image']))) if requires_empty else None
    by_label=(per_bin_empty_images(geometry,files,raw[-1]['observation']['source_stream_id'])
              if motion_options['destination_empty_check_enabled'] else {})
    lease=(base/c['known']['device_lease']).resolve();stops=[(base/p).resolve() for p in c['known']['stop_files']]

    def cycle(journal,owner=None):
        tracker=InstanceLedger(.85);controllers=[]
        def factory(index):
            if owner:owner.check()
            stats=PromptedStationStatistics(geometry.runtime) if owner else Statistics()
            capture=geometry.runtime if owner else dict(old_stats.c,model_version=geometry.runtime['model_version'],
                runtime_id=settings['settings_sha256'],prompt_mode='text')
            camera_source=StationCamera(capture,PromptedHTTP(settings),stats if owner else old_stats)
            prepared=raw[-1].get('_prepared_observation') if index==0 else None
            if prepared is not None and frame_count()==1:
                from .phase_dataflow import PreparedFrameCamera
                initial=dict(raw[-1]);initial.pop('_prepared_observation',None)
                initial['observation']=prepared
                verifier=getattr(camera_source.detector,'verify_identity',None)
                if verifier is not None:camera_source=PreparedFrameCamera(camera_source,initial,verifier)
            source=PromptedWindowedCamera(camera_source)
            if index:source.after(time.monotonic()*1000)
            motion=BatchMotion(geometry.runtime,journal,geometry,lease,stops,ledger/'pending.json',owner=owner) if owner else None
            if motion:
                motion.scene_check_mode=scene_check_mode(options)
                motion.sequence_visual_checked=False
                motion.sequence_scene_checks=0
            def setup_target(controller,choice):
                activation=configure_target_region(controller,choice,owner)
                statistics=PromptedStationStatistics(controller.geometry.runtime)
                controller.stats=statistics
                statistics.bind_next_capture=options.get('fuse_precommit_baseline',False)
                controller.source.source.statistics=statistics
                controller.motion.camera.statistics=statistics
                return activation
            cls=PromptedBatchController if batch else PromptedMotionController
            controller=cls(c,source,stats,motion,journal,geometry,empty,base,
                policy=policy,tracker=tracker,mode='known_physical' if owner else 'preview',
                prompt_settings=settings,trial_labels=scope['official_labels'],
                allowed_labels=allowed,empty_by_label=by_label,
                authorization_check=owner.check if owner else None,
                target_region_setup=setup_target if owner else None,item_heights=item_heights)
            for label,height in heights.items():
                controller.c['profiles'][label]['object_height_mm']=height
                controller.policy['models'][label]['object_height_mm']=height
            controllers.append(controller)
            return controller
        if batch and owner:
            report=run_registered_batch(factory,journal,batch,owner.check,time.monotonic)
            report['candidate_order']=controllers[0].candidate_order if controllers else []
            report['choice']={'label':material}
        else:
            controller=factory(0)
            if batch:
                from .prompted_batch import freeze_batch_plan
                planned=freeze_batch_plan(controller,batch['max_items'])
                if planned:controller.prepared_batch_item=planned[0]
            report=controller.cycle()
            if batch:
                report['batch_preview']=dict(planned_count=len(planned),
                    max_items=batch['max_items'],verification_mode='end_of_batch',
                    planning_mode='initial_scene_all_routes_before_motion',
                    frozen_plan=[dict(choice=p['selected'][2],
                        plan=[dict(step=n,command=d.decode(),duration_ms=ms) for n,d,ms in p['selected'][3]])
                        for p in planned])
        report.update(prompted_model=True,settings_sha256=settings['settings_sha256'],
            calibration_evidence_sha256=receipt['evaluation_sha256'],material_selection=material,
            observation_timing=dict(max_frame_age_ms=options['frame_max_age_ms'],frame_timeout_s=options['frame_timeout_s'],
                decision_frame_count=options['stable_frames'],
                original_max_frame_age_ms=c['max_age_ms'],operator_requested_increase_percent=200),
            execution_blockers=blockers,physical_validation_completed=False)
        report['motion_options']=motion_options
        report['requested_grasp_override']=grasp_override
        report['execution_scope']=scope
        from .execution_model_context import current_profile
        active=current_profile()
        if active is not None:
            report['execution_model']={k:copy.deepcopy(active.get(k)) for k in
                ('id','name','model_sha256','profile_sha256','upstream_configuration_sha256','family','binding_revision','reference_bank_sha256')}
        report.update(supplied_object_height_mm=supplied_height,effective_heights_mm=heights,
                      mechanical_height_profile_matches=mechanical_height_match,
                      height_application='selected_attempt_coordinate_projection',
                      grasp_z_mm={m:c['profiles'][m]['grasp_z_mm'] for m in allowed})
        if height_plan is not None:
            report.update(instance_height_plan=height_plan,height_application='per_instance_coordinate_projection',
                effective_heights_mm=None,mechanical_height_profile_matches=None)
        report['projected_region_clipping']=getattr(geometry,'region_projection_records',[])
        journal.save(report['task_id']+'.json',report)
        return report

    if not execute:
        with Journal(ROOT/'local-only/yoloe-recyclables/motion-preview-ledger') as journal:return cycle(journal)
    with Journal(ledger) as journal:
        need(journal.pending() is None and not (ledger/'batch-pending.json').exists(),'prompted_previous_motion_unresolved')
        with BatchLease(lease,stops,time.monotonic()+(batch['max_seconds'] if batch else 240)) as owner:
            activate_regions(old_stats,geometry,owner)
            return cycle(journal,owner)
