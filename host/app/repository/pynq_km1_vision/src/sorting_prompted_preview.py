"""Read-only TACO-to-existing-motion planner. Never opens UART or changes PL regions.

Uses three fresh captured frames to build a frozen preview. Frozen observations
and historical specimen mechanics do NOT qualify a route for physical execution.
"""
import argparse
import base64
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request
import uuid
import cv2
import numpy as np
from .prompted_recyclables import read_policy, settings_for, classify, digest
from .sorting_loop import Journal, mask_of, need, iou
from .sorting_io import Statistics
from .sorting_instances import InstanceLedger
from .sorting_relocated_run import (raw_window, motion_shift_limit, motion_marker_limit,
    initial_marker_shift_limit, initial_window_age_limit)
from .sorting_relocated_geometry import RelocatedGeometry, pixels_to_card
from .sorting_unified import UnifiedController,registered_bin_zone,external_bin_edge_review
from .sorting_unified_run import load_unified, apply_geometry, per_bin_empty_images
from .sorting_review_policy import review_factor
from .prompted_motion_policy import read_motion_options
from .sorting_unified_plan import slot_rect
from .prompted_region_geometry import geometry_for_policy,apply_prompted_geometry
from .prompted_observation import raw_prompted_window,frame_count

ROOT = Path(__file__).resolve().parents[2]


def project_row(policy, settings, result, image, profiles,reference_profiles=None,source_roi=None):
    need(result['settings_sha256'] == settings['settings_sha256'], 'prompt_settings_changed')
    need(result['model_sha256'] == policy['model']['sha256'], 'prompt_model_changed')
    need(result['image_size'] == [image.shape[1],image.shape[0]], 'prompt_image_size_changed')
    need(result['source_bgr_sha256'] == hashlib.sha256(image.tobytes()).hexdigest(), 'prompt_mask_image_binding_changed')
    objects, masks, excluded, decisions = [], [], [], []
    for instance in result['instances']:
        mask = mask_of(instance, result['image_size'])
        zone=registered_bin_zone(mask,profiles,result['image_size'],
            reference_profiles=reference_profiles,source_roi=source_roi)
        if zone is None:
            zone=instance.get('station_zone','source')
            if zone not in ('source','bin_paper','bin_plastic','bin_metal'):zone='source'
        row = classify(policy,settings,instance,zone)
        decisions.append(row)
        if row['material'] is None:
            excluded.append(row)
        else:
            objects.append(dict(row,label=row['material']))
            masks.append(mask)
    # Unknown/excluded masks cannot be mapped to a false material height.
    # Conservatively block the whole preview rather than drop their obstacles.
    return dict(image=image,observation=dict(instances=objects),masks=masks,
                excluded=excluded,decisions=decisions)


def stable_official_labels(rows, minimum_iou=.7):
    """Match each material and mask uniquely; keep raw classes for reporting."""
    first = rows[0]
    need(all(not r['excluded'] for r in rows), 'excluded_or_unknown_object_in_scene')
    for row in rows[1:]:
        need(len(row['masks']) == len(first['masks']), 'prompt_instance_count_changed')
        matched=set()
        for obj,mask in zip(first['observation']['instances'],first['masks']):
            hits=[i for i,(other,m) in enumerate(zip(row['observation']['instances'],row['masks']))
                  if obj['label']==other['label'] and iou(mask,m)>=minimum_iou]
            need(len(hits)==1 and hits[0] not in matched,'prompt_material_or_mask_unstable')
            matched.add(hits[0])
            # The weakest frame controls the semantic gate.
            other=row['observation']['instances'][hits[0]]
            obj['semantic_gate_passed'] &= other['semantic_gate_passed']
    return first


class PromptedPreviewController(UnifiedController):
    def __init__(self,*args,**kwargs):
        self.motion_options=read_motion_options()
        super().__init__(*args,**kwargs)
        self.c['stable_frames']=self.motion_options.get('stable_frames',3)

    def target_bin_edge_review(self,mask,rect):
        # The bin ROI is a coarse image rectangle. A registered source target
        # may touch its outer strip while remaining outside the protected core.
        # Use the normal 5% bound; do not inherit legacy expanded tolerances or
        # change destination occupancy, obstacle masks, or mechanical checks.
        pickup=self.c.get('pickup_source_roi')
        ys,xs=np.nonzero(mask)
        if pickup is None or not len(xs):return None
        x,y=float(xs.mean()),float(ys.mean());a,b,d,e=pickup
        if not (a<=x<d and b<=y<e):return None
        review=external_bin_edge_review(mask,rect,self.c['image_size'],self.policy.get('bin_padding_px',5),.05)
        if review is not None:
            review.update(scope='registered_source_target_at_coarse_bin_edge',pickup_source_roi=list(pickup),
                          centroid_inside_pickup=True,expanded_legacy_tolerance_used=False)
        return review

    def free_slots(self,row,label,ignore_object=None):
        if self.motion_options['destination_empty_check_enabled']:
            return super().free_slots(row,label,ignore_object)
        slots=[]
        for slot in self.policy['bin_slots'][label]:
            # Disabling occupancy is not qualification of a new release path.
            if self.mode=='known_physical' and slot['origin']=='candidate':
                continue
            self.slot_checks.append(dict(label=label,slot_id=slot['slot_id'],
                roi=slot_rect(self.c['profiles'][label],slot),
                destination_empty_check_enabled=False,semantic_occupancy_checked=False,
                rgb_empty_reference_checked=False,reason='operator_disabled_destination_empty_check'))
            slots.append(slot)
        return slots

    def candidate(self, row, target, track_id):
        obj,_=target
        need(obj.get('station_grasp_candidate',True),'station_filter_target_excluded')
        need(not row.get('excluded'), 'excluded_or_unknown_object_in_scene')
        need(obj['semantic_gate_passed'], 'prompt_class_or_zone_confidence_below_threshold')
        need(obj['zone']=='source','prompt_target_already_in_bin')
        separation=self.c['min_separation_px']
        if not self.motion_options.get('minimum_silhouette_separation_check_enabled',True):
            self.c['min_separation_px']=0
        try:
            choice,plan=super().candidate(row,target,track_id)
        finally:
            self.c['min_separation_px']=separation
        choice.update(official_label=obj['official_label'],confidence=obj['confidence'],
                      decision_threshold=obj['decision_threshold'],
                      execution_basis='frozen_preview_only_new_taxonomy_not_physically_qualified',
                      automatic_motion_allowed=False)
        choice['destination_empty_check_enabled']=self.motion_options['destination_empty_check_enabled']
        choice['minimum_silhouette_separation_check_enabled']=self.motion_options.get('minimum_silhouette_separation_check_enabled',True)
        return choice,plan


def preview(url, settings, *, binding=None, output=None):
    policy=read_policy()
    need(settings_for(policy,settings)==settings,'invalid_prompt_settings_snapshot')
    if binding is None:
        pointer_path=ROOT/'local-only/station-grasp-integration/active.json'
        pointer=json.loads(pointer_path.read_text(encoding='utf-8'))
        binding=(pointer_path.parent/pointer['path']).resolve()
        need(hashlib.sha256(binding.read_bytes()).hexdigest()==pointer['sha256'],'active_binding_hash_changed')
    bundle, motion_policy=load_unified(binding)
    b,c,base,cam,station,reg,scene,files=bundle
    raw,pose,_=raw_prompted_window(c,cam,station,reg,scene,min_fixed_markers=motion_marker_limit(b),
        max_marker_shift_px=initial_marker_shift_limit(b),initial_window_max_age_ms=initial_window_age_limit(b,c),
        review_tolerance_factor=review_factor(b))
    geometry=apply_prompted_geometry(geometry_for_policy(c,cam,station,reg,scene,pose,raw[-1]['image'],
        motion_shift_px=motion_shift_limit(b),min_fixed_markers=motion_marker_limit(b),
        review_tolerance_factor=review_factor(b)),motion_policy)
    for material in policy['materials']:
        need(policy['materials'][material]['bin_id']==c['profiles'][material]['bin_id'],'material_destination_binding_mismatch')
    # Height changes ray-plane projection only. Existing grip/hover Z and measured
    # reachability domains are preserved. Revalidate mechanics before execution.
    geometry.map=lambda label,points:pixels_to_card(points,settings['heights_mm'][label],pose,cam,station,reg)
    empty=geometry.warp_empty_reference(cv2.imread(str(files['empty_image'])))
    motion_options=read_motion_options()
    empty_by_label=(per_bin_empty_images(geometry,files,raw[-1]['observation']['source_stream_id'])
                    if motion_options['destination_empty_check_enabled'] else {})
    task=uuid.uuid4().hex
    folder=Path(output or ROOT/'local-only/yoloe-recyclables/previews')/task
    folder.mkdir(parents=True,exist_ok=False)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    rows=[]
    for i,frame in enumerate(raw):
        ok,png=cv2.imencode('.png',frame['image']);need(ok,'cannot_encode_preview_frame')
        data=png.tobytes();(folder/f'{i}.png').write_bytes(data)
        payload=json.dumps(dict(image_base64=base64.b64encode(data).decode(),control_only=True)).encode()
        request=urllib.request.Request(url+'/detect',data=payload,headers={'Content-Type':'application/json'})
        with opener.open(request,timeout=25) as response:result=json.load(response)
        result.pop('overlay_base64',None)
        (folder/f'{i}.json').write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8')
        rows.append(project_row(policy,settings,result,frame['image'],geometry.runtime['profiles'],
            reference_profiles=geometry.original['profiles'],source_roi=geometry.runtime['source_roi']))
    report=dict(status='blocked',task_id=task,settings_sha256=settings['settings_sha256'],
        policy_sha256=digest(policy),model_sha256=policy['model']['sha256'],
        physical_motion_attempted=False,automatic_motion_allowed=False,
        execution_blockers=['new_taxonomy_and_height_require_physical_validation'],
        heights_mm=settings['heights_mm'],projected_regions=geometry.runtime['statistics']['regions'],
        frames=[dict(stream_id=r['observation']['source_stream_id'],frame_id=r['observation']['source_frame_id'],
                     image_sha256=r['source_bytes_sha256']) for r in raw],
        candidates=[],rejected=[],decisions=rows[-1]['decisions'],artifact_dir=str(folder),
        frozen_snapshot=True)
    report['motion_options']=motion_options
    report['projected_region_clipping']=getattr(geometry,'region_projection_records',[])
    try:
        row=stable_official_labels(rows)
        with Journal(folder/'ledger') as journal:
            controller=PromptedPreviewController(c,None,Statistics(),None,journal,geometry,empty,base,
                policy=motion_policy,tracker=InstanceLedger(.7),mode='preview',empty_by_label=empty_by_label)
            # Restore unrelaxed confidence/scene thresholds for the new model.
            controller.c['min_confidence']=policy['detection_floor']
            controller.c['min_scene_iou']=.7
            controller.c['max_scene_changed_fraction']=.05
            controller.c['max_other_bin_changed_fraction']=.05
            controller.policy['empty_bin_max_changed_fraction']=.05
            controller.policy['relax_bin_edge_overlap']=False
            for material,height in settings['heights_mm'].items():
                controller.c['profiles'][material]['object_height_mm']=height
                controller.policy['models'][material]['object_height_mm']=height
            for other in rows[1:]:controller.unchanged(row,other)
            for index,target in enumerate(controller.source_objects(row)):
                try:
                    choice,plan=controller.candidate(row,target,'preview-'+str(index))
                    report['candidates'].append(dict(choice=choice,
                        steps=[dict(name=n,command=p.decode(),duration_ms=d) for n,p,d in plan]))
                except ValueError as exc:
                    report['rejected'].append(dict(official_label=target[0]['official_label'],reason=str(exc)))
            report['status']='preview_ready' if report['candidates'] else 'no_pickable_instance'
            report['slot_checks']=controller.slot_checks
    except ValueError as exc:
        report['reason']=str(exc)
    report['generated_at']=time.time()
    (folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8811')
    args=parser.parse_args()
    result=preview(args.url,json.load(sys.stdin))
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
