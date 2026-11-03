"""One transaction per selected instance, shared by single and batch sorting."""
import copy
import hashlib
import time
import uuid
import cv2
import numpy as np
from .sorting_loop import Controller, LABELS, fingerprint, need, resolve_material_duplicates
from .sorting_instances import associate
from .sorting_relocated import RelocatedController
from .sorting_scene import adjust_exposure_reference
from .sorting_unified_plan import compile_route, slot_rect, validate_policy, apply_decision_thresholds
from .sorting_unified_motion import movement_wait
from .sorting_review_policy import review_factor
from .static_visual_verification import roi_mask
from .bin_occupancy import bin_occupancy, verification_options
from .sorting_item_heights import rank_candidates
from .contour_grasp import contour_section


def registered_bin_zone(mask,profiles,image_size,*,source_roi=None,reference_profiles=None):
    """Keep 95% containment; allow a bound bin ROI only outside the source.

    A small projection shift must not turn a known bin occupant into an
    unregistered source obstacle. This does not certify a new placement.
    """
    area=int(mask.sum())
    if not area:return None
    for label,profile in profiles.items():
        if int((mask&roi_mask(profile['bin_roi'],image_size)).sum())>=area*.95:
            return 'bin_'+label
    if source_roi is None or (mask&roi_mask(source_roi,image_size)).any():return None
    matches=[label for label,profile in (reference_profiles or {}).items()
             if label in profiles and int((mask&roi_mask(profile['bin_roi'],image_size)).sum())>=area*.95]
    return 'bin_'+matches[0] if len(matches)==1 else None


def registered_workspace_roi(current,reference):
    """The registered pickup region only; bins never enlarge the height pool.

    Keep an immutable pickup ROI while per-target RGB evidence uses a crop.
    Without a bound layout retain legacy behavior.
    """
    if not reference or 'source_roi' not in reference or not reference.get('profiles'):return None
    rect=current.get('pickup_source_roi',current.get('statistics',{}).get('regions',{}).get('unified_source',current['source_roi']))
    w,h=current['image_size'];x,y,x1,y1=rect
    return [max(0,x),max(0,y),min(w,x1),min(h,y1)]


def external_bin_edge_review(mask,rect,image_size,padding,limit):
    """A bounded outside-centroid edge test; never accepts the protected core."""
    area=int(mask.sum())
    if not area:return None
    x0,y0,x1,y1=rect;ys,xs=np.nonzero(mask)
    if x0<=float(xs.mean())<x1 and y0<=float(ys.mean())<y1:return None
    region=roi_mask(rect,image_size)
    dx=int(np.ceil((x1-x0)*limit));dy=int(np.ceil((y1-y0)*limit))
    inner=[x0+dx,y0+dy,x1-dx,y1-dy]
    if inner[0]>=inner[2] or inner[1]>=inner[3]:return None
    if (mask&roi_mask(inner,image_size)).any():return None
    padded=cv2.dilate(mask.astype('uint8'),np.ones((padding*2+1,padding*2+1),np.uint8)).astype(bool)
    pixels=int((padded&region).sum());fraction=pixels/int(region.sum())
    if not pixels or fraction>limit:return None
    return dict(padded_overlap_px=pixels,slot_area_px=int(region.sum()),
        padded_slot_overlap_fraction=fraction,max_slot_overlap_fraction=limit,
        object_overlap_fraction=float((mask&region).sum()/area),centroid_outside_slot=True,
        inner_slot_overlap_px=0,inner_slot_roi=inner,interpretation='external_object_on_coarse_rectangle_edge_only')


class UnifiedController(RelocatedController):
    def __init__(self, original, source, statistics, motion, journal, geometry, empty,
                 base, *, policy, tracker, mode='preview', authorization_check=None, allowed_labels=None,
                 target_region_setup=None,
                 empty_by_label=None,
                 clock=time.monotonic, sleep=time.sleep, item_heights=None):
        validate_policy(policy, original)
        self.placement_verification=verification_options(policy)
        need(self.placement_verification['mode']!='bin_occupied' or
             policy.get('batch_verification_mode','per_item')=='per_item' or
             (policy.get('batch_verification_mode')=='end_of_batch' and
              policy.get('skip_per_item_visual_verification') is True and
              bool(policy.get('batch_final_operator_statement'))),
             'bin_occupancy_requires_per_item_verification')
        if mode == 'known_physical':
            need(callable(authorization_check), 'unified_physical_owner_required')
            authorization_check()
            need('detection_window_px' not in policy or callable(target_region_setup),
                 'full_frame_motion_requires_target_region_binding')
            need(statistics.backend == 'pl_board_local' and
                 statistics.cfg['regions'] == geometry.runtime['statistics']['regions'],
                 'unified_requires_bound_board_regions')
        super().__init__(original, source, statistics, motion, journal, geometry, empty,
                         base, mode='simulation' if mode == 'simulation' else 'preview',
                         motion_profile='qualified', clock=clock, sleep=sleep)
        self.mode, self.policy, self.tracker = mode, copy.deepcopy(policy), tracker
        self.c=apply_decision_thresholds(self.c,policy)
        self.c['max_objects'] = policy['max_objects']
        self.authorization_check = authorization_check
        self.target_region_setup=target_region_setup
        self.empty_by_label={} if empty_by_label is None else dict(empty_by_label)
        self.allowed_labels=frozenset(LABELS if allowed_labels is None else allowed_labels)
        need(self.allowed_labels and self.allowed_labels.issubset(LABELS),'invalid_unified_material_selection')
        self.final_source = None
        self.verification_attempts=[]
        self.batch_context=None
        self.slot_checks=[]
        self.item_heights=item_heights
        self.c['pickup_source_roi']=copy.deepcopy(self.c['source_roi'])
        self.candidate_order=[]
        self.ranked_plans=[]

    def movement_wait(self, name, duration):
        return movement_wait(self.policy, name, duration)

    def verification_frames(self,barrier,rows=None,deadline=None):
        self.source.after(barrier)
        rows=[] if rows is None else rows
        original,original_age=self.c['frame_timeout_s'],self.c['max_age_ms']
        inner=getattr(self.source,'source',None);inner_config=None if inner is None else inner.c
        self.c['frame_timeout_s']=self.policy.get('verification_timeout_s',5)
        if deadline is not None:self.c['frame_timeout_s']=max(.05,min(self.c['frame_timeout_s'],deadline-self.clock()))
        self.c['max_age_ms']=self.policy.get('verification_max_age_ms',int(self.c['frame_timeout_s']*1000))
        if inner is not None:inner.c=dict(inner.c,max_age_ms=self.c['max_age_ms'])
        try:
            count=1 if self.placement_verification['mode']=='bin_occupied' else self.c['stable_frames']
            for _ in range(count):rows.append(self.acquire(not_before_ms=barrier))
            return rows
        finally:
            self.c['frame_timeout_s'],self.c['max_age_ms']=original,original_age
            if inner is not None:inner.c=inner_config

    def verify_rgb_first(self,before,target,choice,barrier,rows,deadline):
        """A new lossless frame can satisfy the existing RGB OR branch first."""
        inner=getattr(self.source,'source',None)
        detector=getattr(inner,'detector',None)
        identity=getattr(detector,'verify_identity',None)
        if not callable(identity) or not callable(getattr(inner,'raw',None)):
            return None
        marker_detector=getattr(detector,'frame_marker_detector',getattr(detector,'local_marker_detector',None))
        if marker_detector is None:return None
        identity()
        self.source.after(barrier)
        row=inner.raw(max(.05,min(self.c['frame_timeout_s'],deadline-self.clock())))
        # Marker-only detection avoids a full semantic inference for this raw frame.
        self.accept_frame(row,barrier,semantic=False,
            layout_check=lambda image:self.geometry.check(image,marker_detector=marker_detector))
        semantic_fallback=False
        try:
            result=self.verify_bin_occupied(before,target[0]['label'],choice,row)
        except ValueError as exc:
            if str(exc)!='unified_target_bin_empty':raise
            predict=getattr(detector,'predict_frame',None)
            if not callable(predict):raise
            need(self.clock()<deadline,'unified_verification_window_expired')
            observation=predict(row,prompt_mode=self.c['prompt_mode'])
            for key in ('model_version','runtime_id','image_size','prompt_mode'):
                need(observation[key]==self.c[key],'observation_identity_mismatch:'+key)
            need(hashlib.sha256(row['image'].tobytes()).hexdigest()==row['bgr_sha256'],'decoded_image_changed')
            observation['model_input_image_sha256']=observation['image_sha256']
            observation.update({k:v for k,v in row['observation'].items() if k!='instances'})
            row['observation']=observation;row['masks']=[]
            from .sorting_loop import finite,mask_of,LABELS
            for obj in observation['instances']:
                need(obj['label'] in LABELS and finite(obj['confidence']) and 0<=obj['confidence']<=1,'invalid_material_instance')
                row['masks'].append(mask_of(obj,self.c['image_size']))
            row['perception_kind']='same_frame_semantic_fallback'
            result=self.verify_bin_occupied(before,target[0]['label'],choice,row)
            semantic_fallback=True
        identity()
        age=self.policy.get('verification_max_age_ms',int(self.policy.get('verification_timeout_s',5)*1000))
        need(self.clock()<=deadline,'unified_verification_window_expired')
        need(0<=self.clock()*1000-row['observation']['capture_started_ms']<=age,'unified_final_window_expired')
        rows.append(row)
        result['dataflow']=dict(rgb_first=True,semantic_inference_skipped=not semantic_fallback,same_frame_fallback=semantic_fallback,
            geometry_backend='pynq_ps_with_pc_marker_only',difference_backend='pynq_ps_cpu')
        return result

    def verify_within_window(self,before,target,choice,barrier,rows):
        deadline=self.clock()+self.policy.get('verification_timeout_s',5)
        self.verification_attempts=[]
        if self.placement_verification['mode']=='bin_occupied' and self.policy.get('rgb_first_verification',False):
            try:
                result=self.verify_rgb_first(before,target,choice,barrier,rows,deadline)
                if result is not None:
                    self.verification_attempts.append(dict(last_frame=rows[-1]['observation']['source_frame_id'],passed=True,rgb_first=True))
                    return result
            except ValueError as exc:
                if str(exc)!='unified_target_bin_empty':raise
                self.verification_attempts.append(dict(passed=False,rgb_first=True,reason=str(exc)))
        need(self.clock()<deadline,'unified_verification_window_expired')
        self.verification_frames(barrier,rows,deadline=deadline)
        while True:
            count=1 if self.placement_verification['mode']=='bin_occupied' else self.c['stable_frames']
            recent=rows[-count:]
            need(self.clock()<=deadline,'unified_verification_window_expired')
            self.verification_remaining_s=max(.05,deadline-self.clock())
            try:
                result=self.verify_selected(before,target,choice,recent)
                age=self.policy.get('verification_max_age_ms',int(self.policy.get('verification_timeout_s',5)*1000))
                need(all(0<=self.clock()*1000-r['observation']['capture_started_ms']<=age for r in recent),
                     'unified_final_window_expired')
                self.verification_attempts.append(dict(last_frame=recent[-1]['observation']['source_frame_id'],passed=True))
                return result
            except ValueError as exc:
                reason=str(exc)
                # Wait only for visual consistency after completed motion.
                # Identity, baseline, camera, transport and marker failures
                # remain immediate errors. No arm command is retried here.
                retryable=reason.startswith('unified_') or reason in (
                    'instance_count_changed','instance_missing_split_or_moved','instance_identity_ambiguous_or_merged')
                self.verification_attempts.append(dict(last_frame=recent[-1]['observation']['source_frame_id'],passed=False,reason=reason))
                if not retryable or self.clock()>=deadline:raise
            original,original_age=self.c['frame_timeout_s'],self.c['max_age_ms']
            inner=getattr(self.source,'source',None);inner_config=None if inner is None else inner.c
            self.c['frame_timeout_s']=max(.05,deadline-self.clock())
            self.c['max_age_ms']=self.policy.get('verification_max_age_ms',int(self.policy.get('verification_timeout_s',5)*1000))
            if inner is not None:inner.c=dict(inner.c,max_age_ms=self.c['max_age_ms'])
            try:rows.append(self.acquire(not_before_ms=barrier))
            finally:
                self.c['frame_timeout_s'],self.c['max_age_ms']=original,original_age
                if inner is not None:inner.c=inner_config

    def known_bin_zone(self,mask):
        original=getattr(getattr(self,'geometry',None),'original',{})
        return registered_bin_zone(mask,self.c['profiles'],self.c['image_size'],
            source_roi=self.c.get('source_roi'),reference_profiles=original.get('profiles'))

    def source_objects(self,row):
        if 'detection_window_px' not in self.policy:return super().source_objects(row)
        pairs=self.obstacle_objects(row)
        # Bins and fixed marker patches remain protected, independently of
        # material selection. Everything else in the full frame is observed.
        fixed=np.zeros(row['image'].shape[:2],bool)
        for rect in self.c['layout']['static_rois']:fixed|=roi_mask(rect,self.c['image_size'])
        original=getattr(getattr(self,'geometry',None),'original',{})
        workspace=registered_workspace_roi(self.c,original)
        work_mask=None if workspace is None else roi_mask(workspace,self.c['image_size'])
        # Keep every raw mask. Bin membership excludes pick/height candidates
        # while the bins remain protected by the existing route checks.
        found=[(o,m) for o,m in pairs if m.any() and (work_mask is None or (m&work_mask).any()) and not (m&fixed).any() and
               self.known_bin_zone(m) is None]
        need(len(found)<=self.policy.get('max_scene_instances',32),'unified_too_many_scene_instances')
        return sorted(found,key=lambda pair:(pair[0]['label'],pair[0]['bbox_xyxy']))

    def obstacle_objects(self,row):
        """All perceived regions, with the existing duplicate policy unchanged."""
        pairs=list(zip(row['observation']['instances'],row['masks']))
        if self.c.get('duplicate_instance_policy') is not None:
            pairs,resolution=resolve_material_duplicates(row['observation']['instances'],row['masks'],self.c['duplicate_instance_policy'])
            row['duplicate_resolution']=resolution
        return [(o,m) for o,m in pairs if m.any()]

    def obstacle_points(self,other,points,target,source_ids):
        # External objects need no invented height. Their silhouettes are
        # checked on the measured target plane as an image obstruction, while
        # registered source neighbours retain their own measured planes.
        return self.map_object(other if id(other) in source_ids else target,points)

    def outside_objects(self,row):
        work={id(o) for o,_ in self.source_objects(row)}
        return [(o,m) for o,m in zip(row['observation']['instances'],row['masks']) if id(o) not in work]

    def evidence_rect(self,mask,pixel):
        ys,xs=np.nonzero(mask);w,h=self.c['image_size']
        need(len(xs) and xs.min()>0 and ys.min()>0 and xs.max()<w-1 and ys.max()<h-1,
             'unified_target_clipped_by_camera')
        rect=[max(0,int(xs.min())-5),max(0,int(ys.min())-5),
              min(w,int(xs.max())+6),min(h,int(ys.max())+6)]
        rectangles={tuple(rect)};x,y=pixel
        for r in self.c['layout']['static_rois']+[p['bin_roi'] for p in self.c['profiles'].values()]:
            next_rectangles=set()
            for a,b,d,e in rectangles:
                if not (max(a,r[0])<min(d,r[2]) and max(b,r[1])<min(e,r[3])):
                    next_rectangles.add((a,b,d,e));continue
                for candidate in ((a,b,min(d,r[0]),e),(max(a,r[2]),b,d,e),
                                  (a,b,d,min(e,r[1])),(a,max(b,r[3]),d,e)):
                    aa,bb,dd,ee=candidate
                    if aa<=x<dd and bb<=y<ee and dd-aa>=8 and ee-bb>=8:next_rectangles.add(candidate)
            rectangles=next_rectangles
        scored=[(int(mask[b:e,a:d].sum()),(d-a)*(e-b),(a,b,d,e)) for a,b,d,e in rectangles]
        need(scored,'unified_target_near_protected_region')
        count,_,chosen=max(scored)
        need(count>=self.c['min_bin_gain_px'],'unified_target_has_insufficient_clear_rgb_evidence')
        return list(chosen)

    def unchanged(self, before, after):
        self.check_height_scene(before,after)
        associate(self.source_objects(before), self.source_objects(after), self.c['min_scene_iou'])
        # Bin instance ordering is independent too; both semantic and RGB scene
        # checks must hold before committing a route.
        old,new=self.outside_objects(before),self.outside_objects(after)
        associate(old, new, self.c['min_scene_iou'])
        changed = self.changed_pixels(before, after)
        for _,mask in self.source_objects(before):
            need(float(changed[mask].mean())<=self.c['max_scene_changed_fraction'],'unified_observed_object_rgb_changed')
        for rect in [self.c['source_roi']]+[p['bin_roi'] for p in self.c['profiles'].values()]:
            need(float(changed[roi_mask(rect,self.c['image_size'])].mean()) <= self.c['max_scene_changed_fraction'],
                 'unified_scene_rgb_changed')

    def check_height_scene(self,before,after):
        if self.item_heights is not None:
            pairs=self.source_objects(after)
            associate(self.source_objects(before),pairs,self.c['min_scene_iou'])
            self.item_heights.check_scene(pairs,after['observation']['source_stream_id'],self.c['min_scene_iou'])

    def frozen_batch_frame(self,before):
        # The motion camera is intentionally JPEG/marker-only. Admission and
        # RGB provenance need the original lossless station capture instead;
        # raw() does not rerun semantic inference or replan the frozen route.
        camera=self.source.source;detector=getattr(camera,'detector',None)
        need(callable(getattr(detector,'verify_identity',None)),'frozen_batch_identity_check_required')
        detector.verify_identity();barrier=self.clock()*1000
        self.source.after(barrier)
        current=camera.raw(self.c['frame_timeout_s'])
        need(current.get('source_encoding')=='png','rgb_verification_requires_lossless_frame')
        need(current['observation']['source_stream_id']==before['observation']['source_stream_id'],
             'motion_camera_stream_changed')
        self.accept_frame(current,barrier,semantic=False)
        detector.verify_identity()
        return current

    def observe(self):
        rows = []
        started = self.clock()
        for _ in range(self.c['stable_frames']):
            row = self.acquire()
            if rows:self.unchanged(rows[0], row)
            rows.append(row)
            need(self.clock()-started <= self.c['observe_timeout_s'], 'unified_observation_timeout')
        need(all(0 <= self.clock()*1000-r['observation']['capture_started_ms'] <= self.c['max_age_ms']
                 for r in rows), 'unified_stable_window_expired')
        return rows

    def bin_edge_review(self,mask,rect):
        """Review a coarse rectangular-bin edge, never an in-bin object.

        The fraction measures the padded silhouette / whole slot rectangle.
        An outside centroid AND an empty protected inner rectangle are mandatory;
        every original silhouette remains present for normal obstacle checks.
        This is a numerical image review, not full-link collision checking.
        """
        if not self.policy.get('relax_bin_edge_overlap',False):return None
        return external_bin_edge_review(mask,rect,self.c['image_size'],self.policy.get('bin_padding_px',5),.05*review_factor(self.policy))

    def target_bin_edge_review(self,mask,rect):
        return self.bin_edge_review(mask,rect)

    def free_slots(self, row, label,ignore_object=None):
        p = self.c['profiles'][label]
        pairs=[(o,m) for o,m in zip(row['observation']['instances'],row['masks']) if o is not ignore_object]
        masks=[m for _,m in pairs]
        union = np.logical_or.reduce(masks) if masks else np.zeros(row['image'].shape[:2],bool)
        padding = self.policy.get('bin_padding_px', 5)
        occupied = cv2.dilate(union.astype('uint8'), np.ones((padding*2+1,padding*2+1),np.uint8)).astype(bool)
        empty_reference=self.empty_by_label.get(label,self.empty_reference)
        need(empty_reference is not None, 'unified_empty_reference_required')
        empty,_ = adjust_exposure_reference(empty_reference,row['image'],dict(self.c['layout'],image_size=self.c['image_size']))
        difference = np.abs(row['image'].astype('int16')-empty.astype('int16')).sum(2) > self.c['rgb_difference_threshold']
        result = []
        empty_limit=self.policy.get('empty_bin_max_changed_fraction',self.c['max_scene_changed_fraction'])
        for slot in self.policy['bin_slots'][label]:
            rect = slot_rect(p,slot)
            region = roi_mask(rect,self.c['image_size'])
            boundary=[];blocking=[]
            for obj,mask in pairs:
                reviewed=self.bin_edge_review(mask,rect)
                if reviewed is None:blocking.append(mask)
                else:boundary.append(dict(label=obj['label'],**reviewed))
            if boundary:
                union_block=np.logical_or.reduce(blocking) if blocking else np.zeros_like(region)
                occupied_block=cv2.dilate(union_block.astype('uint8'),np.ones((padding*2+1,padding*2+1),np.uint8)).astype(bool)
            else:occupied_block=occupied
            occupied_px=int((occupied_block&region).sum());changed=float(difference[region].mean())
            self.slot_checks.append(dict(label=label,slot_id=slot['slot_id'],roi=rect,
                occupied_px=occupied_px,changed_fraction=changed,
                raw_occupied_px=int((occupied&region).sum()),boundary_reviews=boundary,
                max_changed_fraction=empty_limit,
                reason='slot_object_detected' if occupied_px else
                       'empty_reference_mismatch' if changed>empty_limit else 'free'))
            if occupied_px or changed > empty_limit:
                continue
            # Candidate extra slots are previewable, never silently promoted
            # to measured release paths merely because the box looks empty.
            if self.mode == 'known_physical' and slot['origin'] == 'candidate':
                continue
            result.append(slot)
        return result

    def candidate(self, row, target, track_id):
        if self.item_heights is None:
            return self.candidate_at_height(row,target,track_id)
        obj,_=target
        measurement=self.item_heights.measurement(obj)
        label=obj['label'];height=measurement['object_height_mm']
        profile=self.c['profiles'][label];model=self.policy['models'][label]
        previous=(profile['object_height_mm'],model['object_height_mm'])
        profile['object_height_mm']=model['object_height_mm']=height
        try:
            need(profile['grasp_z_mm'] < height,'instance_top_below_existing_grasp_z:object_height_mm='+
                 str(height)+',grasp_z_mm='+str(profile['grasp_z_mm']))
            choice,plan=self.candidate_at_height(row,target,track_id)
            choice['height_source']='operator_per_instance'
            choice['height_item_id']=measurement['item_id']
            choice['grasp_z_mm']=profile['grasp_z_mm']
            choice['height_changes_grasp_z']=False
            return choice,plan
        finally:
            profile['object_height_mm'],model['object_height_mm']=previous

    def map_object(self,obj,points):
        if self.item_heights is None:
            return self.geometry.map(obj['label'],points)
        from .sorting_relocated_geometry import pixels_to_card
        height=self.item_heights.measurement(obj)['object_height_mm']
        return pixels_to_card(points,height,self.geometry.pose,self.geometry.camera,
                              self.geometry.station,self.geometry.registration)

    def candidate_at_height(self, row, target, track_id):
        obj,mask = target
        need(obj['label'] in self.allowed_labels and obj['confidence']>=self.c['min_confidence'],
             'unified_target_class_or_confidence')
        p = self.c['profiles'][obj['label']]
        area = int(mask.sum())
        live=self.policy.get('live_object_admission')
        if live:
            need(isinstance(live,dict) and live.get('kind')=='operator-live-geometry-v1' and
                 live.get('operator_statement') and live.get('physical_validation_completed') is False,
                 'invalid_live_object_admission')
            need(area>0,'unified_empty_object_mask')
        else:
            need(p['min_area_px'] <= area <= p['max_area_px'],
                 'unified_specimen_area:observed='+str(area)+',allowed='+str(p['min_area_px'])+'..'+str(p['max_area_px']))
        count,components,component_stats,_=cv2.connectedComponentsWithStats(mask.astype('uint8'),connectivity=8)
        largest=1+int(np.argmax(component_stats[1:,cv2.CC_STAT_AREA]))
        main_area=int(component_stats[largest,cv2.CC_STAT_AREA])
        minimum_main_fraction=self.policy.get('min_geometry_component_fraction',.99)
        need(main_area>=area*minimum_main_fraction,'unified_fragmented_specimen')
        cleanup=dict(raw_area_px=area,geometry_area_px=main_area,removed_island_px=area-main_area,
                     minimum_main_fraction=minimum_main_fraction,raw_mask_preserved_for_verification=True)
        if count>2:mask=components==largest
        overlaps=[(pf['bin_roi'],float(mask[pf['bin_roi'][1]:pf['bin_roi'][3],pf['bin_roi'][0]:pf['bin_roi'][2]].sum())/main_area)
                  for pf in self.c['profiles'].values()]
        overlap=max(fraction for _,fraction in overlaps)
        edge_reviews=[]
        for rect,fraction in overlaps:
            if fraction<=.05:continue
            review=self.target_bin_edge_review(mask,rect)
            need(review is not None,'unified_target_crosses_bin_boundary')
            edge_reviews.append(dict(bin_roi=rect,**review))
        ys,xs = np.nonzero(mask)
        card = self.map_object(obj,np.c_[xs,ys])
        values,vectors = np.linalg.eigh(np.cov(card.T))
        elongated=values[1] >= values[0]*1.2
        need(elongated or live, 'unified_grasp_axis_ambiguous')
        axis = vectors[:,1]
        angle = float(np.degrees(np.arctan2(axis[1],axis[0])) % 180)
        if live and not elongated:
            # A near-round object has no meaningful principal direction. Use
            # the existing planner's zero-axis route, preserving all waypoints.
            angle=0.0
            axis=np.array([1.,0.])
            vectors=np.column_stack(([0.,1.],axis))
        width,length = float(np.ptp(card@vectors[:,0])),float(np.ptp(card@axis))
        if live and not elongated:
            width,length=float(np.ptp(card[:,1])),float(np.ptp(card[:,0]))
        if live:
            jaw_limit=self.policy['jaw_open_span_mm']-2*self.policy['obstacle_margin_mm']
            need(0<width<=jaw_limit and length>0,
                 'unified_object_exceeds_gripper:width_mm='+str(round(width,2))+',limit_mm='+str(jaw_limit))
        else:
            need(p['width_bounds_mm'][0] <= width <= p['width_bounds_mm'][1] and
                 p['length_bounds_mm'][0] <= length <= p['length_bounds_mm'][1], 'unified_known_profile_dimensions')
        distance = cv2.distanceTransform(np.pad(mask.astype('uint8'),1),cv2.DIST_L2,5)[1:-1,1:-1]
        yy,xx = np.where(distance >= max(self.c['min_interior_px'],float(distance.max())*.8))
        need(len(xx), 'unified_no_interior_grasp')
        index = int(np.argmin((xx-xs.mean())**2+(yy-ys.mean())**2))
        pixel = [int(xx[index]),int(yy[index])]
        xy = self.map_object(obj,[pixel])[0]
        section=contour_section(mask,lambda points:self.map_object(obj,points),pixel,angle)
        # Center between the two actual contour contacts. Keep the historical
        # calibrated-specimen grasp unchanged unless width control is enabled.
        if live or p.get('jaw_calibration') is not None or self.policy.get('jaw_calibration') is not None:
            xy=np.asarray(section['center_card_xy_mm'])
            # The exported pixel must refer to the updated metric grasp as well.
            nearest=int(np.argmin(np.sum((card-xy)**2,axis=1)))
            pixel=[int(xs[nearest]),int(ys[nearest])]
            need(distance[pixel[1],pixel[0]]>=self.c['min_interior_px'],
                 'unified_contour_center_insufficient_interior')
        evidence_rect=self.evidence_rect(mask,pixel) if 'detection_window_px' in self.policy else self.c['source_roi']
        # The nominal open tool footprint is advisory, per the operator's
        # requested removal of this rejection. Keep actual overlap evidence
        # and clearance ranking; this is not a full link collision model.
        margin = self.policy['obstacle_margin_mm']
        separation=cv2.distanceTransform((~mask).astype('uint8'),cv2.DIST_L2,5)
        neighbor_clearance=None
        footprint_clear=True
        source_ids={id(o) for o,_ in self.source_objects(row)}
        for other,other_mask in self.obstacle_objects(row):
            if other is obj:continue
            need(float(separation[other_mask].min())>=self.c['min_separation_px'],'unified_target_too_close_to_other_object')
            oy,ox = np.nonzero(other_mask)
            points = self.obstacle_points(other,np.c_[ox,oy],obj,source_ids)-xy
            across,along = points@vectors[:,0],points@axis
            collision = (np.abs(across) <= self.policy['jaw_open_span_mm']/2+margin) & (
                np.abs(along) <= self.policy['jaw_depth_mm']/2+margin)
            footprint_clear=footprint_clear and not bool(collision.any())
            distances=np.hypot(np.maximum(np.abs(across)-self.policy['jaw_open_span_mm']/2-margin,0),
                               np.maximum(np.abs(along)-self.policy['jaw_depth_mm']/2-margin,0))
            clearance=float(distances.min())
            neighbor_clearance=clearance if neighbor_clearance is None else min(neighbor_clearance,clearance)
        slots = self.free_slots(row,obj['label'],ignore_object=obj)
        need(slots, 'unified_no_free_verified_bin_slot:'+obj['label'])
        measurement=self.item_heights.measurement(obj) if self.item_heights is not None else {}
        jaw_width=measurement.get('grasp_width_mm',section['width_mm'])
        errors = []
        for slot in slots:
            try:
                route = compile_route(self.c,obj['label'],xy.tolist(),angle,self.policy,slot,
                                      grasp_width_mm=jaw_width)
                choice = dict(label=obj['label'],bin_id=p['bin_id'],track_id=track_id,
                    slot_id=slot['slot_id'],slot_roi=slot_rect(p,slot),pick_pixel=pixel,
                    pick_card_xy_mm=xy.tolist(),axis_deg=angle,width_mm=width,length_mm=length,
                    object_profile_id=p['object_profile_id'],object_height_mm=p['object_height_mm'],
                    geometry=self.stats.geometry(mask),base_pwm=route['base_pwm'],wrist_pwm=route['wrist_pwm'],
                    source_chain_index=route['source_chain_index'],model_sha256=route['model_sha256'],
                    complete_route_compiled_before_motion=True,nominal_tool_footprint_clear=footprint_clear,
                    nominal_tool_footprint_check_enforced=False,
                    full_link_collision_check=False,workspace_physically_qualified=False,
                    execution_basis='operator_authorized_unified_model_trial')
                choice['source_evidence_roi']=list(evidence_rect)
                choice['grasp_cross_section']=section
                choice['grasp_width_mm']=section['width_mm']
                choice['jaw_input_width_mm']=jaw_width
                choice['jaw_width_source']='operator_measured' if 'grasp_width_mm' in measurement else 'visual_contour'
                if 'measurement_profile' in measurement:choice['measurement_profile']=copy.deepcopy(measurement['measurement_profile'])
                choice['jaw_closure']=route['jaw_closure']
                choice['jaw_closure']['width_source']=choice['jaw_width_source']
                choice['jaw_closure']['visual_contour_width_mm']=section['width_mm']
                if 'motion_timing' in route:choice['motion_timing']=route['motion_timing']
                choice['mask_geometry_cleanup']=cleanup
                choice['specimen_admission']=dict(mode='live_object' if live else 'calibrated_specimen',
                    observed_area_px=area,observed_width_mm=width,observed_length_mm=length,
                    principal_axis_used=bool(elongated),sample_size_template_required=not bool(live),
                    mechanical_reachability_required=True,physical_validation_completed=False)
                choice['source_bin_overlap_fraction']=overlap
                choice['source_bin_edge_reviews']=edge_reviews
                choice['nominal_neighbor_clearance_mm']=neighbor_clearance
                plan = [(s['name'],s['command'].encode(),s['duration_ms']) for s in route['steps']]
                return choice,plan
            except ValueError as exc:errors.append(str(exc))
        raise ValueError('unified_no_route_to_free_slot:'+','.join(errors))

    def select(self, row):
        pairs = self.source_objects(row)
        if self.item_heights is not None:
            self.item_heights.bind(pairs,row['observation']['source_stream_id'],self.c['min_scene_iou'])
        need(sum(o['label'] in self.allowed_labels for o,_ in pairs)<=self.policy['max_objects'],
             'unified_too_many_selected_material_instances')
        tracks = self.tracker.observe(pairs)
        candidates, rejected = [], []
        for track_id,obj,mask in tracks:
            if obj['label'] not in self.allowed_labels:continue
            try:
                choice,plan = self.candidate(row,(obj,mask),track_id)
                candidates.append((track_id,(obj,mask),choice,plan))
            except ValueError as exc:
                rejected.append(dict(track_id=track_id,label=obj['label'],reason=str(exc)))
        need(not any(o['label'] in self.allowed_labels for _,o,_ in tracks) or candidates,
             'unified_no_pickable_instance:'+str(rejected))
        if self.item_heights is not None or self.policy.get('selection_policy')=='clearance_then_route_v1':
            rank_candidates(candidates,self.movement_wait)
        else:
            candidates.sort(key=lambda v:v[0])
        self.candidate_order=[copy.deepcopy(v[2]) for v in candidates]
        self.ranked_plans=[(copy.deepcopy(v[2]),v[3]) for v in candidates]
        if candidates and self.item_heights is not None:
            choice=candidates[0][2];label=choice['label'];height=choice['object_height_mm']
            self.c['profiles'][label]['object_height_mm']=height
            self.policy['models'][label]['object_height_mm']=height
            self.geometry.runtime['profiles'][label]['object_height_mm']=height
        return (candidates[0] if candidates else None),rejected

    def verify_selected(self, before, target, choice, after):
        if self.placement_verification['mode']=='bin_occupied':
            return self.verify_bin_occupied(before,target[0]['label'],choice,after[-1])
        obj,selected = target
        destination = roi_mask(choice['slot_roi'],self.c['image_size'])
        remaining = [(o,m) for o,m in self.source_objects(before) if o is not obj]
        old_outside = self.outside_objects(before)
        gains,targets,reports = [],[],[]
        for row in after:
            current_source = self.source_objects(row)
            associate(remaining,current_source,self.c['min_scene_iou'])
            outside = self.outside_objects(row)
            matches = associate(old_outside,outside,self.c['min_scene_iou'],allow_extra=True)
            new = [(o,m) for j,(o,m) in enumerate(outside) if j not in matches]
            need(len(new) == 1, 'unified_new_bin_instance_not_unique')
            placed,mask = new[0]
            profile=self.c['profiles'][obj['label']]
            # Live admission also applies to the post-placement observation;
            # the old sample area must not reject an otherwise verified item.
            area=int(mask.sum())
            live=self.policy.get('live_object_admission')
            area_allowed=(area>0 if live and live.get('kind')=='operator-live-geometry-v1' and
                          live.get('operator_statement') and live.get('physical_validation_completed') is False
                          else profile['min_area_px']<=area<=profile['max_area_px'])
            need(placed['label'] == obj['label'] and placed['confidence'] >= self.c['min_confidence'] and
                 area_allowed and
                 int((mask&destination).sum()) >= mask.sum()*.95, 'unified_wrong_class_or_landing_slot')
            remaining_masks=[m for _,m in current_source]
            union=np.logical_or.reduce(remaining_masks) if remaining_masks else np.zeros_like(selected)
            erased = self.stats.visual(selected,union)
            need(erased['erased']/int(selected.sum()) >= self.c['min_erased_fraction'], 'unified_selected_not_removed')
            changed = self.changed_pixels(before,row)
            for _,m in remaining:
                need(float(changed[m].mean()) <= self.c['max_scene_changed_fraction'], 'unified_remaining_object_rgb_changed')
            for _,m in old_outside:
                need(float(changed[m].mean()) <= self.c['max_scene_changed_fraction'], 'unified_old_bin_object_rgb_changed')
            gain_mask = mask&destination&changed
            gain = self.stats.visual(np.zeros_like(mask),gain_mask)
            need(gain['beyond'] >= self.c['min_bin_gain_px'], 'unified_semantic_rgb_gain_missing')
            for p in self.c['profiles'].values():
                if p['bin_id'] != choice['bin_id']:
                    region = roi_mask(p['bin_roi'],self.c['image_size'])
                    need(float(changed[region].mean()) <= self.c['max_other_bin_changed_fraction'], 'unified_wrong_bin_change')
            gains.append(gain_mask);targets.append(mask)
            reports.append(dict(frame_id=row['observation']['source_frame_id'],source=erased,bin=gain,
                                placed_confidence=placed['confidence']))
        need(int(np.logical_and.reduce(gains).sum()) >= self.c['min_bin_gain_px'], 'unified_bin_gain_not_persistent')
        need(np.logical_and.reduce(targets).sum()/np.logical_or.reduce(targets).sum() >= self.c['min_persistent_fraction'],
             'unified_new_instance_unstable')
        board = None
        if self.stats.backend == 'pl_board_local':
            task = self.journal.pending()['task_id']
            original=self.stats.c
            self.stats.c=dict(original,frame_timeout_s=getattr(self,'verification_remaining_s',self.policy.get('verification_timeout_s',5)),
                max_age_ms=self.policy.get('verification_max_age_ms',int(self.policy.get('verification_timeout_s',5)*1000)))
            try:
                board = self.stats.verify_cycle(after,choice['bin_id'],self.source,
                    lambda i,r:self.journal.capture(task,'board-extra-'+str(i),r))
            finally:self.stats.c=original
            self.journal.save(task+'.board-vision.json',board)
        self.final_source = self.source_objects(after[-1])
        return dict(status='visual_evidence_pass',frames=reports,board_rgb=board,
                    remaining_count=len(remaining),independently_validated=False)

    def verify_bin_occupied(self,before,label,choice,row):
        profile=self.c['profiles'][label]
        need(choice['bin_id']==profile['bin_id'],'bin_occupancy_destination_mismatch')
        # The current pre-motion frame records the actual box walls/background;
        # an old warped empty image can turn moved cardboard edges into objects.
        background,exposure=adjust_exposure_reference(before['image'],row['image'],
            dict(self.c['layout'],image_size=self.c['image_size']))
        evidence=bin_occupancy(row['image'],background,profile['bin_roi'],self.placement_verification)
        destination=roi_mask(profile['bin_roi'],self.c['image_size'])
        detections=[]
        for obj,mask in zip(row['observation']['instances'],row['masks']):
            area=int(mask.sum())
            if (area and obj['confidence']>=self.placement_verification['min_detection_confidence'] and
                    int((mask&destination).sum())>=area*.5):
                detections.append(dict(label=obj['label'],confidence=obj['confidence']))
        need(evidence['occupied'] or detections,'unified_target_bin_empty')
        return dict(status='bin_occupied_pass',mode='bin_occupied',
            scope='target_bin_occupancy_only',bin_id=choice['bin_id'],
            frames=[dict(frame_id=row['observation']['source_frame_id'],**evidence)],
            detection_matches=detections,
            basis='any_bin_detection' if detections else 'visible_bin_change_since_before_motion',
            baseline_frame_id=before['observation']['source_frame_id'],
            exposure=exposure,independently_validated=False,physical_success_verified=False,
            class_verified=False,new_item_verified=False,source_removal_verified=False)

    def cycle(self):
        need(self.journal.pending() is None, 'unified_unresolved_previous_motion')
        task = uuid.uuid4().hex
        report = dict(task_id=task,station_id=self.c['station_id'],mode=self.mode,
            operation_scope='supervised-unified-instance-v1',status='observing',steps=[],evidence=[],
            config_sha256=fingerprint(self.c),unified_policy_sha256=fingerprint(self.policy),
            production_quality_claimed=False,workspace_physically_qualified=False,
            unattended_accepted=False,physical_success_verified=False,automatic_retry=False,
            physical_motion_attempted=False)
        attempted,started = False,self.clock()
        after = []
        try:
            prepared=getattr(self,'prepared_batch_item',None)
            if prepared is None:
                rows = self.observe();before = rows[-1]
                selected,rejected = self.select(before)
            else:
                before,selected=prepared['before'],prepared['selected']
                self.c['pickup_source_roi']=copy.deepcopy(prepared['pickup_source_roi'])
                rejected=[]
                choice=selected[2];label=choice['label'];height=choice['object_height_mm']
                self.c['profiles'][label]['object_height_mm']=height
                self.policy['models'][label]['object_height_mm']=height
                self.geometry.runtime['profiles'][label]['object_height_mm']=height
                report['batch_plan_frozen_before_motion']=True
                report['batch_plan_index']=prepared['index']
            report['rejected_candidates'] = rejected
            report['candidate_order'] = self.candidate_order
            report['scene_tracks'] = [dict(track_id=i,label=o['label']) for i,o,_ in self.tracker.live]
            report['material_selection']=sorted(self.allowed_labels)
            report['detection_window_px']=self.policy.get('detection_window_px',self.c['source_roi'])
            if selected is None:
                report['status']='no_matching_material' if self.tracker.live else 'empty_scene'
                self.final_source=self.source_objects(before);return report
            track,target,choice,plan = selected
            report.update(config_sha256=fingerprint(self.c),unified_policy_sha256=fingerprint(self.policy))
            self.batch_context=dict(before=before,target=target,choice=choice,track_id=track)
            report.update(choice=choice,plan=[dict(step=n,command=d.decode(),duration_ms=ms) for n,d,ms in plan],
                          statistics=self.stats.provenance())
            report['evidence'].append(self.journal.capture(task,'before',before))
            if self.mode == 'preview':report['status']='preview_only';return report
            continuation=prepared is not None and prepared['index']>0
            if self.target_region_setup and not continuation:
                report['region_activation']=self.target_region_setup(self,choice)
                report['config_sha256']=fingerprint(self.c)
                report['statistics']=self.stats.provenance()
                self.source.after(self.clock()*1000)
                # Precommit/home require fresh individual captures around USB
                # reset, not queued frames from an earlier three-frame batch.
                if hasattr(self.source,'batch_next'):self.source.batch_next=False
            if not continuation:
                fresh=self.acquire();self.unchanged(before,fresh)
                report['evidence'].append(self.journal.capture(task,'precommit',fresh))
                report['motion_baseline']=self.prepare_motion(fresh,task)
            else:
                report['motion_baseline']=dict(scope='initial_batch_scene',semantic_replanning=False)
            report['status']='motion_intent_reserved'
            self.journal.save('pending.json',report);attempted=True
            if self.authorization_check:self.authorization_check()
            if self.mode=='known_physical':report['physical_motion_attempted']=True
            serial_started=self.clock();self.motion.open(plan)
            report['serial_open_elapsed_s']=self.clock()-serial_started
            for name,data,duration in plan:
                need(self.clock()-started <= self.c['max_cycle_s'], 'unified_cycle_timeout')
                self.motion.check_stop();guard=None
                if name == 'home':
                    if continuation and self.mode=='known_physical':
                        # Check camera/fixture identity without interpreting a
                        # changed scene as another selection or placement test.
                        current=self.frozen_batch_frame(before)
                    elif continuation:
                        current=None
                    else:
                        current=self.acquire();self.unchanged(before,current)
                    if current is not None:
                        report['evidence'].append(self.journal.capture(task,'guard_home',current))
                        stamp=current['observation']['capture_started_ms']
                        def guard():need(0 <= self.clock()*1000-stamp <= self.c['max_age_ms'], 'unified_prewrite_frame_expired')
                    else:
                        guard=self.motion.check_stop
                entry=dict(step=name,status='may_be_in_flight',duration_ms=duration)
                report['steps'].append(entry);self.journal.save('pending.json',report)
                send_started=self.clock();self.motion.send(data,duration,guard=guard)
                entry['guard_and_send_elapsed_s']=self.clock()-send_started
                entry['required_wait_s']=self.movement_wait(name,duration)
                wait_started=self.clock();self.motion.wait(entry['required_wait_s'])
                entry['wait_elapsed_s']=self.clock()-wait_started
                if getattr(self.motion,'last_receipt',None) is not None:entry['transport_receipt']=copy.deepcopy(self.motion.last_receipt)
                entry['status']='sent_and_waited_unverified';self.journal.save('pending.json',report)
            self.motion.close();self.sleep(self.c['retreat_settle_s'])
            barrier=self.clock()*1000
            report['verification_window']=dict(total_timeout_s=self.policy.get('verification_timeout_s',5),
                max_frame_age_ms=self.policy.get('verification_max_age_ms',int(self.policy.get('verification_timeout_s',5)*1000)),
                motion_max_frame_age_ms=self.c['max_age_ms'],stable_frames=self.c['stable_frames'])
            report['placement_verification']=self.placement_verification
            if self.placement_verification['mode']=='bin_occupied':
                report['verification_window']['stable_frames']=1
            deferred=self.policy.get('batch_verification_mode')=='end_of_batch'
            if deferred and self.mode=='known_physical':
                need(all(e.get('transport_receipt',{}).get('echo_matched') is True and
                         e['transport_receipt'].get('completed_movement_wait') is True for e in report['steps']),
                     'unified_deferred_requires_confirmed_commands')
            report['motion_completed']=True
            if deferred and self.policy.get('skip_per_item_visual_verification',False):
                # No per-item placement verdict or motion retry.
                report['verification']=dict(status='deferred_to_batch_end',frames=[],
                    physical_success_verified=False)
            else:
                try:
                    report['verification']=self.verify_within_window(before,target,choice,barrier,after)
                except (ValueError,TimeoutError) as exc:
                    text=str(exc)
                    visual=(text.startswith(('unified_','instance_','station_source_or_bin_',
                            'station_unexpected_region_change:','station_change_not_',
                            'station_post_retreat_window_','station_cycle_evidence_expired',
                            'three_frame_inference_deadline_exceeded')))
                    if not deferred or not visual:raise
                    report['verification']=dict(status='per_item_visual_failed',reason=type(exc).__name__+':'+text)
            if deferred:
                self.tracker.defer(track)
                report['status']='motion_completed_awaiting_batch_verification'
                report['continuation_basis']=('complete_route_echoed_and_waited; frozen_batch_plan'
                    if prepared is not None else 'complete_route_echoed_and_waited; fresh_scene_required_before_next_motion')
            else:
                if self.placement_verification['mode']=='bin_occupied':
                    # Retire this attempt without using detector association as
                    # a second, hidden placement test. Next motion observes anew.
                    self.tracker.defer(track)
                else:self.tracker.retire(track,self.final_source)
                report['status']='simulated_visual_pass' if self.mode=='simulation' else 'completed_visual_evidence'
            return report
        except (Exception,KeyboardInterrupt) as exc:
            report.update(status='needs_review' if attempted else 'rejected',reason=type(exc).__name__+':'+str(exc))
            if attempted:self.journal.save('pending.json',report)
            return report
        finally:
            if attempted:self.motion.close()
            report['elapsed_s']=self.clock()-started
            report['slot_checks']=copy.deepcopy(self.slot_checks)
            report['verification_attempts']=copy.deepcopy(self.verification_attempts)
            for i,row in enumerate(after):report['evidence'].append(self.journal.capture(task,'placed_'+str(i),row))
            self.journal.save(task+'.json',report)
            if attempted and report['status'] in ('completed_visual_evidence','simulated_visual_pass',
                                                  'motion_completed_awaiting_batch_verification'):self.journal.clear()
            if attempted and report['status']=='needs_review':self.journal.save('pending.json',report)
