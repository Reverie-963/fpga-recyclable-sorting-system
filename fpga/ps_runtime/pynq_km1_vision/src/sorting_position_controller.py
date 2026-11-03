"""Explicit supervised position trials in the existing fixed-marker controller.

The old robot qualification remains unchanged. This opt-in policy authorizes
bounded model trials and reports their evidence; it never marks a workspace as
measured or enables unattended operation.
"""
import copy
import math
import cv2
import numpy as np
from .sorting_loop import need, mapped
from .sorting_position_plan import compile_position_route
from .sorting_relocated import RelocatedController
from .sorting_relocated_geometry import bounded_rectangle
from .station_geometry import fingerprint, world_to_pixels
from .sorting_scene import adjust_exposure_reference
from .static_visual_verification import roi_mask
from .sorting_relocated_camera import WindowedStationCamera


def policy_polygon(value):
    p=np.asarray(value,np.float32)
    need(p.ndim==2 and p.shape[1]==2 and len(p)>=3 and np.isfinite(p).all() and
         cv2.isContourConvex(p) and cv2.contourArea(p)>1,'invalid_position_policy_polygon')
    return p


def validate_position_policy(policy,c):
    need(policy.get('kind')=='operator-supervised-position-policy-v1' and
         policy.get('operator_authorized') is True and bool(policy.get('operator_statement')),
         'position_trials_require_explicit_operator_policy')
    need(policy.get('label')=='paper' and policy.get('object_height_mm')==65 and
         c['profiles']['paper']['object_height_mm']==65,'position_policy_requires_known_65mm_carton')
    need(policy.get('workspace_physically_qualified') is False and policy.get('unattended_accepted') is False,
         'position_trial_must_not_claim_workspace_qualification')
    tolerance=policy.get('axis_tolerance_deg')
    need(type(tolerance) in (int,float) and math.isfinite(tolerance) and 0<tolerance<=30,
         'position_trial_axis_limit_invalid')
    centers=policy_polygon(policy['pick_center_card_polygon_mm'])
    source=policy_polygon(policy['source_card_polygon_mm'])
    need(all(cv2.pointPolygonTest(source,tuple(map(float,p)),False)>=0 for p in centers),
         'position_centers_outside_source_surface')
    compile_position_route(c,'paper',centers.mean(0).tolist(),90,policy['model'])
    return policy


def apply_position_geometry(geometry,policy):
    """Project the explicitly authorized trial surface using existing registration."""
    validate_position_policy(policy,geometry.original)
    polygon=np.asarray(policy['source_card_polygon_mm'],float)
    world=mapped(np.linalg.inv(np.asarray(geometry.registration['world_to_card_xy'],float)),polygon)
    xyz=np.vstack([np.c_[world,np.full(len(world),h)] for h in (0,policy['object_height_mm'])])
    pixels=world_to_pixels(xyz,geometry.pose,geometry.camera,geometry.station)
    rect=bounded_rectangle(pixels,geometry.runtime['image_size'])
    regions=copy.deepcopy(geometry.runtime['statistics']['regions']);regions['source']=rect
    masks=[];w,h=geometry.runtime['image_size']
    for r in regions.values():
        mask=np.zeros((h,w),bool);x0,y0,x1,y1=r;mask[y0:y1,x0:x1]=True
        need(not any((mask&other).any() for other in masks),'position_trial_regions_overlap')
        masks.append(mask)
    geometry.runtime['source_roi']=rect;geometry.runtime['statistics']['regions']=regions
    geometry.signature=fingerprint(dict(base_geometry_sha256=geometry.signature,
        position_policy_sha256=fingerprint(policy),projected_regions=regions))
    return geometry


class PositionWindowedCamera(WindowedStationCamera):
    def after(self,stamp):
        super().after(stamp)
        # Preserve the retreat barrier and frame ages. PL window completeness
        # is checked independently by verify_cycle, rather than forcing every
        # third inference request to wait for one selected cache phase.
        self.source.post_retreat_count=None


class PositionController(RelocatedController):
    def __init__(self,*args,position_policy,**kwargs):
        self.position_policy=copy.deepcopy(position_policy)
        validate_position_policy(self.position_policy,args[0])
        self.position_route_active=False
        super().__init__(*args,**kwargs)

    def plan(self,row,target):
        obj,mask=target;policy=self.position_policy
        if obj['label']!=policy['label']:
            choice,plan=super().plan(row,target)
            choice['route_mode']='legacy_checked_points'
            return choice,plan
        # Only the position model uses its tested low-speed schedule. Other
        # materials retain their original recipe and selected timing profile.
        self.position_route_active=True;self.motion_profile='qualified'
        p=self.c['profiles'][obj['label']];area=int(mask.sum())
        need(p['min_area_px']<=area<=p['max_area_px'],'position_object_area_outside_profile')
        need(cv2.connectedComponents(mask.astype('uint8'),connectivity=8)[0]==2,'fragmented_position_instance')
        ys,xs=np.nonzero(mask);card=self.geometry.map(obj['label'],np.c_[xs,ys])
        values,vectors=np.linalg.eigh(np.cov(card.T))
        need(values[1]>=values[0]*1.2,'position_axis_ambiguous')
        angle=float(np.degrees(np.arctan2(vectors[1,1],vectors[0,1]))%180)
        need(abs((angle-90+90)%180-90)<=policy['axis_tolerance_deg'],'position_orientation_outside_trial')
        width=float(np.ptp(card@vectors[:,0]));length=float(np.ptp(card@vectors[:,1]))
        need(p['width_bounds_mm'][0]<=width<=p['width_bounds_mm'][1] and
             p['length_bounds_mm'][0]<=length<=p['length_bounds_mm'][1],'position_specimen_dimensions')
        distance=cv2.distanceTransform(np.pad(mask.astype('uint8'),1),cv2.DIST_L2,5)[1:-1,1:-1]
        yy,xx=np.where(distance>=max(self.c['min_interior_px'],float(distance.max())*.8))
        need(len(xx)>0,'no_position_interior_grasp')
        index=int(np.argmin((xx-xs.mean())**2+(yy-ys.mean())**2));point=[int(xx[index]),int(yy[index])]
        xy=self.geometry.map(obj['label'],[point])[0]
        need(cv2.pointPolygonTest(policy_polygon(policy['pick_center_card_polygon_mm']),tuple(map(float,xy)),False)>=0,
             'position_pick_outside_operator_trial_surface')
        region=roi_mask(p['bin_roi'],self.c['image_size'])
        need(not any((m&region).any() for m in row['masks']),'known_destination_not_empty_semantically')
        need(self.empty_reference is not None,'known_empty_reference_required')
        empty,_=adjust_exposure_reference(self.empty_reference,row['image'],dict(self.c['layout'],image_size=self.c['image_size']))
        delta=np.abs(row['image'].astype('int16')-empty.astype('int16')).sum(2)
        need(float((delta[region]>self.c['rgb_difference_threshold']).mean())<=self.c['max_scene_changed_fraction'],
             'known_destination_rgb_not_empty')
        route=compile_position_route(self.c,obj['label'],xy.tolist(),angle,policy['model'])
        # All packets are compiled before the original state machine reserves
        # intent or opens serial. "home" retains its post-USB target recheck.
        plan=[('home' if i==0 else s['name'],s['command'].encode(),s['duration_ms']) for i,s in enumerate(route['steps'])]
        choice=dict(label=obj['label'],bin_id=p['bin_id'],pick_pixel=point,pick_card_xy_mm=xy.tolist(),
            width_mm=width,length_mm=length,axis_deg=angle,geometry=self.stats.geometry(mask),
            base_pwm=route['base_pwm'],wrist_pwm=route['wrist_pwm'],control_xy_nominal=route['control_xy_nominal'],
            route_mode='supervised_position_model',
            position_policy_sha256=fingerprint(policy),route_model_sha256=route['model_sha256'],
            complete_route_compiled_before_motion=True,nominal_interpolation=route['nominal_interpolation'],
            inside_legacy_checked_point_domain=route['inside_legacy_checked_point_domain'],
            execution_basis='explicit_operator_supervised_position_model_trial',
            workspace_physically_qualified=False,old_robot_qualification_rewritten=False,
            station_calibration=dict(calibration_sha256=self.geometry.signature,
                source_frame_id=row['observation']['source_frame_id'],source_stream_id=row['observation']['source_stream_id'],
                bgr_sha256=row['bgr_sha256'],height_mm=p['object_height_mm'],old_image_homography_used=False))
        return choice,plan

    def movement_wait(self,name,duration):
        if self.position_route_active:return duration/1000+(1 if name=='release' else .5)
        return super().movement_wait(name,duration)

    def cycle(self):
        report=super().cycle()
        report.update(position_mode='operator_supervised_position_model',workspace_physically_qualified=False,
            position_policy_sha256=fingerprint(self.position_policy),legacy_robot_qualification_preserved=True)
        self.journal.save(report['task_id']+'.json',report)
        return report
