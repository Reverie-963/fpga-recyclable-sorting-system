"""Visible-image ROI clipping for the explicitly relaxed 8810 projection policy.

The legacy geometry implementation remains unchanged. Its initializer is kept as
an explicit local implementation, with only the region rectangle builder changed;
inherited world mapping, marker guards and mechanical domains remain identical.
"""
import copy
import cv2
import numpy as np
from contextlib import contextmanager
import threading
from .sorting_relocated_geometry import RelocatedGeometry,bounded_rectangle,rectangle_points
from .station_geometry import need,fingerprint,world_to_pixels,map_xy
from .sorting_loop import mapped
from .prompted_motion_policy import read_motion_options


def visible_rectangle(points,size,inset=False):
    p=np.asarray(points,float)
    need(p.ndim==2 and p.shape[1]==2 and len(p)>0 and np.isfinite(p).all(),'invalid_projected_region_coordinates')
    size=np.asarray(size,int)
    low=np.maximum(0,np.ceil(p.min(0)) if inset else np.floor(p.min(0))).astype(int)
    high=np.minimum(size,(np.floor(p.max(0)) if inset else np.ceil(p.max(0)))+1).astype(int)
    need((high-low>=8).all(),'projected_region_has_no_usable_visible_area')
    if inset:
        poly=p.astype(np.float32)
        for _ in range(int(max(size))):
            if all(cv2.pointPolygonTest(poly,tuple(map(float,q)),False)>=0 for q in rectangle_points([*low,*high])):
                break
            low+=1;high-=1
            need((high-low>=8).all(),'projected_region_has_no_usable_visible_area')
    return [int(low[0]),int(low[1]),int(high[0]),int(high[1])]


class PromptedGeometry(RelocatedGeometry):
    def __init__(self, c, camera, station, registration, scene, pose, reference, motion_shift_px=None,min_fixed_markers=3,review_tolerance_factor=1):
        self.region_projection_records=[]
        from .sorting_review_policy import checked_factor
        factor=checked_factor(review_tolerance_factor)
        need(type(min_fixed_markers) is int and 2<=min_fixed_markers<=4,'invalid_min_fixed_markers')
        self.min_fixed_markers=min_fixed_markers
        if motion_shift_px is None:motion_shift_px=c['layout']['max_layout_shift_px']
        need(type(motion_shift_px) in (int,float) and np.isfinite(motion_shift_px) and
             1<=motion_shift_px<=5*factor,'invalid_station_motion_shift_limit')
        self.motion_shift_px=float(motion_shift_px)
        need(scene['kind']=='known-station-scene-v1' and scene['camera_sha256']==fingerprint(camera) and
             scene['station_sha256']==fingerprint(station) and
             scene['registration_sha256']==fingerprint(registration), 'station_scene_calibration_changed')
        self.original=copy.deepcopy(c);self.camera=camera;self.station=station;self.registration=registration
        self.scene=scene;self.pose=pose;self.reference=reference.copy()
        # Existing robot polygons define the actual tested pick domain. The
        # example paper-board polygon is not a robot commissioning certificate.
        self.robot_domains={k:copy.deepcopy(p['pick_hulls_card_mm']) for k,p in c['profiles'].items()}
        self.runtime=copy.deepcopy(c);rc=self.runtime;size=c['image_size']
        rc['layout']['max_layout_shift_px']=motion_shift_px
        rc['layout']['search_margin_px']=max(rc['layout']['search_margin_px'],int(np.ceil(motion_shift_px)))
        projected={}
        for name, data in scene['scene_regions'].items():
            xyz=np.asarray(data['world_xyz_mm'],float)
            if name=='source':
                highest=max(p['object_height_mm'] for p in c['profiles'].values())
                xyz=np.vstack((xyz,np.c_[xyz[:,:2],np.full(len(xyz),highest)]))
            projected[name]=world_to_pixels(xyz,pose,camera,station)
        rc['source_roi']=self.region_rectangle(projected['source'],size,region='source')
        for p in rc['profiles'].values():p['bin_roi']=self.region_rectangle(projected[p['bin_id']],size,inset=True,region=p['bin_id'])
        patches=[]
        known={m['id']:m['corners_world_mm'] for m in station['markers']}
        action_rects=[rc['source_roi']]+[p['bin_roi'] for p in rc['profiles'].values()]
        for sid in sorted(pose['marker_ids']):
            pix=np.asarray(world_to_pixels(known[sid],pose,camera,station))
            lo=np.floor(pix.min(0)-6).astype(int);hi=np.ceil(pix.max(0)+7).astype(int)
            candidate=[int(lo[0]),int(lo[1]),int(hi[0]),int(hi[1])]
            overlap=any(max(candidate[0],q[0])<min(candidate[2],q[2]) and
                        max(candidate[1],q[1])<min(candidate[3],q[3]) for q in action_rects)
            if (lo>=0).all() and (hi<=size).all() and not overlap:patches.append(candidate)
            if len(patches)==2:break
        need(len(patches)==2,'two_fixed_marker_patches_required')
        rc['layout']['static_rois']=patches
        regions=dict(source=rc['source_roi'],static_a=patches[0],static_b=patches[1])
        regions.update({p['bin_id']:p['bin_roi'] for p in rc['profiles'].values()})
        # Reject overlapping projected regions rather than losing evidence.
        masks=[]
        for rect in regions.values():
            mask=np.zeros((size[1],size[0]),bool);x0,y0,x1,y1=rect;mask[y0:y1,x0:x1]=True
            need(not any((mask&old).any() for old in masks),'projected_station_regions_overlap');masks.append(mask)
        rc['statistics']['regions']=regions
        # Preserve the physically checked size while adapting its pixel area to
        # camera scale. Width/length checks still run in the original mm frame.
        inverse=np.linalg.inv(np.asarray(registration['world_to_card_xy'],float))
        for label,p in rc['profiles'].items():
            center=np.asarray(c['profiles'][label]['pick_hulls_card_mm'][0],float).mean(0)
            world=map_xy(inverse,center).tolist()+[p['object_height_mm']]
            pixel=np.asarray(world_to_pixels([world],pose,camera,station)[0])
            old_h=np.asarray(p['image_to_card_mm'],float)
            old_pixel=map_xy(np.linalg.inv(old_h),center)
            def old_map(points):return np.array([map_xy(old_h,q) for q in points])
            sample=np.array([[0.,0.],[1.,0.],[0.,1.]])
            before=old_map(old_pixel+sample);after=self.map(label,pixel+sample)
            j_old=(before[1:]-before[0]).T;j_new=(after[1:]-after[0]).T
            scale=abs(float(np.linalg.det(j_old)/np.linalg.det(j_new)))
            need(np.isfinite(scale) and .1<=scale<=10,'camera_scale_outside_supported_range')
            p['min_area_px']=max(1,int(np.floor(p['min_area_px']*scale)))
            p['max_area_px']=int(np.ceil(p['max_area_px']*scale))
            p['relocated_pixel_area_scale']=scale
        signature=dict(camera=camera,station=station,registration=registration,scene=scene,
                       pose=pose,robot_domains=self.robot_domains)
        if motion_shift_px!=1:signature['motion_shift_limit_px']=motion_shift_px
        if min_fixed_markers!=3:signature['min_fixed_markers']=min_fixed_markers
        self.signature=fingerprint(signature)
        self.expected={sid:np.asarray(world_to_pixels(known[sid],pose,camera,station)) for sid in known}
        self.check(reference)

    def region_rectangle(self,points,size,inset=False,region='source'):
        result=visible_rectangle(points,size,inset)
        p=np.asarray(points,float)
        self.region_projection_records.append(dict(region=region,projected_points_px=p.tolist(),
            visible_roi=result,clipped=bool((p<0).any() or (p>=size).any()),
            projection_bounds_check_enabled=False))
        return result


_marker_boundary_lock=threading.RLock()


@contextmanager
def external_marker_boundary(detector):
    """Supply verified visual inputs to the unchanged legacy layout guard."""
    from . import station_geometry
    with _marker_boundary_lock:
        original=station_geometry.detect_markers
        station_geometry.detect_markers=detector
        try:yield
        finally:station_geometry.detect_markers=original


def geometry_for_policy(*args,marker_detector=None,**kwargs):
    options=read_motion_options()
    cls=RelocatedGeometry if options.get('projected_region_bounds_check_enabled',True) else PromptedGeometry
    if marker_detector is not None:
        base=cls
        default_marker_detector=marker_detector
        class ExternalMarkerGeometry(base):
            def check(self,image,marker_detector=None):
                with external_marker_boundary(marker_detector or default_marker_detector):
                    return super().check(image)
        cls=ExternalMarkerGeometry
    geometry=cls(*args,**kwargs)
    geometry.signature=fingerprint(dict(base_geometry=geometry.signature,motion_options=options))
    return geometry


def apply_prompted_geometry(geometry,policy):
    poly=np.asarray(policy['source_card_polygon_mm'],float)
    world=mapped(np.linalg.inv(np.asarray(geometry.registration['world_to_card_xy'],float)),poly)
    heights=(0,max(v['object_height_mm'] for v in policy['models'].values()))
    xyz=np.vstack([np.c_[world,np.full(len(world),h)] for h in heights])
    points=world_to_pixels(xyz,geometry.pose,geometry.camera,geometry.station)
    if isinstance(geometry,PromptedGeometry):
        rect=geometry.region_rectangle(points,geometry.runtime['image_size'],region='unified_source')
    else:rect=bounded_rectangle(points,geometry.runtime['image_size'])
    regions=copy.deepcopy(geometry.runtime['statistics']['regions']);regions['source']=rect
    masks=[];w,h=geometry.runtime['image_size']
    for r in regions.values():
        mask=np.zeros((h,w),bool);mask[r[1]:r[3],r[0]:r[2]]=True
        need(not any((mask&other).any() for other in masks),'unified_projected_regions_overlap');masks.append(mask)
    geometry.runtime['source_roi']=rect;geometry.runtime['statistics']['regions']=regions
    geometry.signature=fingerprint(dict(base_geometry=geometry.signature,unified_policy=fingerprint(policy),regions=regions))
    return geometry
