"""Calibrated pixel rays and scene regions for the existing known-item routes."""
import copy
from itertools import combinations

import cv2
import numpy as np

from .station_geometry import (camera_parameters, fingerprint, homography, map_xy, need,
    pose_parameters, registration_ray_point, world_to_pixels)
from .station_relocalization import legacy_domains
from .sorting_scene import check_motion_layout,MotionLayoutCache


def pixels_to_card(points, height, pose, camera, station, registration):
    k, d, r, t = pose_parameters(pose, camera, station)
    points = np.asarray(points, float).reshape(-1, 2)
    need(len(points) > 0 and np.isfinite(points).all() and
         (points >= 0).all() and (points < camera['image_size']).all(), 'invalid_grasp_pixels')
    need(station['height_range_mm'][0] <= height <= station['height_range_mm'][1], 'invalid_grasp_height')
    rays = np.c_[cv2.undistortPoints(points.reshape(-1, 1, 2), k, d).reshape(-1, 2), np.ones(len(points))]
    directions = (r.T@rays.T).T; center = -r.T@t
    need(center[2] > height and (directions[:, 2] < -.1).all(), 'invalid_grasp_rays')
    xyz = center+((height-center[2])/directions[:, 2])[:, None]*directions
    h = homography(registration['world_to_card_xy'])
    return (h@np.c_[xyz[:, :2], np.ones(len(xyz))].T).T[:, :2]


def rectangle_points(rect):
    x0,y0,x1,y1=rect
    return [[x0,y0],[x1-1,y0],[x1-1,y1-1],[x0,y1-1]]


def scene_from_frame(c, row, pose, camera, station, registration, bin_heights):
    need(set(bin_heights) == {'bin_plastic','bin_metal','bin_paper'} and
         all(type(h) in (int,float) and 0 <= h <= station['height_range_mm'][1] for h in bin_heights.values()),
         'three_explicit_bin_floor_heights_required')
    regions = {'source':dict(rect=c['source_roi'],height=0)}
    regions.update({p['bin_id']:dict(rect=p['bin_roi'],height=bin_heights[p['bin_id']]) for p in c['profiles'].values()})
    world = {}
    for name, data in regions.items():
        world[name] = dict(height_mm=data['height'], world_xyz_mm=[registration_ray_point(pixel, data['height'],
            pose, camera, station)['world_xyz_mm'] for pixel in rectangle_points(data['rect'])])
    return dict(kind='known-station-scene-v1', world_frame_id=station['world_frame_id'],
        camera_sha256=fingerprint(camera), station_sha256=fingerprint(station),
        registration_sha256=fingerprint(registration), scene_regions=world,
        reference_pose=pose, reference_image_sha256=row.get('bgr_sha256'),
        bin_floor_height_source='direct_operator_input', scene_geometry_confirmed=False,
        empty_bins_confirmed=False, automatic_motion_allowed=False)


def bounded_rectangle(points, size, inset=False):
    p = np.asarray(points, float)
    need(np.isfinite(p).all() and (p >= 0).all() and (p < size).all(), 'station_region_outside_image')
    low=np.ceil(p.min(0)).astype(int); high=np.floor(p.max(0)).astype(int)+1
    if inset:
        # Shrink an axis-aligned box until all corners lie inside the projected
        # bin floor. Never count a neighbouring patch as part of the destination.
        poly=p.astype(np.float32)
        for _ in range(max(size)):
            if all(cv2.pointPolygonTest(poly, tuple(map(float,q)),False)>=0 for q in rectangle_points([*low,*high])):break
            low += 1; high -= 1
            need((high-low >= 8).all(), 'projected_bin_floor_too_narrow')
    else:
        low=np.floor(p.min(0)).astype(int);high=np.ceil(p.max(0)).astype(int)+1
    need((low >= 0).all() and (high <= size).all() and (high-low >= 8).all(), 'station_region_too_small_or_clipped')
    return [int(low[0]),int(low[1]),int(high[0]),int(high[1])]


class RelocatedGeometry:
    def __init__(self, c, camera, station, registration, scene, pose, reference, motion_shift_px=None,min_fixed_markers=3,review_tolerance_factor=1):
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
        rc['source_roi']=bounded_rectangle(projected['source'],size)
        for p in rc['profiles'].values():p['bin_roi']=bounded_rectangle(projected[p['bin_id']],size,inset=True)
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

    def map(self,label,points):
        return pixels_to_card(points,self.original['profiles'][label]['object_height_mm'],self.pose,
                              self.camera,self.station,self.registration)

    def check(self,image,marker_detector=None):
        from .station_geometry import detect_markers
        self.last_patch_attempts=[]
        detections=(marker_detector or detect_markers)(image,self.camera)
        known=[d for d in detections if d['id'] in self.expected]
        need(len({d['id'] for d in known})==len(known),'motion_duplicate_marker_ids')
        minimum=getattr(self,'min_fixed_markers',3)
        need(len(known)>=minimum,'motion_requires_two_fixed_markers' if minimum==2 else 'motion_requires_three_fixed_markers')
        shift=max(float(np.linalg.norm(np.asarray(d['corners_px'])-self.expected[d['id']],axis=1).max()) for d in known)
        # Compare against measured corner means, not perfect reprojections:
        # systematic subpixel lens residual must not consume a motion margin.
        records=self.pose['evidence']
        measured={sid:np.mean([next(d['corners_px'] for d in rec['detections'] if d['id']==sid)
            for rec in records if any(d['id']==sid for d in rec['detections'])],axis=0) for sid in self.pose['marker_ids']}
        common=[d for d in known if d['id'] in measured]
        need(len(common)>=minimum,'motion_marker_support_changed')
        observed=max(float(np.linalg.norm(np.asarray(d['corners_px'])-measured[d['id']],axis=1).max()) for d in common)
        need(observed<=self.motion_shift_px,'motion_camera_or_fixed_board_moved')
        # The arm can legitimately hide a fixed left-hand template during a
        # transfer. Select two currently visible, independently registered
        # marks from the SAME frozen reference; never fit a new pose or weaken
        # the displacement check above to accommodate the moving arm.
        width,height=self.runtime['image_size'];patches={}
        for row in common:
            sid=row['id'];corners=measured[sid]
            low=np.floor(corners.min(0)-6).astype(int);high=np.ceil(corners.max(0)+7).astype(int)
            if (low>=0).all() and (high<=[width,height]).all():
                patches[sid]=[*map(int,low),*map(int,high)]
        pairs=list(combinations(sorted(patches),2))
        pairs.sort(key=lambda pair:-float(np.linalg.norm(measured[pair[0]].mean(0)-measured[pair[1]].mean(0))))
        static=None;selected=None;qualified=[]
        with MotionLayoutCache(self.reference,image) as frame_layout_cache:
            for pair in pairs:
                try:
                    checked=check_motion_layout(self.reference,image,dict(self.runtime['layout'],
                        image_size=self.runtime['image_size'],static_rois=[patches[sid] for sid in pair]),_frame_cache=frame_layout_cache)
                except ValueError as exc:
                    self.last_patch_attempts.append(dict(marker_ids=list(pair),passed=False,reason=str(exc)))
                    continue
                self.last_patch_attempts.append(dict(marker_ids=list(pair),passed=True))
                correlations=[m['correlation'] for m in checked.get('markers',[])]
                quality=min(correlations) if correlations else 0
                span=float(np.linalg.norm(measured[pair[0]].mean(0)-measured[pair[1]].mean(0)))
                qualified.append((quality,span,checked,list(pair)))
        if qualified:
            _,_,static,selected=max(qualified,key=lambda item:(item[0],item[1]))
        need(static is not None,'motion_static_marker_moved_occluded_or_ambiguous')
        return dict(calibration_sha256=self.signature,marker_ids=[d['id'] for d in common],
                    minimum_fixed_markers=minimum,
                    observed_shift_px=observed,motion_shift_limit_px=self.motion_shift_px,
                    reprojection_diagnostic_px=shift,static=static,
                    photometric_marker_ids=selected,photometric_selection='visible_registered_pair_frozen_reference',
                    patch_attempts=self.last_patch_attempts,camera_pose_recomputed=False)

    def warp_empty_reference(self,image):
        """Warp only an explicitly confirmed empty reference, never detector input."""
        need(self.scene.get('empty_bins_confirmed') is True,'empty_bins_reference_not_confirmed')
        width,height=self.runtime['image_size'];yy,xx=np.indices((height,width))
        # Per destination pixel, intersect the declared floor plane and project
        # into the old reference; evaluate each height separately for each bin.
        k,d,r,t=pose_parameters(self.pose,self.camera,self.station)
        pix=np.c_[xx.ravel(),yy.ravel()].astype(float)
        rays=np.c_[cv2.undistortPoints(pix.reshape(-1,1,2),k,d).reshape(-1,2),np.ones(len(pix))]
        dirs=(r.T@rays.T).T;center=-r.T@t
        result=np.zeros_like(image)
        heights={0.}
        heights.update(float(v['height_mm']) for v in self.scene['scene_regions'].values())
        for z in heights:
            xyz=center+((z-center[2])/dirs[:,2])[:,None]*dirs
            uv=np.asarray(world_to_pixels(xyz,self.scene['reference_pose'],self.camera,self.station),np.float32).reshape(height,width,2)
            remapped=cv2.remap(image,uv[:,:,0],uv[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            valid=(uv[:,:,0]>=0)&(uv[:,:,0]<width-1)&(uv[:,:,1]>=0)&(uv[:,:,1]<height-1)
            if z==0:result=remapped
            for name,data in self.scene['scene_regions'].items():
                if data['height_mm']!=z or name=='source':continue
                label=next(k for k,p in self.runtime['profiles'].items() if p['bin_id']==name)
                x0,y0,x1,y1=self.runtime['profiles'][label]['bin_roi']
                need(valid[y0:y1,x0:x1].all(),'empty_reference_new_view_not_covered')
                result[y0:y1,x0:x1]=remapped[y0:y1,x0:x1]
        return result
