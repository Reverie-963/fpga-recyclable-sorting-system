"""Non-destructive workstation annotations for a separate model candidate.

This does not remove obstacle masks, change detections, or authorize movement.
The caller must supply a matching, registered empty reference and workcell ID.
"""
import copy
import hashlib
import cv2
import numpy as np


def _mask(runs,shape):
    if not isinstance(runs,list) or any(type(v) is not int or v<0 for v in runs) or sum(runs)!=int(np.prod(shape)):
        raise ValueError('Invalid full-frame mask RLE')
    return np.repeat(np.arange(len(runs),dtype=np.uint32)%2,runs).reshape(shape).astype(bool)


def annotate_station(frame,reference,instances,*,regions,workcell_id,reference_workcell_id,
                     alignment_verified,reference_sha256,inside_fraction=.8,
                     changed_fraction=.02,pixel_difference=24):
    """Mark source candidates and bin observations; keep every original mask.

    Pixel/area thresholds are explicit engineering settings, not calibrated
    guarantees for transparent items or a moving camera. Scores are unchanged.
    """
    if frame.dtype!=np.uint8 or reference.dtype!=np.uint8 or frame.shape!=reference.shape or frame.ndim!=3 or frame.shape[2]!=3:
        raise ValueError('Reference and frame must have identical uint8 BGR geometry')
    if hashlib.sha256(reference.tobytes()).hexdigest()!=reference_sha256:
        raise ValueError('Empty reference content changed')
    if not workcell_id or workcell_id!=reference_workcell_id or alignment_verified is not True:
        raise ValueError('Workcell identity/alignment has not been verified')
    if not 0<inside_fraction<=1 or not 0<=changed_fraction<=1 or not 0<=pixel_difference<=255:
        raise ValueError('Invalid filter settings')
    h,w=frame.shape[:2]
    if 'source' not in regions:raise ValueError('Source region is required')
    roi_masks={}
    for name,roi in regions.items():
        area=np.zeros((h,w),np.uint8)
        if len(roi)==4 and all(type(x) is int for x in roi):
            x0,y0,x1,y1=roi
            if not 0<=x0<x1<=w or not 0<=y0<y1<=h:raise ValueError('Region outside image')
            area[y0:y1,x0:x1]=1
        else:
            if len(roi)<3 or any(not isinstance(p,list) or len(p)!=2 or any(type(x) is not int for x in p) for p in roi):
                raise ValueError('Invalid region polygon')
            points=np.asarray(roi,np.int32)
            if np.any(points<0) or np.any(points[:,0]>=w) or np.any(points[:,1]>=h):
                raise ValueError('Region polygon outside image')
            if cv2.contourArea(points)<=0:raise ValueError('Empty region polygon')
            cv2.fillPoly(area,[points],1)
        roi_masks[name]=area.astype(bool)
    changed=np.max(np.abs(frame.astype(np.int16)-reference.astype(np.int16)),axis=2)>pixel_difference
    output=[]
    for instance in instances:
        row=copy.deepcopy(instance);mask=_mask(row['mask_rle'],(h,w));area=int(mask.sum())
        coverage={name:float((mask&roi).sum()/area) if area else 0. for name,roi in roi_masks.items()}
        fraction=float((mask&changed).sum()/area) if area else 0.
        zone='outside'
        # Bin observations remain available for post-drop verification.
        for name in roi_masks:
            if name!='source' and coverage[name]>=inside_fraction:zone=name;break
        if zone=='outside' and coverage['source']>=inside_fraction:zone='source'
        reasons=[]
        if not area:reasons.append('empty_mask')
        if zone!='source':reasons.append('outside_source_grasp_region')
        if fraction<changed_fraction:reasons.append('matches_registered_empty_background')
        row.update(station_zone=zone,station_coverage=coverage,changed_pixel_fraction=fraction,
                   station_grasp_candidate=not reasons,station_filter_reasons=reasons,
                   obstacle_mask_preserved=True,automatic_motion_allowed=False)
        output.append(row)
    return dict(instances=output,raw_instances_preserved=True,workcell_id=workcell_id,
                reference_bgr_sha256=reference_sha256,automatic_motion_allowed=False)
