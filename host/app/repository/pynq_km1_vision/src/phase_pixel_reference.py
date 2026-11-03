"""Observation-only CPU oracle/packing for the uncommissioned pixel candidate.

No overlay, transport, actuator or production-selection imports. LUT fitting,
frame/asset identity, freshness and unchanged geometry/8-connected checks remain
the caller's responsibility; these reductions never authorize a grasp/verdict.
"""
import numpy as np
from .sorting_io import image_stats
from .rgb_affine import affine_lut


def checked_image(image):
    if (not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.ndim!=3 or
            image.shape[2]!=3 or not 1<=image.shape[0]<=480 or not 1<=image.shape[1]<=640):
        raise ValueError('phase_pixel_requires_bounded_bgr_u8')
    return image


def checked_masks(masks,shape):
    if not isinstance(masks,(list,tuple)) or len(masks)>8:raise ValueError('phase_pixel_max_eight_masks')
    if any(not isinstance(m,np.ndarray) or m.dtype!=np.bool_ or m.shape!=shape for m in masks):
        raise ValueError('phase_pixel_requires_same_frame_bool_masks')
    return masks


def packed_frame(image,masks=()):
    image=checked_image(image);checked_masks(masks,image.shape[:2])
    words=image[:,:,0].astype(np.uint32)
    words|=image[:,:,1].astype(np.uint32)<<8
    words|=image[:,:,2].astype(np.uint32)<<16
    for i,mask in enumerate(masks):words|=mask.astype(np.uint32)<<(24+i)
    return words


def packed_lut(gain,bias):
    # 3 consecutive B/G/R tables. Same float64 clip/truncate as deployed LUT.
    return np.ascontiguousarray(affine_lut(gain,bias)[:,0,:].T)


def reference_evidence(before,current,masks,lut,rois,threshold,job_id):
    before,current=checked_image(before),checked_image(current)
    if before.shape!=current.shape:raise ValueError('phase_pixel_frame_shape_mismatch')
    checked_masks(masks,before.shape[:2])
    if not isinstance(lut,np.ndarray) or lut.dtype!=np.uint8 or lut.shape!=(3,256):raise ValueError('phase_pixel_lut_layout')
    if type(threshold) is not int or not 0<=threshold<=765:raise ValueError('phase_pixel_threshold')
    if type(job_id) is not int or not 1<=job_id<2**32:raise ValueError('phase_pixel_job_id')
    h,w=before.shape[:2]
    if (not isinstance(rois,list) or len(rois)!=6 or any(not isinstance(r,list) or len(r)!=4 or
        not all(type(x) is int for x in r) or not 0<=r[0]<r[2]<=w or not 0<=r[1]<r[3]<=h for r in rois)):
        raise ValueError('phase_pixel_six_half_open_rois')
    adjusted=np.empty_like(before)
    for c in range(3):adjusted[:,:,c]=lut[c,before[:,:,c]]
    delta=np.abs(current.astype(np.int16)-adjusted.astype(np.int16)).sum(axis=2)
    changed=(delta>threshold).astype(np.uint8)
    moments=np.zeros((8,10),dtype=np.uint64)
    keys=('m00','m10','m01','m20','m02','m11','min_col','min_row','max_col','max_row')
    for i,mask in enumerate(masks):
        stats=image_stats(mask,job_id)
        moments[i]=[stats[k] for k in keys]
    regions=np.asarray([[int(changed[y0:y1,x0:x1].sum()),int(delta[y0:y1,x0:x1].sum())]
                        for x0,y0,x1,y1 in rois],dtype=np.uint32)
    return dict(changed=changed,moments=moments,regions=regions,
                completed_job=job_id,status=1,observation_only=True,physical_success_verified=False)
