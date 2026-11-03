"""Fixed-camera geometry guard for arm motion under automatic exposure.

Motion changes scene brightness and occludes the source. This guard checks
calibrated static marker positions and visibility, not grip or task success.
RGB differences remain diagnostics; final placement uses separate evidence.
"""
import math
import hashlib
import json
import copy
import cv2
import numpy as np

from .static_visual_verification import layout_evidence


def verify_setup_reanchor(reference, current, policy):
    """Read-only setup check for a lighting-only template refresh.

    Never called during motion. Independent corner tracks must establish
    subpixel agreement before an operator-confirmed empty-box setup can save a
    new template. No homography, pixel warp or robot calibration is changed.
    """
    check_motion_layout(reference, reference, policy)
    check_motion_layout(current, current, policy)
    reports = []
    for x0, y0, x1, y1 in policy['static_rois']:
        a = reference[y0:y1, x0:x1]
        b = current[y0:y1, x0:x1]
        ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        margin = policy['search_margin_px']
        sx, sy = max(0, x0-margin), max(0, y0-margin)
        search = cv2.cvtColor(current[sy:y1+margin, sx:x1+margin], cv2.COLOR_BGR2GRAY)
        _, score, _, loc = cv2.minMaxLoc(cv2.matchTemplate(search, ga, cv2.TM_CCOEFF_NORMED))
        shift = math.hypot(sx+loc[0]-x0, sy+loc[1]-y0)
        if score < .9 or shift > 1:
            raise ValueError('setup_anchor_moved_or_occluded')
        gains, biases = [], []
        for channel in range(3):
            aa, bb = a[:, :, channel].astype(float), b[:, :, channel].astype(float)
            variance = aa.var()
            if variance < 64:
                raise ValueError('setup_anchor_lacks_contrast')
            gain = float(np.mean((aa-aa.mean())*(bb-bb.mean()))/variance)
            bias = float(bb.mean()-gain*aa.mean())
            # Setup may have spatially different illumination after removing
            # the large white card. Offset is diagnostic, not compensation;
            # visibility and independent corner geometry remain mandatory.
            if not .6 <= gain <= 1.4:
                raise ValueError('setup_anchor_contrast_changed')
            gains.append(gain); biases.append(bias)
        # Normalize intensity only. Coordinates remain exactly the original
        # patch coordinates in both the forward and backward optical flow.
        aa = np.clip((ga.astype(float)-ga.mean())/ga.std()*40+128, 0, 255).astype(np.uint8)
        bb = np.clip((gb.astype(float)-gb.mean())/gb.std()*40+128, 0, 255).astype(np.uint8)
        points = cv2.goodFeaturesToTrack(aa, maxCorners=100, qualityLevel=.01, minDistance=3)
        if points is None or len(points) < 8:
            raise ValueError('setup_anchor_too_few_corners')
        forward, status, _ = cv2.calcOpticalFlowPyrLK(aa, bb, points, None, winSize=(15, 15), maxLevel=1)
        backward, back_status, _ = cv2.calcOpticalFlowPyrLK(bb, aa, forward, None, winSize=(15, 15), maxLevel=1)
        good = ((status[:, 0] == 1) & (back_status[:, 0] == 1) &
                (np.linalg.norm(backward[:, 0]-points[:, 0], axis=1) < .5))
        if good.sum() < 8 or good.mean() < .85:
            raise ValueError('setup_anchor_tracks_inconsistent')
        delta = forward[:, 0][good]-points[:, 0][good]
        displacement = np.linalg.norm(delta, axis=1)
        support = float(cv2.contourArea(cv2.convexHull(points[good])))/ga.size
        median = float(np.linalg.norm(np.median(delta, axis=0)))
        if (support < .2 or median > .5 or float(np.percentile(displacement, 95)) > .75 or
                float(displacement.max()) > 1):
            raise ValueError('setup_anchor_subpixel_agreement_failed')
        reports.append(dict(raw_correlation=score, integer_shift_px=shift, corners=len(points),
            tracked_corners=int(good.sum()), support=support, median_shift_px=np.median(delta, axis=0).tolist(),
            p95_displacement_px=float(np.percentile(displacement, 95)),
            max_displacement_px=float(displacement.max()), gain=gains, bias=biases))
    return dict(kind='operator_setup_lighting_reanchor_only', markers=reports,
                geometric_warp_applied=False, robot_mapping_changed=False, motion_policy_relaxed=False)


def adjust_exposure_reference(reference, current, policy):
    """Fit bounded global color change from fixed marks; never move pixels.

    This is the stationary before/after comparison, separate from motion guards.
    Both the original and adjusted marks must preserve position and visibility.
    """
    from .sorting_review_policy import checked_factor
    factor=checked_factor(policy.get('review_relaxation_factor',1))
    geometry = check_motion_layout(reference, current, policy)
    a = np.concatenate([reference[y0:y1, x0:x1].reshape(-1, 3)
                        for x0, y0, x1, y1 in policy['static_rois']]).astype(float)
    b = np.concatenate([current[y0:y1, x0:x1].reshape(-1, 3)
                        for x0, y0, x1, y1 in policy['static_rois']]).astype(float)
    gain, bias = [], []
    for channel in range(3):
        variance = np.var(a[:, channel])
        if variance < 64/factor:
            raise ValueError('insufficient_exposure_reference')
        g = float(np.mean((a[:, channel]-a[:, channel].mean()) *
                          (b[:, channel]-b[:, channel].mean())) / variance)
        t = float(b[:, channel].mean()-g*a[:, channel].mean())
        if not 1-.3*factor <= g <= 1+.3*factor or not -32*factor <= t <= 32*factor:
            raise ValueError('exposure_out_of_range')
        gain.append(g)
        bias.append(t)
    from .rgb_affine import apply_rgb_affine_u8
    adjusted = apply_rgb_affine_u8(reference,gain,bias)
    checks = layout_evidence(adjusted, current, policy)
    return adjusted, dict(method='bounded_global_rgb_affine_no_geometric_warp', gain=gain, bias=bias,
                          motion_geometry=geometry, adjusted_layout=checks,
                          physical_alignment_verified=False)


class MotionLayoutCache:
    """One serialized guard call; verify pixels at both ends of the scope."""
    def __init__(self,reference,current):
        self.reference,self.current=reference,current
        self.data={};self.active=False;self.hashes=None
    def __enter__(self):
        if self.active:raise ValueError('motion_guard_cache_already_active')
        self.data.clear()
        self.hashes=(hashlib.sha256(self.reference.tobytes()).hexdigest(),
                     hashlib.sha256(self.current.tobytes()).hexdigest())
        self.active=True
        return self
    def __exit__(self,kind,value,traceback):
        try:
            if kind is None:
                current=(hashlib.sha256(self.reference.tobytes()).hexdigest(),
                         hashlib.sha256(self.current.tobytes()).hexdigest())
                if current!=self.hashes:raise ValueError('motion_guard_pixels_changed_during_check')
        finally:self.active=False
        return False


def check_motion_layout(reference, current, policy, *, _frame_cache=None):
    from .sorting_review_policy import checked_factor
    factor=checked_factor(policy.get('review_relaxation_factor',1))
    def need(condition, reason):
        if not condition:
            raise ValueError(reason)
    need(isinstance(reference,np.ndarray) and isinstance(current,np.ndarray) and
         reference.shape==current.shape and reference.ndim==3 and reference.shape[2]==3 and
         reference.dtype==np.uint8 and current.dtype==np.uint8,'invalid_motion_guard_image')
    w,h=policy['image_size']
    need(reference.shape==(h,w,3),'motion_guard_size_mismatch')
    patch_cache={}
    if _frame_cache is not None:
        if isinstance(_frame_cache,MotionLayoutCache):
            need(_frame_cache.active and _frame_cache.reference is reference and _frame_cache.current is current,
                 'motion_guard_cache_frame_mismatch')
            hashes=_frame_cache.hashes
            _frame_cache=_frame_cache.data
        else:
            hashes=(hashlib.sha256(reference.tobytes()).hexdigest(),hashlib.sha256(current.tobytes()).hexdigest())
        signature=(*hashes,
                   json.dumps({k:v for k,v in policy.items() if k!='static_rois'},sort_keys=True,allow_nan=False))
        if _frame_cache.get('signature')!=signature:
            _frame_cache.clear()
            _frame_cache.update(signature=signature,before=cv2.cvtColor(reference,cv2.COLOR_BGR2GRAY),
                                after=cv2.cvtColor(current,cv2.COLOR_BGR2GRAY),patches={})
        before,after=_frame_cache['before'],_frame_cache['after']
        patch_cache=_frame_cache['patches']
    else:
        before=cv2.cvtColor(reference,cv2.COLOR_BGR2GRAY)
        after=cv2.cvtColor(current,cv2.COLOR_BGR2GRAY)
    reports=[]
    margin=policy['search_margin_px']
    need(type(margin) is int and 1<=margin<=16,'invalid_motion_search_margin')
    for roi in policy['static_rois']:
        if tuple(roi) in patch_cache:
            reports.append(copy.deepcopy(patch_cache[tuple(roi)]))
            continue
        x0,y0,x1,y1=roi
        need(0<=x0<x1<=w and 0<=y0<y1<=h,'invalid_motion_static_roi')
        template=before[y0:y1,x0:x1]
        patch=after[y0:y1,x0:x1]
        need(float(template.std())>=policy['min_patch_std'] and float(patch.std())>=policy['min_patch_std'],
             'motion_marker_has_no_texture')
        need(float(np.percentile(patch,95)-np.percentile(patch,5))>=policy['min_motion_patch_dynamic_range'],
             'motion_marker_exposure_not_usable')
        corners=cv2.goodFeaturesToTrack(template,maxCorners=100,qualityLevel=.01,minDistance=3)
        need(corners is not None and len(corners)>=policy['min_patch_corners'],'motion_marker_lacks_corners')
        support=float(cv2.contourArea(cv2.convexHull(corners)))/template.size
        need(support>=.1/factor,'motion_marker_support_too_local')
        sx,sy=max(0,x0-margin),max(0,y0-margin)
        search=after[sy:min(h,y1+margin),sx:min(w,x1+margin)]
        scores=cv2.matchTemplate(search,template,cv2.TM_CCOEFF_NORMED)
        _,score,_,location=cv2.minMaxLoc(scores)
        alternatives=scores.copy()
        yy,xx=np.indices(scores.shape)
        radius=max(1,policy['max_layout_shift_px'])
        alternatives[(xx-location[0])**2+(yy-location[1])**2<radius**2]=-1
        peak=score-float(alternatives.max())
        shift=math.hypot(sx+location[0]-x0,sy+location[1]-y0)
        need(score>=policy['min_patch_correlation'] and shift<=policy['max_layout_shift_px'] and
             peak>=policy['min_patch_peak_margin'],'motion_static_marker_moved_occluded_or_ambiguous')
        reports.append(dict(correlation=score,shift_px=shift,peak_margin=peak,corners=len(corners),
            patch_dynamic_range=float(np.percentile(patch,95)-np.percentile(patch,5)),
            rgb_mean_difference_diagnostic=float(np.abs(current[y0:y1,x0:x1].astype(np.int16)-
                                                       reference[y0:y1,x0:x1].astype(np.int16)).mean())))
        patch_cache[tuple(roi)]=copy.deepcopy(reports[-1])
    need(len(reports)>=2,'need_two_motion_markers')
    return dict(kind='fixed_static_marker_geometry_during_motion',markers=reports,
                geometric_warp_applied=False,grasp_or_success_inferred=False)
