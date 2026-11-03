"""Fuse YOLO semantics with PL binary geometry, and reject ambiguous grasps."""
import cv2
import numpy as np
from .vision import candidates, map_point
from .control import same_target


def overlap(a, b):
    return max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))


def fuse(detections, mask, cfg, cal):
    """Each accepted detection must own exactly one isolated foreground component.
    Grasp candidates are interior material points, never blind box centres.
    This is a planar heuristic, not a general antipodal/collision solver.
    """
    g = cfg['grasp']; v = cfg['vision']; objects = candidates(mask, v['min_area'], v['max_area'], already_open=True)
    accepted = []
    for d in detections:
        if d['confidence'] < g['confidence'] or d['label'] not in cfg['arm']['bins']: continue
        box = d['xyxy']; matches = []
        for obj in objects:
            x,y,w,h = obj['bbox']; ob = [x,y,x+w,y+h]
            if overlap(box, ob) / (w*h) >= g['min_box_coverage']: matches.append(obj)
        if len(matches) != 1: continue
        obj = matches[0]; x,y,w,h = obj['bbox']; ob = [x,y,x+w,y+h]
        # Any second box on this component makes its ownership ambiguous.
        if any(other is not d and overlap(other['xyxy'], ob) > 0 for other in detections): continue
        pad = g['clearance_px']; expanded = [x-pad,y-pad,x+w+pad,y+h+pad]
        if any(other is not obj and overlap(expanded, [other['bbox'][0],other['bbox'][1],other['bbox'][0]+other['bbox'][2],other['bbox'][1]+other['bbox'][3]]) > 0 for other in objects): continue
        crop = mask[y:y+h,x:x+w]
        distance = cv2.distanceTransform(np.pad(crop, 1), cv2.DIST_L2, 5)[1:-1,1:-1]
        maximum = float(distance.max())
        if maximum < g['min_interior_px']: continue
        # Choose the deepest interior pixel nearest centroid (handles ring holes).
        ys,xs = np.where(distance >= maximum-1e-5)
        j = np.argmin((xs+x-obj['center'][0])**2+(ys+y-obj['center'][1])**2)
        point = [float(xs[j]+x), float(ys[j]+y)]
        if cal:
            hull = np.asarray(cal['pixel_hull'], np.float32)
            # Keep entire candidate component in calibrated picking region.
            if any(cv2.pointPolygonTest(hull, (float(px),float(py)), False) < 0 for px,py in [(x,y),(x+w,y),(x,y+h),(x+w,y+h)]): continue
            xy = map_point(point, cal); xmin,xmax,ymin,ymax = cfg['arm']['pick_bounds_mm']
            if not (xmin <= xy[0] <= xmax and ymin <= xy[1] <= ymax): continue
        accepted.append({**obj, 'center': point, 'label':d['label'],
                         'confidence':d['confidence'], 'xyxy':box, 'interior_px':maximum})
    return sorted(accepted, key=lambda o:(-o['confidence'], -o['interior_px'], o['center'][1], o['center'][0]))


def match_target(old, old_frame, choices, frame, tolerance):
    found = [o for o in choices if o['label']==old['label'] and same_target(old,old_frame,o,frame,tolerance)]
    return found[0] if len(found)==1 else None


def verify_masks(before_mask, after_mask, target, label, cfg):
    x,y,w,h = target['bbox']; pre = np.count_nonzero(before_mask[y:y+h,x:x+w]); post = np.count_nonzero(after_mask[y:y+h,x:x+w])
    cleared = pre > 0 and post <= pre * .25
    bx,by,bw,bh = cfg['verification']['bin_rois'][label]
    gain = int(np.count_nonzero(after_mask[by:by+bh,bx:bx+bw])-np.count_nonzero(before_mask[by:by+bh,bx:bx+bw]))
    increased = gain >= cfg['verification']['min_bin_gain_px']
    return {'source_cleared':bool(cleared), 'bin_foreground_increased':bool(increased),
            'bin_gain_px':gain, 'status':'visual_evidence_pass' if cleared and increased else 'needs_review'}
