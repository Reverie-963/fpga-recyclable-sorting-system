import json,math,time
from pathlib import Path
import cv2,numpy as np
from .vision import candidates,mask_cpu

def objects_in_area(mask,vision,cal):
    found=candidates(mask,vision['min_area'],vision['max_area'],already_open=vision.get('pl_kernel','threshold')=='open')
    if cal:
        hull=np.asarray(cal['pixel_hull'],np.float32)
        found=[o for o in found if cv2.pointPolygonTest(hull,tuple(o['center']),False)>=0]
    return found

def same_target(old,old_frame,new,new_frame,tolerance):
    if np.linalg.norm(np.asarray(old['center'])-new['center'])>tolerance:return False
    if not .75<=new['area']/old['area']<=1.33:return False
    def crop(frame,obj):
        x,y,w,h=obj['bbox'];return cv2.resize(frame[y:y+h,x:x+w],(16,16)).astype(float)
    # Additional appearance check, not an identity guarantee.
    return np.abs(crop(old_frame,old)-crop(new_frame,new)).mean()<=25

def claim_task(folder,task_id,record):
    if not task_id.replace('-','').isalnum():raise ValueError('Invalid task id')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    with (folder/(task_id+'.json')).open('x') as f:
        json.dump({'status':'claimed_no_automatic_replay',**record},f)
        f.flush()
        __import__('os').fsync(f.fileno())

def verify_outcome(before,after,target,label,cfg):
    v=cfg['vision'];pre=mask_cpu(before,v['background_bgr'],v['threshold']);post=mask_cpu(after,v['background_bgr'],v['threshold'])
    x,y,w,h=target['bbox'];old=np.count_nonzero(pre[y:y+h,x:x+w]);new=np.count_nonzero(post[y:y+h,x:x+w])
    cleared=old>0 and new<=old*.25
    roi=cfg.get('verification',{}).get('bin_rois',{}).get(label);changed=None
    if roi:
        bx,by,bw,bh=roi
        if not (0<=bx<bx+bw<=after.shape[1] and 0<=by<by+bh<=after.shape[0]):raise ValueError('Invalid bin verification ROI')
        gain=int(np.count_nonzero(post[by:by+bh,bx:bx+bw])-np.count_nonzero(pre[by:by+bh,bx:bx+bw]))
        changed=gain>=cfg['verification'].get('min_bin_gain_px',100)
    return {'source_cleared':bool(cleared),'bin_foreground_increased':changed,
            'status':'visual_evidence_pass' if cleared and changed else 'needs_review',
            'note':'Visual heuristic only; no servo ACK, force sensing, or automatic retry.'}
