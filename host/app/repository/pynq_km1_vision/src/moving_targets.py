"""Encoder-compensated, conservative one-to-one association and finite-window intercepts."""
import math
import numpy as np
from .vision import map_point

class Tracker:
    def __init__(self,cfg):self.cfg=cfg;self.tracks={};self.next_id=1
    def update(self,objects,stamp,belt_position,cal):
        axis=np.asarray(self.cfg['axis_xy'],float);new={};gates={}
        for i,obj in enumerate(objects):
            xy=np.asarray(map_point(obj['center'],cal));matches=[]
            for tid,t in self.tracks.items():
                predicted=np.asarray(t['xy'])+axis*(belt_position-t['belt'])
                if stamp-t['stamp']<=self.cfg['track_max_gap_s'] and np.linalg.norm(xy-predicted)<=self.cfg['association_mm'] and .6<obj['area']/t['object']['area']<1.67:matches.append(tid)
            gates[i]=(xy,matches)
        # Ambiguous merges/crossings invalidate IDs instead of carrying old class labels.
        for i,obj in enumerate(objects):
            xy,matches=gates[i]
            unique=len(matches)==1 and sum(matches[0] in v[1] for v in gates.values())==1
            tid=matches[0] if unique else self.next_id
            if not unique:self.next_id+=1
            old=self.tracks.get(tid,{})
            new[tid]={'id':tid,'xy':xy.tolist(),'belt':belt_position,'stamp':stamp,'object':obj,
                      'seen':old.get('seen',0)+1,'label':old.get('label'),'semantic_stamp':old.get('semantic_stamp',0),
                      'confidence':old.get('confidence',0)}
        self.tracks=new;return new
    def bind(self,snapshot,detections,stamp):
        # snapshot is geometry/IDs from the exact frame submitted to YOLO.
        from .grasp_geometry import overlap
        for tid,t in snapshot.items():
            if tid not in self.tracks:continue
            x,y,w,h=t['object']['bbox'];box=[x,y,x+w,y+h]
            ds=[d for d in detections if overlap(box,d['xyxy'])/(w*h)>=self.cfg['semantic_coverage']]
            if len(ds)!=1:continue
            d=ds[0]
            owners=sum(overlap([o['object']['bbox'][0],o['object']['bbox'][1],o['object']['bbox'][0]+o['object']['bbox'][2],o['object']['bbox'][1]+o['object']['bbox'][3]],d['xyxy'])/(o['object']['bbox'][2]*o['object']['bbox'][3])>=self.cfg['semantic_coverage'] for o in snapshot.values())
            if owners!=1:continue
            current=self.tracks[tid]
            if current['label'] is not None and current['label']!=d['label']:
                del self.tracks[tid];continue
            current.update(label=d['label'],semantic_stamp=stamp,confidence=d['confidence'])

def inside(xy,window,margin=0):
    x,y=xy;a,b,c,d=window
    return a+margin<=x<=b-margin and c+margin<=y<=d-margin

def intercept(track,sample,now,horizon,cfg):
    d=cfg['dynamic'];v=sample['velocity_mm_s']
    if v is None or not d['min_speed_mm_s']<=v<=d['max_speed_mm_s']:return None
    if track['seen']<d['min_track_frames'] or now-track['stamp']>d['max_track_age_s']:return None
    if now-track['semantic_stamp']>d['semantic_max_age_s'] or track['label'] not in cfg['arm']['bins'] or track['confidence']<cfg['grasp']['confidence']:return None
    age=now-track['stamp'];travel=sample['position_mm']-track['belt']+v*(now+horizon-sample['time'])
    point=(np.asarray(track['xy'])+np.asarray(d['axis_xy'])*travel).tolist()
    bound=d['position_uncertainty_mm']+abs(v)*(d['capture_uncertainty_s']+sample['uncertainty_s']+d['actuation_uncertainty_s'])+.5*d['max_accel_mm_s2']*(age+horizon)**2
    if bound>d['max_error_mm'] or not inside(point,d['pick_window_mm'],bound):return None
    return {'track_id':track['id'],'label':track['label'],'xy':point,'contact_time':now+horizon,
            'belt_at_contact':sample['position_mm']+v*(now+horizon-sample['time']),
            'error_bound_mm':bound,'velocity_mm_s':v}

def verify_deposit(before,after,label,cfg):
    # A belt carries an ungrasped item away too: source disappearance is NOT success.
    x,y,w,h=cfg['verification']['bin_rois'][label]
    gain=int(np.count_nonzero(after[y:y+h,x:x+w])-np.count_nonzero(before[y:y+h,x:x+w]))
    return {'status':'bin_visual_evidence_pass' if gain>=cfg['verification']['min_bin_gain_px'] else 'needs_review',
            'bin_gain_px':gain,'source_disappearance_used':False,'physical_success_requires_ground_truth':True}
