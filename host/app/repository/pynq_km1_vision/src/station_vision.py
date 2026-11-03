"""CPU oracle and bounded state for board-local RGB/ROI evidence processing.

No camera, overlay, serial or acceptance side effects on import.
"""
import hashlib
import json
import math
import time
import numpy as np

REGION_NAMES=('source','bin_plastic','bin_metal','bin_paper','static_a','static_b')


def validate_regions(regions,size):
    w,h=size
    if (type(w) is not int or type(h) is not int or not 1<=w<=640 or not 1<=h<=480 or
        set(regions)!=set(REGION_NAMES)):
        raise ValueError('invalid_station_vision_regions')
    for rect in regions.values():
        if (not isinstance(rect,list) or len(rect)!=4 or any(type(v) is not int for v in rect) or
            not 0<=rect[0]<rect[2]<=w or not 0<=rect[1]<rect[3]<=h):
            raise ValueError('invalid_station_vision_rectangle')


def pixels(image):
    if (not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3 or
        not 1<=image.shape[1]<=640 or not 1<=image.shape[0]<=480):
        raise ValueError('expected_bounded_bgr_u8')
    return image


def oracle(baseline,current,regions,threshold,history=None):
    baseline,current=pixels(baseline),pixels(current)
    if baseline.shape!=current.shape or type(threshold) is not int or not 0<=threshold<=765:
        raise ValueError('invalid_rgb_comparison')
    validate_regions(regions,list(current.shape[1::-1]))
    difference=np.abs(current.astype(np.int16)-baseline.astype(np.int16)).sum(axis=2,dtype=np.uint16)
    changed=difference>threshold
    if history is not None and (history.shape!=changed.shape or history.dtype!=np.bool_):
        raise ValueError('invalid_history')
    persistent=changed.copy() if history is None else changed & history
    result={}
    for name in REGION_NAMES:
        x0,y0,x1,y1=regions[name];mask=changed[y0:y1,x0:x1];yy,xx=np.nonzero(mask)
        result[name]=dict(changed_px=int(mask.sum()),persistent_px=int(persistent[y0:y1,x0:x1].sum()),
            delta_sum=int(difference[y0:y1,x0:x1].sum()),area_px=(x1-x0)*(y1-y0),
            bbox_inclusive_px=[int(xx.min())+x0,int(yy.min())+y0,int(xx.max())+x0,int(yy.max())+y0] if len(xx) else [0]*4)
    return result,persistent


class CPUVision:
    backend='cpu_reference'
    def reset(self,baseline,regions,threshold):
        self.baseline=pixels(baseline).copy();self.regions=regions;self.threshold=threshold;self.history=None
    def process(self,current):
        result,self.history=oracle(self.baseline,current,self.regions,self.threshold,self.history)
        return result
    def begin_window(self):self.history=None
    def close(self):pass


class StationVision:
    """Baseline changes are explicit and stale/repeated frames never advance history."""
    def __init__(self,backend,regions,size=(640,480),threshold=45,max_gap_s=.6,window_frames=3):
        validate_regions(regions,list(size))
        if (type(threshold) is not int or not 0<=threshold<=765 or not 0<max_gap_s<=2 or
            type(window_frames) is not int or not 2<=window_frames<=10):
            raise ValueError('invalid_station_vision_policy')
        self.backend=backend;self.regions=json.loads(json.dumps(regions));self.size=list(size)
        self.threshold=threshold;self.max_gap_s=max_gap_s;self.window_frames=window_frames
        self.config_sha256=hashlib.sha256(json.dumps(dict(size=self.size,regions=self.regions,
            threshold=threshold,max_gap_s=max_gap_s,window_frames=window_frames),sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self.baseline_id=None;self.baseline_sha=None;self.last_id=None;self.last_capture=None;self.samples=0

    def bind_baseline(self,frame,frame_id,capture_time):
        if (pixels(frame).shape[1::-1]!=tuple(self.size) or type(frame_id) is not int or frame_id<0 or
            type(capture_time) not in (int,float) or not math.isfinite(capture_time)):
            raise ValueError('invalid_baseline_identity')
        self.backend.reset(frame,self.regions,self.threshold)
        self.baseline_id=frame_id;self.baseline_sha=hashlib.sha256(frame.tobytes()).hexdigest()
        self.last_id=frame_id;self.last_capture=capture_time;self.samples=0

    def process(self,frame,frame_id,capture_time):
        if self.baseline_id is None:raise ValueError('baseline_not_bound')
        if (type(frame_id) is not int or frame_id<=self.last_id or type(capture_time) not in (int,float) or
            not math.isfinite(capture_time) or capture_time<=self.last_capture):
            raise ValueError('stale_or_repeated_frame')
        if capture_time-self.last_capture>self.max_gap_s:
            self.baseline_id=None;raise ValueError('capture_gap_baseline_invalidated')
        if pixels(frame).shape[1::-1]!=tuple(self.size):raise ValueError('frame_size_changed')
        if self.samples%self.window_frames==0:self.backend.begin_window()
        started=time.perf_counter_ns();result=self.backend.process(frame);elapsed=(time.perf_counter_ns()-started)/1e6
        self.last_id=frame_id;self.last_capture=capture_time;self.samples+=1
        return dict(schema_version='station-vision-v1',backend=self.backend.backend,frame_id=frame_id,
            baseline_frame_id=self.baseline_id,baseline_bgr_sha256=self.baseline_sha,
            frame_bgr_sha256=hashlib.sha256(frame.tobytes()).hexdigest(),config_sha256=self.config_sha256,
            consecutive_frames=(self.samples-1)%self.window_frames+1,total_frames=self.samples,
            window_complete=self.samples%self.window_frames==0,processing_ms=elapsed,regions=result,
            semantic_verified=False,automatic_motion_allowed=False,physical_success_verified=False)
