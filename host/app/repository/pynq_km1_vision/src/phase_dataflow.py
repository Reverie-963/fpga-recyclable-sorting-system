"""Reuse a fully verified initial inference once, within original frame limits.

No device, acceptance, timing-threshold or actuator changes. The caller owns
model/generation verification and supplies a fresh fallback camera.
"""
import copy
import json
import hashlib
import time

class PreparedFrameCamera:
    enabled=True
    def __init__(self, source, row, verify_identity, clock=time.monotonic):
        self.source,self.clock,self.verify_identity=source,clock,verify_identity
        # Camera metadata originates as JSON; clone it with the bounded C
        # encoder/decoder instead of recursively copying every RLE integer.
        self.seed=json.loads(json.dumps({k:v for k,v in row.items() if k not in ('image','source_png_bytes')},allow_nan=False))
        if row.get('source_png_bytes') is not None:self.seed['source_png_bytes']=row['source_png_bytes']
        self.seed['image']=row['image'].copy()
        self.seed_hash=hashlib.sha256(self.seed['image'].tobytes()).hexdigest()
        self.used=False
    def __getattr__(self,name):return getattr(self.source,name)
    def __setattr__(self,name,value):
        if name in {'source','clock','verify_identity','seed','seed_hash','used'}:
            object.__setattr__(self,name,value)
        else:setattr(self.source,name,value)
    def after(self,stamp):
        self.seed=None
        return self.source.after(stamp)
    def next(self,timeout):
        row,self.seed=self.seed,None
        if not self.enabled:row=None
        if row is None:return self.source.next(timeout)
        c=self.source.c;obs=row['observation'];stamp=obs['capture_started_ms']
        age=self.clock()*1000-stamp
        identity=('camera_id','model_version','runtime_id','image_size','prompt_mode')
        if not (0<=age<=c['max_age_ms'] and all(obs.get(k)==c[k] for k in identity)):
            return self.source.next(timeout)
        if row.get('bgr_sha256')!=self.seed_hash or hashlib.sha256(row['image'].tobytes()).hexdigest()!=self.seed_hash:
            raise ValueError('prepared_frame_pixels_changed')
        start=self.clock();self.verify_identity()
        if self.clock()-start>=timeout or not 0<=self.clock()*1000-stamp<=c['max_age_ms']:
            return self.source.next(max(.05,timeout-(self.clock()-start)))
        # Continue the camera's original monotonically increasing counters.
        self.source.sequence=obs['frame_id']
        self.source.previous=int(obs['source_frame_id'])
        self.source.previous_stamp=stamp
        self.source.stream=obs['source_stream_id']
        observer=getattr(self.source,'observe_prepared',None)
        if callable(observer):observer(row)
        self.used=True
        row['dataflow']=dict(initial_inference_reused=True,original_capture_started_ms=stamp,
                             original_source_frame_id=obs['source_frame_id'],frame_age_ms=age)
        return row
