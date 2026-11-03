"""Persistent buffers for station_vision_accel. No implicit overlay recovery."""
import hashlib
import os
from pathlib import Path
import time
import numpy as np

_quarantined_buffers=[]


class StationVisionAccel:
    backend='pl_board_local'

    def __init__(self,bitfile,assets,size=(640,480),load=False):
        import fcntl
        if os.geteuid()!=0:
            raise RuntimeError('Use the privileged PYNQ runtime with xrt_setup.sh sourced')
        from pynq import Overlay,allocate
        self.allocate=allocate;self.size=tuple(size);self.job=0;self.poisoned=False;self.buffers=[]
        w,h=self.size
        if type(w) is not int or type(h) is not int or not 1<=w<=640 or not 1<=h<=480:
            raise ValueError('invalid_frame_size')
        bitfile=Path(bitfile).resolve()
        actual={key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in
            [('station_vision.bit',bitfile),('station_vision.hwh',bitfile.with_suffix('.hwh'))]}
        if actual!=assets:raise ValueError('station_vision_assets_mismatch')
        self.lease=open('/run/lock/km1-station-vision.lock','a')
        fcntl.flock(self.lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.assets=actual;self.overlay=Overlay(str(bitfile),download=load)
        if not self.overlay.is_loaded():raise RuntimeError('reviewed_station_overlay_not_loaded')
        self.ip=self.overlay.station_vision_0
        n=w*h
        for count,dtype in [(n,np.uint32),(n,np.uint32),(n,np.uint8),(24,np.uint32),(48,np.uint32)]:
            self.buffers.append(allocate((count,),dtype=dtype))
        self.base,self.frame,self.history,self.rois,self.results=self.buffers
        for offset,buf in zip((0x10,0x20,0x30,0x40,0x50),self.buffers):
            self.ip.write(offset,int(buf.physical_address)&0xffffffff)
            self.ip.write(offset+4,int(buf.physical_address)>>32)
        self.ip.write(0x60,w);self.ip.write(0x68,h)
        self.reset_history=True

    def copy_image(self,target,image):
        w,h=self.size
        if image.shape!=(h,w,3) or image.dtype!=np.uint8:raise ValueError('expected_bgr_u8')
        packed=target.view(np.uint8).reshape(h,w,4)
        packed[:,:,:3]=image;packed[:,:,3]=0;target.flush()

    def reset(self,baseline,regions,threshold):
        from src.station_vision import validate_regions,REGION_NAMES
        validate_regions(regions,list(self.size))
        if type(threshold) is not int or not 0<=threshold<=765:raise ValueError('invalid_threshold')
        self.copy_image(self.base,baseline)
        self.rois[:]=[v for name in REGION_NAMES for v in regions[name]];self.rois.flush()
        self.ip.write(0x70,threshold);self.begin_window()

    def begin_window(self):self.reset_history=True

    def process(self,current):
        from src.station_vision import REGION_NAMES
        if self.poisoned or not self.overlay.is_loaded():raise RuntimeError('station_overlay_unavailable')
        self.copy_image(self.frame,current)
        self.job+=1
        if self.job>=2**32:raise RuntimeError('job_id_exhausted')
        self.ip.write(0x78,int(self.reset_history));self.ip.write(0x80,self.job);self.ip.read(0)
        self.poisoned=True  # Any exception after AP_START retains DMA-visible memory.
        self.ip.write(0,1);deadline=time.monotonic()+2
        while not self.ip.read(0)&2:
            if time.monotonic()>deadline:
                self.poisoned=True
                raise TimeoutError('station_vision_timeout_no_automatic_reload')
            time.sleep(.0001)
        if self.ip.read(0x90)!=1 or self.ip.read(0x88)!=self.job:
            self.poisoned=True;raise RuntimeError('station_vision_job_or_status_error')
        self.results.invalidate();self.reset_history=False
        result={name:dict(changed_px=int(row[0]),persistent_px=int(row[1]),delta_sum=int(row[2]),
            bbox_inclusive_px=[int(v) for v in row[3:7]],area_px=int(row[7]))
            for name,row in zip(REGION_NAMES,self.results.reshape(6,8))}
        self.poisoned=False
        return result

    def compare_pair(self, baseline, current, threshold):
        """Isolated PL difference mask; caller MUST hold the capture store lock.

        Uses the existing kernel and separate buffers. The live baseline and
        temporal history are preserved; a failed AXI job remains quarantined.
        """
        if self.poisoned or not self.overlay.is_loaded():
            raise RuntimeError('station_overlay_unavailable')
        w,h=self.size
        for image in (baseline,current):
            if image.dtype!=np.uint8 or image.shape!=(h,w,3):
                raise ValueError('expected_bgr_u8')
        if type(threshold) is not int or not 0<=threshold<=765:
            raise ValueError('invalid_threshold')
        started=time.perf_counter_ns()
        if not hasattr(self,'pair_buffers'):
            self.pair_buffers=[]
            try:
                for count,dtype in [(w*h,np.uint32),(w*h,np.uint32),(w*h,np.uint8),(24,np.uint32),(48,np.uint32)]:
                    self.pair_buffers.append(self.allocate((count,),dtype=dtype))
            except BaseException:
                for buf in self.pair_buffers:buf.freebuffer()
                del self.pair_buffers
                raise
        prepared=time.perf_counter_ns()
        offsets=(0x10,0x14,0x20,0x24,0x30,0x34,0x40,0x44,0x50,0x54,0x70,0x78,0x80)
        registers={offset:self.ip.read(offset) for offset in offsets}
        original=(self.base,self.frame,self.history,self.rois,self.results,self.reset_history)
        self.base,self.frame,self.history,self.rois,self.results=self.pair_buffers
        try:
            for offset,buf in zip((0x10,0x20,0x30,0x40,0x50),self.pair_buffers):
                self.ip.write(offset,int(buf.physical_address)&0xffffffff)
                self.ip.write(offset+4,int(buf.physical_address)>>32)
            self.copy_image(self.base,baseline)
            self.rois[:]=[0,0,w,h]*6;self.rois.flush()
            self.ip.write(0x70,threshold);self.reset_history=True
            self.process(current)
            self.history.invalidate()
            mask=np.asarray(self.history).reshape(h,w).copy()
            finished=time.perf_counter_ns()
            timing=dict(backend='pl_pair_difference_v1',total_ms=(finished-started)/1e6,
                allocation_ms=(prepared-started)/1e6,call_ms=(finished-prepared)/1e6,
                includes_pack_sync_readback=True,kernel_only=False)
        finally:
            self.base,self.frame,self.history,self.rois,self.results,self.reset_history=original
            if not self.poisoned:
                for offset,value in registers.items():self.ip.write(offset,value)
        timing['total_ms']=(time.perf_counter_ns()-started)/1e6
        timing['includes_register_restore']=True
        return mask,timing

    def close(self):
        # A timed-out AXI master may still own memory. Retain buffers until an
        # independently confirmed reset; never reload an overlay from here.
        if self.poisoned:
            _quarantined_buffers.extend(self.buffers+getattr(self,'pair_buffers',[]));self.buffers=[];self.pair_buffers=[];return
        for buf in self.buffers:buf.freebuffer()
        for buf in getattr(self,'pair_buffers',[]):buf.freebuffer()
        self.pair_buffers=[]
        self.buffers=[];self.lease.close()
