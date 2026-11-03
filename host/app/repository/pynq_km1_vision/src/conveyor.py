"""PL encoder snapshots. Host midpoint is a bounded clock bridge, not camera exposure time."""
import math,time
from collections import deque

class Encoder:
    def __init__(self,overlay,cfg):
        self.io=getattr(overlay,'conveyor_gpio').mmio;self.cfg=cfg
        self.io.write(4,0);self.io.write(12,0xffffffff)
        self.toggle=self.io.read(0)&256
        if self.reg(0)!=0x4b4d3501:raise ValueError('Wrong conveyor overlay/register protocol')
        self.raw=None;self.total=0;self.last_ticks=None;self.error0=None;self.samples=deque(maxlen=4096)
    def reg(self,n):
        self.io.write(0,self.toggle|n)
        return self.io.read(8)
    def sample(self):
        start=time.monotonic();self.toggle^=256;self.io.write(0,self.toggle|1)
        while self.io.read(8)!=(self.toggle>>8):
            if time.monotonic()-start>.05:raise TimeoutError('Encoder snapshot timeout')
        raw=self.reg(2);ticks=self.reg(3)|(self.reg(4)<<32);errors=self.reg(5)
        end=time.monotonic()
        if end-start>self.cfg['snapshot_max_s']:raise TimeoutError('Encoder snapshot jitter')
        if self.last_ticks is not None and ticks<=self.last_ticks:raise RuntimeError('Encoder clock reset/stalled')
        if self.error0 is None:self.error0=errors
        if errors!=self.error0:raise RuntimeError('Illegal quadrature transition')
        if self.raw is not None:self.total+=((raw-self.raw+2**31)%2**32)-2**31
        self.raw=raw;self.last_ticks=ticks
        t=(start+end)/2;pos=self.total*self.cfg['mm_per_count']*self.cfg['direction']
        self.samples.append((ticks/self.cfg['clock_hz'],pos,t))
        velocity=None
        if len(self.samples)>1:
            # Keep a finite velocity estimation window; no fabricated velocity before two samples.
            while len(self.samples)>2 and self.samples[-1][0]-self.samples[1][0]>self.cfg['velocity_window_s']:self.samples.popleft()
            first=self.samples[0];dt=ticks/self.cfg['clock_hz']-first[0]
            if dt>=self.cfg['min_velocity_span_s']:velocity=(pos-first[1])/dt
        return {'time':t,'position_mm':pos,'velocity_mm_s':velocity,'ticks':ticks,
                'uncertainty_s':(end-start)/2,'errors':errors}

def at_time(sample,stamp):
    if sample['velocity_mm_s'] is None:raise ValueError('Encoder velocity not ready')
    return sample['position_mm']+sample['velocity_mm_s']*(stamp-sample['time'])
