"""32-bit AXIS DMA contract; timeout faults require overlay reload, never silent CPU fallback."""
import time
import numpy as np

class Segmenter:
    def __init__(self,bitfile,shape,kernel="threshold"):
        from pynq import Overlay,allocate
        self.overlay=Overlay(bitfile)
        self.ip=self.overlay.segment_0;self.dma=self.overlay.axi_dma_0
        self.kernel=kernel
        expected='segment_open' if kernel=='open' else 'segment'
        if self.overlay.ip_dict['segment_0']['type'].split(':')[-2]!=expected:raise ValueError('Overlay kernel does not match configuration')
        self.shape=shape;self.n=shape[0]*shape[1]
        if kernel=='open' and not all(0<d<=1920 for d in shape):raise ValueError('Morphology dimensions exceed 1920')
        if not 0<self.n*4<2**23: raise ValueError('Frame exceeds DMA contract')
        self.tx=allocate((self.n,),dtype=np.uint32);self.rx=allocate((self.n,),dtype=np.uint32)
        self.failed=False
    def run(self,bgr,bg,threshold,timeout=2):
        if self.failed: raise RuntimeError('DMA fault: restart process and reload overlay')
        if tuple(bgr.shape[:2])!=tuple(self.shape): raise ValueError('Frame shape changed')
        # Packed 0x00RRGGBB: OpenCV input is BGR.
        v=bgr.astype(np.uint32)
        self.tx[:]=(v[:,:,0]|(v[:,:,1]<<8)|(v[:,:,2]<<16)).reshape(-1)
        self.tx.flush()
        packed=int(bg[0])|(int(bg[1])<<8)|(int(bg[2])<<16)
        if self.kernel=='open':
            self.ip.write(0x10,self.shape[1]);self.ip.write(0x18,self.shape[0]);self.ip.write(0x20,packed);self.ip.write(0x28,int(threshold))
        else:
            self.ip.write(0x10,self.n);self.ip.write(0x18,packed);self.ip.write(0x20,int(threshold))
        deadline=time.monotonic()+timeout
        try:
            self.dma.recvchannel.transfer(self.rx)
            self.ip.write(0,1)
            self.dma.sendchannel.transfer(self.tx)
            while True:
                ss=self.dma.read(0x04);rs=self.dma.read(0x34)
                if (ss|rs)&0x770: raise RuntimeError('DMA status error: %x %x'%(ss,rs))
                if ss&2 and rs&2: break
                if time.monotonic()>deadline: raise TimeoutError('DMA timeout (TLAST/clock/length)')
                time.sleep(.0001)
            self.dma.sendchannel.wait();self.dma.recvchannel.wait()
            self.rx.invalidate()
            return self.rx.reshape(self.shape).astype(np.uint8)
        except Exception:
            self.failed=True
            # Stop issuing transfers; buffers remain allocated while DMA may be active.
            raise
    def close(self):
        if not self.failed:
            self.tx.freebuffer();self.rx.freebuffer()
