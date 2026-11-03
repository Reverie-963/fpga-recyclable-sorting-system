"""Fault injection: truncate DMA input; require bounded failure and recovery."""
import json
import time
from pathlib import Path

import numpy as np

from km1_dma_runtime import DualCoreSession
from km1_mask_overlay import MaskAccel
from km1_visual_overlay import VisualAccel


def main():
    owner=DualCoreSession('vision.bit')
    mask,vis=MaskAccel(session=owner),VisualAccel(session=owner)
    image=np.ones((6,640),np.uint8)
    results=[]
    try:
        for name,core in [('dma_mask',mask),('dma_vis_b',vis)]:
            channel=getattr(owner.overlay,name).sendchannel
            original=channel.transfer
            def truncated(buffer):
                return original(buffer,nbytes=buffer.nbytes-4)
            channel.transfer=truncated
            started=time.monotonic()
            try:
                if core is mask: mask.run(image,timeout_sec=.2)
                else: vis.run(image,image,mode=1,timeout_sec=.2)
            except TimeoutError:
                elapsed=time.monotonic()-started
                if elapsed>5: raise AssertionError('timeout+recovery exceeded 5 seconds')
                results.append({'fault':name+' short input','timeout_plus_recovery_sec':elapsed})
            else: raise AssertionError('truncated transfer unexpectedly completed')
            # Both users must rebind after reset of the shared overlay.
            s,out=mask.run(image,job_id=701)
            assert s['m00']==image.size and s['job_id']==701 and np.all(out==1)
            s,out=vis.run(image,image,mode=1,job_id=702)
            assert s['inter']==image.size and s['union']==image.size and not np.any(out)
        for bad in (np.zeros((0,1),np.uint8),np.zeros((481,640),np.uint8),np.zeros((480,641),np.uint8)):
            try: mask.run(bad)
            except ValueError: pass
            else: raise AssertionError('invalid shape accepted')
        print(json.dumps({'passed':True,'faults':results}),flush=True)
        Path('fault-test.json').write_text(json.dumps({'passed':True,'faults':results},indent=2))
    finally:
        if not owner.poisoned: owner.overlay.download()


if __name__=='__main__': main()
