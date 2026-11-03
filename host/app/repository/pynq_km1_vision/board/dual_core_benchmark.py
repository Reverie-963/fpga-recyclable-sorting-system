"""Same-PS CPU versus complete PL API latency; no isolated-kernel claim."""
import json
from pathlib import Path
import time

import cv2
import numpy as np

from dual_core_acceptance import geometry, evidence, check
from km1_dma_runtime import DualCoreSession
from km1_mask_overlay import MaskAccel
from km1_visual_overlay import VisualAccel


def measure(fn, count=100):
    wall=[]; cpu=[]
    for _ in range(3): fn()
    for _ in range(count):
        c=time.process_time(); t=time.perf_counter()
        fn()
        wall.append((time.perf_counter()-t)*1000)
        cpu.append((time.process_time()-c)*1000)
    return {'count':count,'wall_median_ms':float(np.median(wall)),
            'wall_p95_ms':float(np.percentile(wall,95)),
            'process_cpu_median_ms':float(np.median(cpu)),
            'process_cpu_to_wall_percent':100*sum(cpu)/sum(wall),
            'serial_calls_per_second':1000*count/sum(wall),
            'wall_ms':wall,'process_cpu_ms':cpu}


def main():
    cv2.setNumThreads(1)
    owner=DualCoreSession('vision.bit')
    mask,vis=MaskAccel(session=owner),VisualAccel(session=owner)
    result={'scope':'Same Zynq PS, OpenCV single-thread CPU reference vs complete PL API including validation/allocation/copies/DMA/stats/free. No isolated kernel timing.',
            'cases':[]}
    try:
        for h,w in ((120,160),(480,640)):
            a=np.zeros((h,w),np.uint8); a[h//4:3*h//4,w//4:3*w//4]=255
            b=np.roll(a,5,axis=1)
            for name,cpu_fn,pl_fn,job in (
                ('mask-open',lambda:geometry(a,1,1),lambda:mask.run(a,erode=1,dilate=1,job_id=81),81),
                ('visual-mask',lambda:evidence(a,b,1,0),lambda:vis.run(a,b,mode=1,threshold=0,job_id=82),82)):
                expected,image=cpu_fn(); stats,got=pl_fn(); check(stats,got,expected,image,job)
                cpu=measure(cpu_fn); pl=measure(pl_fn)
                record={'core':name,'h':h,'w':w,'cpu':cpu,'pl':pl,
                        'cpu_over_pl_median':cpu['wall_median_ms']/pl['wall_median_ms']}
                result['cases'].append(record)
                print(name,h,w,'CPU ms',cpu['wall_median_ms'],'PL ms',pl['wall_median_ms'],flush=True)
    finally:
        Path('benchmark.json').write_text(json.dumps(result,indent=2))
        if not owner.poisoned: owner.overlay.download()


if __name__=='__main__': main()
