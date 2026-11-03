"""Deterministic board acceptance; no camera, serial or mechanical operations.

Run beside paired vision.bit/vision.hwh and the three drivers. Synthetic data
are explicitly labelled; this is NOT sorting accuracy or camera calibration.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import cv2
import numpy as np
import pynq
from pynq import Clocks

from km1_dma_runtime import DualCoreSession
from km1_mask_overlay import MaskAccel
from km1_visual_overlay import VisualAccel


def bbox(image):
    points = cv2.findNonZero(image)
    if points is None:
        return dict(min_row=0,max_row=0,min_col=0,max_col=0)
    x,y,w,h = cv2.boundingRect(points)
    return dict(min_row=y,max_row=y+h-1,min_col=x,max_col=x+w-1)


def geometry(image, erode, dilate):
    out = (image != 0).astype(np.uint8)
    kernel = np.ones((3,3),np.uint8)
    if erode: out = cv2.erode(out,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)
    if dilate: out = cv2.dilate(out,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)
    m = cv2.moments(out,binaryImage=True)
    stats = {key:int(m[key]) for key in ('m00','m10','m01','m20','m02','m11')}
    stats.update(bbox(out))
    return stats,out


def evidence(a,b,mode,threshold):
    if mode == 0:
        out = (cv2.absdiff(a,b)>threshold).astype(np.uint8)
        stats = {'diff_count':cv2.countNonZero(out)}
    else:
        aa,bb=(a!=0).astype(np.uint8),(b!=0).astype(np.uint8)
        out=cv2.bitwise_xor(aa,bb)
        stats={'inter':cv2.countNonZero(cv2.bitwise_and(aa,bb)),
               'erased':cv2.countNonZero(cv2.bitwise_and(aa,cv2.bitwise_not(bb))),
               'beyond':cv2.countNonZero(cv2.bitwise_and(bb,cv2.bitwise_not(aa))),
               'union':cv2.countNonZero(cv2.bitwise_or(aa,bb))}
    stats.update(bbox(out))
    return stats,out


def check(stats,out,expected,image,job):
    for key,value in expected.items():
        if stats[key]!=value: raise AssertionError('%s: got %s expected %s' % (key,stats[key],value))
    if stats['job_id']!=job or stats['status']!=1: raise AssertionError('job/status mismatch')
    if not np.array_equal(out.reshape(image.shape),image): raise AssertionError('pixel mismatch')


def pattern(rng,h,w,index):
    if index%8==0: return np.zeros((h,w),np.uint8)
    if index%8==1: return np.full((h,w),255,np.uint8)
    if index%8==2:
        image=np.zeros((h,w),np.uint8); image[-1,-1]=255; return image
    if index%8==3:
        image=np.zeros((h,w),np.uint8); image[h//4:max(h//4+1,3*h//4),w//4:max(w//4+1,3*w//4)]=255; return image
    return (rng.random_sample((h,w)) < (.9 if index%8==4 else .4)).astype(np.uint8)*255


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--bit',default='vision.bit')
    parser.add_argument('--count',type=int,default=1280,help='cases PER core; 256 edge regressions then random masks')
    parser.add_argument('--output',default='acceptance.json')
    args=parser.parse_args()
    if args.count < 1: parser.error('count must be positive')
    cv2.setNumThreads(1)
    session=DualCoreSession(args.bit)
    mask,visual=MaskAccel(session=session),VisualAccel(session=session)
    rng=np.random.RandomState(20261027)
    sizes=[(1,1),(1,63),(1,64),(1,65),(1,80),(2,5),(5,2),(480,1),(1,640),
           (6,640),(480,640),(479,639),(31,97),(120,160),(240,320),(48,64)]
    records=[]
    mask_hashes=set()
    report={'data':'deterministic synthetic arrays, seed=20261027','cases_per_core':args.count,
            'runtime':{'pynq':pynq.__version__,'numpy':np.__version__,'opencv':cv2.__version__,
                       'platform':platform.platform(),'fclk0_mhz':Clocks.fclk0_mhz,'cpu_mhz':Clocks.cpu_mhz},
            'assets':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(args.bit),Path(args.bit).with_suffix('.hwh'))},
            'dma_bits':{n:session.overlay.ip_dict[n]['parameters']['c_sg_length_width'] for n in ('dma_mask','dma_vis_a','dma_vis_b')},
            'records':records,'passed':False}
    started=time.monotonic()
    try:
        for i in range(args.count):
            h,w=sizes[i%len(sizes)]
            # Cycle flags independently of size and pattern (avoid correlated coverage).
            flags=((i//len(sizes)) if i<256 else (i//32))%4
            if i < 256:
                a=pattern(rng,h,w,i//len(sizes))
            else:
                h,w=(480,640) if i%32==0 else (rng.randint(16,481),rng.randint(16,641))
                a=(rng.random_sample((h,w)) < rng.uniform(.05,.95)).astype(np.uint8)*255
            input_hash=hashlib.sha256(a.tobytes()).hexdigest()
            mask_hashes.add((h,w,input_hash))
            t=time.perf_counter(); exp,out=geometry(a,flags&1,flags>>1); cpu=(time.perf_counter()-t)*1000
            t=time.perf_counter(); stats,got=mask.run(a,erode=flags&1,dilate=flags>>1,job_id=i+1); pl=(time.perf_counter()-t)*1000
            check(stats,got,exp,out,i+1)
            records.append({'core':'mask','i':i,'h':h,'w':w,'flags':flags,'input_sha256':input_hash,'cpu_ms':cpu,'pl_total_ms':pl})
            mode=(i//len(sizes)//6)%2
            if mode:
                b=pattern(rng,h,w,(i//len(sizes))+3)
            else:
                a=rng.randint(0,256,(h,w)).astype(np.uint8)
                b=rng.randint(0,256,(h,w)).astype(np.uint8)
            threshold=(0,1,30,127,254,255)[(i//len(sizes))%6]
            t=time.perf_counter(); exp,out=evidence(a,b,mode,threshold); cpu=(time.perf_counter()-t)*1000
            t=time.perf_counter(); stats,got=visual.run(a,b,mode=mode,threshold=threshold,job_id=i+10001); pl=(time.perf_counter()-t)*1000
            check(stats,got,exp,out,i+10001)
            records.append({'core':'visual','i':i,'h':h,'w':w,'mode':mode,'threshold':threshold,'cpu_ms':cpu,'pl_total_ms':pl})
            if i%50==0: print('PASS pairs=%d elapsed=%.1fs' % (i+1,time.monotonic()-started),flush=True)
        if args.count>=1256 and len(mask_hashes)<1000:
            raise AssertionError('fewer than 1000 distinct synthetic mask inputs')
        report['passed']=True
    except BaseException as exc:
        report['error']=repr(exc)
        raise
    finally:
        report['elapsed_sec']=time.monotonic()-started
        report['distinct_mask_inputs']=len(mask_hashes)
        Path(args.output).write_text(json.dumps(report,indent=2),encoding='utf-8')
        if not session.poisoned: session.overlay.download()
    print('PASS %d mask + %d visual cases' % (args.count,args.count),flush=True)


if __name__=='__main__': main()
