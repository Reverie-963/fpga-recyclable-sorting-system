from .config import validate
"""Capture a full frame and segmented single-object crop (no GUI needed)."""
import argparse,json,time
from pathlib import Path
import cv2
from .vision import mask_cpu,candidates

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='config.json');p.add_argument('--label')
    p.add_argument('--split',choices=['train','val','test'],default='train');p.add_argument('--count',type=int,default=1)
    p.add_argument('--output-dir',default='data/frames',help='Directory for full frames; use one folder per physical sample/session')
    p.add_argument('--interval',type=float,default=2);a=p.parse_args();c=validate(json.loads(Path(a.config).read_text(encoding='utf-8-sig')));v=c['vision']
    if a.label and (not a.label.replace('_','').isalnum() or not a.label.isascii()): p.error('Use ASCII class name')
    cap=cv2.VideoCapture(c['camera']['index']);cap.set(3,c['camera']['width']);cap.set(4,c['camera']['height'])
    try:
        if not cap.isOpened(): raise RuntimeError('No camera')
        for _ in range(10): cap.read()
        for i in range(a.count):
            # Keep draining the camera during the interval rather than reading an old buffered frame.
            end=time.monotonic()+max(.1,a.interval)
            while time.monotonic()<end:
                ok,img=cap.read()
                if not ok: raise RuntimeError('Camera failure')
            stamp=str(time.time_ns());out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
            frame_path=out/(stamp+'.jpg')
            if not cv2.imwrite(str(frame_path),img): raise IOError('Could not save frame: '+str(frame_path))
            if a.label:
                objs=candidates(mask_cpu(img,v['background_bgr'],v['threshold']),v['min_area'],v['max_area'])
                if len(objs)!=1: print('Skipped crop: need exactly one foreground object');continue
                x,y,w,h=objs[0]['bbox'];dest=Path('data/crops')/a.split/a.label;dest.mkdir(parents=True,exist_ok=True)
                cv2.imwrite(str(dest/(stamp+'.png')),img[y:y+h,x:x+w])
            print('saved',str(frame_path),flush=True)
    finally: cap.release()
if __name__=='__main__':main()
