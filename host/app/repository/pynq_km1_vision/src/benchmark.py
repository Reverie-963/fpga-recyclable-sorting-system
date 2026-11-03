from .config import validate
"""Same images, same operation, full packing+DMA+output-copy cost included for PL."""
import argparse,json,time
from pathlib import Path
import cv2,numpy as np
from .vision import mask_cpu,open_mask

def summary(x):return {'mean_ms':float(np.mean(x)),'p50_ms':float(np.percentile(x,50)),'p95_ms':float(np.percentile(x,95)),'p99_ms':float(np.percentile(x,99)),'max_ms':float(np.max(x))}
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='config.json');p.add_argument('--images',default='data/frames')
    p.add_argument('--pl',action='store_true');p.add_argument('--repeat',type=int,default=20)
    p.add_argument('--output',default='report/benchmark.json')
    p.add_argument('--evidence-dir',help='Save first-pass images and masks outside the timed operation')
    a=p.parse_args()
    c=validate(json.loads(Path(a.config).read_text(encoding='utf-8-sig')));v=c['vision'];images=[];names=[]
    def golden(im):
        raw=mask_cpu(im,v['background_bgr'],v['threshold'])
        return open_mask(raw) if v.get('pl_kernel','threshold')=='open' else raw
    for f in sorted(Path(a.images).glob('*')):
        img=cv2.imread(str(f))
        if img is not None: images.append(img);names.append(f.name)
    if not images or a.repeat<1: raise ValueError('Need images and positive repeat')
    if len(set(im.shape for im in images))!=1: raise ValueError('Images must have identical dimensions')
    evidence=Path(a.evidence_dir) if a.evidence_dir else None
    if evidence:evidence.mkdir(parents=True,exist_ok=True)
    def save_image(name,img):
        if not cv2.imwrite(str(evidence/name),img):raise OSError('Cannot save evidence: '+name)
    pl=None;cpu_times=[];pl_times=[];checks=[]
    try:
        if a.pl:
            from .pl import Segmenter
            pl=Segmenter(c['bitfile'],images[0].shape[:2],kernel=v.get('pl_kernel','threshold'))
        for _ in range(3):
            golden(images[0])
            if pl:pl.run(images[0],v['background_bgr'],v['threshold'])
        for i in range(a.repeat):
            for index,im in enumerate(images):
                # Alternate timing order to reduce systematic warm-cache bias.
                def cpu():
                    t=time.perf_counter();r=golden(im);cpu_times.append((time.perf_counter()-t)*1000);return r
                def hw():
                    t=time.perf_counter();r=pl.run(im,v['background_bgr'],v['threshold']);pl_times.append((time.perf_counter()-t)*1000);return r
                if pl and i%2: actual=hw();expected=cpu()
                else:
                    expected=cpu()
                    if pl:actual=hw()
                mismatch=int(np.count_nonzero(expected!=actual)) if pl else None
                if i==0:
                    checks.append({'file':names[index],'foreground_pixels':int(np.count_nonzero(expected)),
                                   'mismatched_pixels':mismatch})
                if evidence and (i==0 or (pl and mismatch)):
                    prefix='%02d_%s_r%02d'%(index,Path(names[index]).stem,i)
                    save_image(prefix+'_input.png',im)
                    save_image(prefix+'_cpu.png',expected)
                    if pl:
                        save_image(prefix+'_pl.png',actual)
                        save_image(prefix+'_diff.png',(expected!=actual).astype(np.uint8)*255)
                if pl and mismatch: raise AssertionError('PL disagrees with CPU golden output: %s, repeat %d, %d pixels'%(names[index],i,mismatch))
        result={'operation':'BGR L1 '+v.get('pl_kernel','threshold'),'samples':len(cpu_times),'shape':list(images[0].shape),'cpu':summary(cpu_times),
                'pl':summary(pl_times) if pl else None,'exact_match':True if pl else None,
                'speedup':np.mean(cpu_times)/np.mean(pl_times) if pl else None,
                'host':__import__('platform').platform(),'images':checks,
                'background_bgr':v['background_bgr'],'threshold':v['threshold'],
                'evidence_dir':str(evidence) if evidence else None}
        output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
    finally:
        if pl:pl.close()
if __name__=='__main__': main()
