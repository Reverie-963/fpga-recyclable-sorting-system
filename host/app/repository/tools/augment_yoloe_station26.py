"""Add explicitly synthetic public-object views to a confirmed empty workcell.

Only public TRAIN masks are pasted. Existing local positive/validation/test
photographs are not read. Synthetic views are never counted as new real objects.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--views-per-class',type=int,default=20)
    args=p.parse_args();root=args.dataset.resolve()
    if not root.is_relative_to(ROOT/'local-only'):raise ValueError('Use local-only dataset')
    if (root/'station-augmentation.json').exists():raise FileExistsError('Keep previous augmentation')
    manifest=read(root/'manifest.json');report=read(root/'report.json')
    workcell_root=ROOT/'local-only/recyclables-final-16-v1'
    workcell=read(workcell_root/'workcell.json');background=workcell_root/workcell['baseline']
    if sha(background)!=workcell['baseline_sha256']:raise ValueError('Registered empty reference changed')
    base=cv2.imread(str(background));h,w=base.shape[:2]
    if [w,h]!=workcell['image_size']:raise ValueError('Reference geometry changed')
    names=manifest['classes'];rng=random.Random(20261030)
    eligible={i:[] for i in range(len(names))}
    for row in manifest['images']:
        if row['split']!='train' or row.get('synthetic'):continue
        path=root/'labels/train'/Path(row['image']).with_suffix('.txt').name
        for line in path.read_text().splitlines():
            values=[float(v) for v in line.split()];cid=int(values[0])
            eligible[cid].append((row,np.asarray(values[1:]).reshape(-1,2)))
    if any(not values for values in eligible.values()):raise ValueError('Missing a real training category')
    # Preserve immutable public-only metadata before appending synthetic views.
    save(root/'manifest-public-only.json',manifest);save(root/'report-public-only.json',report)
    added=[];source_keys=set();preview=[]
    for cid in range(len(names)):
        pool=eligible[cid]
        reviewed=[p for p in pool if p[0]['source_annotation_reviewed']]
        pool=reviewed or pool
        rng.shuffle(pool)
        generated=0;attempts=0
        while generated<args.views_per_class and attempts<args.views_per_class*10:
            parent,points=pool[attempts%len(pool)];attempts+=1
            image=cv2.imread(str(root/parent['image']));ih,iw=image.shape[:2]
            mask=np.zeros((ih,iw),np.uint8);cv2.fillPoly(mask,[np.rint(points*[iw,ih]).astype(np.int32)],255)
            ys,xs=np.nonzero(mask)
            if len(xs)<40:continue
            x0,x1,y0,y1=int(xs.min()),int(xs.max())+1,int(ys.min()),int(ys.max())+1
            color=image[y0:y1,x0:x1];alpha=mask[y0:y1,x0:x1]
            span=rng.randint(60,115)
            scale=span/max(color.shape[:2])
            tw,th=max(2,round(color.shape[1]*scale)),max(2,round(color.shape[0]*scale))
            color=cv2.resize(color,(tw,th));alpha=cv2.resize(alpha,(tw,th),interpolation=cv2.INTER_NEAREST)
            size=max(tw,th)*2+8;center=(size//2,size//2)
            canvas=np.zeros((size,size,3),np.uint8);am=np.zeros((size,size),np.uint8)
            ox,oy=center[0]-tw//2,center[1]-th//2
            canvas[oy:oy+th,ox:ox+tw]=color;am[oy:oy+th,ox:ox+tw]=alpha
            transform=cv2.getRotationMatrix2D(center,rng.uniform(-180,180),1.)
            canvas=cv2.warpAffine(canvas,transform,(size,size));am=cv2.warpAffine(am,transform,(size,size),flags=cv2.INTER_NEAREST)
            ys,xs=np.nonzero(am)
            if len(xs)<20:continue
            x0,x1,y0,y1=int(xs.min()),int(xs.max())+1,int(ys.min()),int(ys.max())+1
            canvas=canvas[y0:y1,x0:x1];am=am[y0:y1,x0:x1]
            zone='source' if rng.random()<.8 else rng.choice(['can','plastic','cardboard'])
            rx0,ry0,rx1,ry1=workcell['regions'][zone];ch,cw=am.shape
            if cw>=rx1-rx0-4 or ch>=ry1-ry0-4:continue
            px=rng.randint(rx0+2,rx1-cw-2);py=rng.randint(ry0+2,ry1-ch-2)
            result=base.copy();full=np.zeros((h,w),np.uint8);full[py:py+ch,px:px+cw]=am
            foreground=am>0
            patch=result[py:py+ch,px:px+cw];patch[foreground]=canvas[foreground]
            gain=rng.uniform(.85,1.15);bias=rng.uniform(-8,8)
            result=np.clip(result.astype(float)*gain+bias,0,255).astype(np.uint8)
            contours,_=cv2.findContours(full,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            if len(contours)!=1 or len(contours[0])<3:continue
            poly=contours[0][:,0,:].astype(float)/[w,h]
            stem='station-synthetic-%02d-%03d'%(cid,generated)
            ip=root/'images/train'/(stem+'.jpg');lp=root/'labels/train'/(stem+'.txt')
            if ip.exists() or lp.exists():raise FileExistsError(stem)
            cv2.imwrite(str(ip),result)
            lp.write_text(str(cid)+' '+' '.join('%.8f'%v for v in poly.flatten())+'\n',encoding='utf-8',newline='\n')
            added.append(dict(image=ip.relative_to(root).as_posix(),split='train',image_key=stem,
                image_sha256=sha(ip),label_sha256=sha(lp),target_classes=[cid],source='public-object-workcell-composite',
                synthetic=True,parent_public_image=parent['image'],parent_image_key=parent['image_key'],
                parent_source_url=parent['source_url'],background_sha256=sha(background),
                source_annotation_reviewed=parent['source_annotation_reviewed'],physical_identity_verified=False,
                synthetic_zone=zone,local_target_photograph_used=False))
            source_keys.add(parent['image_key']);generated+=1
            if generated==1:
                cv2.polylines(result,[np.rint(poly*[w,h]).astype(np.int32)],True,(0,255,0),1)
                preview.append((cid,result))
    # One true empty image, plus seven explicit lighting augmentations; no
    # fabricated extra real captures or object counts.
    for i in range(8):
        result=base.copy() if i==0 else np.clip(base.astype(float)*rng.uniform(.8,1.2)+rng.uniform(-10,10),0,255).astype(np.uint8)
        stem='station-empty-%02d'%i;ip=root/'images/train'/(stem+'.jpg');lp=root/'labels/train'/(stem+'.txt')
        cv2.imwrite(str(ip),result);lp.write_text('',encoding='utf-8')
        added.append(dict(image=ip.relative_to(root).as_posix(),split='train',image_key=stem,
            image_sha256=sha(ip),label_sha256=sha(lp),target_classes=[],source='registered-empty-workcell',
            synthetic=i!=0,parent_background_sha256=sha(background),source_annotation_reviewed=True,
            physical_identity_verified=False,local_target_photograph_used=False))
    manifest['images'].extend(added);manifest['station_augmentation']=dict(background_sha256=sha(background),
        unique_parent_public_photos=len(source_keys),synthetic_views=sum(r['synthetic'] for r in added),
        local_real_empty_images=1,local_positive_images=0,all_public_parents_train_only=True)
    report['public_only_per_class']=report['per_class']
    report['training_instances_with_augmentation']={names[i]:sum(r['target_classes'].count(i) for r in manifest['images'] if r['split']=='train') for i in range(len(names))}
    report['images']=len(manifest['images']);report['splits']=dict(Counter(r['split'] for r in manifest['images']))
    report['station_augmentation']=manifest['station_augmentation']
    save(root/'manifest.json',manifest);save(root/'report.json',report);save(root/'station-augmentation.json',manifest['station_augmentation'])
    sheet=np.full((7*180,4*240,3),245,np.uint8)
    for i,(cid,im) in enumerate(preview):
        tile=cv2.resize(im,(240,180));cv2.putText(tile,'%02d %s'%(cid,names[cid][:19]),(3,16),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,0,255),1)
        y,x=i//4*180,i%4*240;sheet[y:y+180,x:x+240]=tile
    cv2.imwrite(str(root/'station-composites-preview.jpg'),sheet)
    print(json.dumps(manifest['station_augmentation']),flush=True)


if __name__=='__main__':main()
