"""Build a traced 26-class public segmentation dataset without using local targets.

TACO source polygons are preserved; community annotations remain explicitly
unreviewed. Network failures and rare classes are reported, never filled by
pretending repeated images are new objects. Local target/test images are not read.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import threading
import time
import urllib.request

import cv2
import numpy as np
import requests

ROOT=Path(__file__).resolve().parents[1]
SESSIONS=threading.local()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def polygons(annotations,categories,names,width,height):
    lines=[]
    for ann in annotations:
        name=categories[ann['category_id']]
        if name not in names:
            continue
        pieces=ann.get('segmentation')
        if ann.get('iscrowd') or not isinstance(pieces,list) or not pieces:
            raise ValueError('target_without_polygon')
        checked=[]
        for piece in pieces:
            points=np.asarray(piece,dtype=np.float64).reshape(-1,2)
            if len(points)<3 or not np.isfinite(points).all():
                raise ValueError('invalid_target_polygon')
            if np.any(points < -1) or np.any(points[:,0]>width+1) or np.any(points[:,1]>height+1):
                raise ValueError('target_polygon_out_of_bounds')
            points[:,0]=np.clip(points[:,0],0,width-1)
            points[:,1]=np.clip(points[:,1],0,height-1)
            checked.append(points)
        if len(checked)>1:
            # YOLO polygons do not represent disconnected pieces faithfully.
            # Keep the image only if all pieces touch and form one outer contour.
            scale=min(1,960/max(width,height))
            mask=np.zeros((round(height*scale),round(width*scale)),np.uint8)
            cv2.fillPoly(mask,[np.rint(p*scale).astype(np.int32) for p in checked],1)
            contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            if len(contours)!=1:
                raise ValueError('disconnected_target_polygon')
            points=contours[0][:,0,:].astype(float)/scale
        else:points=checked[0]
        if len(points)<3 or cv2.contourArea(points.astype(np.float32))<4:
            raise ValueError('degenerate_target_polygon')
        points=points/[width,height]
        lines.append((names.index(name),points,ann['id']))
    return lines


def dhash(image):
    small=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(9,8))
    bits=(small[:,1:]>small[:,:-1]).flatten()
    return sum(int(bit)<<i for i,bit in enumerate(bits))


def download(job,cache):
    key=job['image_key']
    target=cache/(sha(key.encode())+'.image')
    sidecar=target.with_suffix('.json')
    if target.exists() and sidecar.exists():
        data=target.read_bytes(); provenance=json.loads(sidecar.read_text())
        if sha(data)!=provenance['sha256']:
            raise ValueError('cache_hash_changed')
    else:
        failures=[]
        if not hasattr(SESSIONS,'session'):
            SESSIONS.session=requests.Session()
            SESSIONS.session.headers['User-Agent']='Mozilla/5.0'
        for url in job['urls']:
            try:
                with SESSIONS.session.get(url,timeout=(6,12)) as source:
                    source.raise_for_status()
                    data=source.content
                if len(data)>20*1024*1024:raise ValueError('oversize_image')
                image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
                if image is None:raise ValueError('not_an_image')
                provenance=dict(url=url,sha256=sha(data),bytes=len(data))
                target.write_bytes(data);write(sidecar,provenance)
                break
            except Exception as error:failures.append(dict(url=url,error=str(error)[:180]))
        else:raise ValueError(json.dumps(failures))
    image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
    if image is None:raise ValueError('invalid_cached_image')
    h,w=image.shape[:2]
    if abs(w/h-job['width']/job['height'])>.015:
        raise ValueError('source_dimensions_or_aspect_mismatch')
    if max(w,h)>1280:
        image=cv2.resize(image,(round(w*1280/max(w,h)),round(h*1280/max(w,h))))
    return job,image,provenance,dhash(image)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sources',type=Path,required=True)
    p.add_argument('--workers',type=int,default=10)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    out=args.output.resolve()
    if not out.is_relative_to(ROOT/'local-only'):raise ValueError('Use local-only output')
    if (out/'report.json').exists():raise FileExistsError('Completed dataset must not be overwritten')
    out.mkdir(parents=True,exist_ok=args.resume)
    cache=out.parent/'cache';cache.mkdir(exist_ok=True)
    original=json.loads((ROOT/'pynq_km1_vision/config/yoloe_taco_recyclables.json').read_text())
    classes=[dict(row) for row in original['classes']]
    classes.extend([dict(official_label='Drink carton',material='paper',grasp_confidence=.75),
                    dict(official_label='Plastic straw',material='plastic',grasp_confidence=.75)])
    names=[row['official_label'] for row in classes]
    if len(names)!=26 or len(set(names))!=26:raise ValueError('Expected 24 existing classes plus two authorized classes')
    write(out/'taxonomy.json',dict(classes=classes,material_note='Sorting policy, not pure material analysis',
                                 source_policy_sha256=sha((ROOT/'pynq_km1_vision/config/yoloe_taco_recyclables.json').read_bytes())))
    jobs=[]; source_records=[]; failures=[]
    mirror_file=args.sources.resolve()/'taco-current.json'
    mirror_raw=mirror_file.read_bytes()
    mirror_images={r['file_name']:r for r in json.loads(mirror_raw)['images']}
    source_records.append(dict(source='publisher_image_mirror_metadata',path=str(mirror_file),sha256=sha(mirror_raw)))
    for source_name,filename in [('official','taco-1.0.json'),('community','taco-unofficial.json')]:
        path=args.sources.resolve()/filename;raw=path.read_bytes();doc=json.loads(raw)
        source_records.append(dict(source=source_name,path=str(path),sha256=sha(raw),
                                   source_reviewed=source_name=='official'))
        categories={r['id']:r['name'] for r in doc['categories']}
        annotations=defaultdict(list)
        for ann in doc['annotations']:annotations[ann['image_id']].append(ann)
        for im in doc['images']:
            anns=annotations[im['id']]
            if not any(categories[a['category_id']] in names for a in anns):continue
            urls=list(dict.fromkeys(im[k] for k in ['flickr_640_url','flickr_url','coco_url'] if im.get(k)))
            if source_name=='official':
                mirrored=mirror_images.get(im['file_name'],{})
                urls=list(dict.fromkeys([mirrored[k] for k in ['flickr_640_url','flickr_url'] if mirrored.get(k)]+urls))
            urls=list(dict.fromkeys([re.sub(r'https://farm\d+\.staticflickr.com/',
                            'https://live.staticflickr.com/',url) for url in urls]+urls))
            if not urls:continue
            match=re.search(r'/([0-9]{8,})[_./]',urls[0])
            key='flickr-'+match.group(1) if match else source_name+'-'+str(im['id'])
            try:lines=polygons(anns,categories,names,im['width'],im['height'])
            except Exception as error:
                failures.append(dict(source=source_name,image_id=im['id'],stage='annotation',error=str(error)));continue
            jobs.append(dict(source=source_name,image_id=im['id'],image_key=key,urls=urls,
                 source_file_name=im['file_name'],width=im['width'],height=im['height'],license=im.get('license'),
                 source_reviewed=source_name=='official',polygons=lines))
    # Prefer reviewed official annotations when sources contain the same photo.
    unique={}
    for job in jobs:unique.setdefault(job['image_key'],job)
    source_duplicate_keys=len(jobs)-len(unique)
    jobs=list(unique.values())
    write(out/'download-plan.json',dict(classes=names,sources=source_records,planned_images=len(jobs),
          duplicate_source_keys=source_duplicate_keys,workers=args.workers,resumed=args.resume,
          label_quality='Source annotation plus geometry checks; community is unreviewed',
          local_target_training_images=0,local_test_images_opened=False))
    print('Planned public images:',len(jobs),flush=True)
    results=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(download,job,cache):job for job in jobs}
        for index,future in enumerate(as_completed(futures),1):
            # Release completed Future objects: each contains a decoded image.
            job=futures.pop(future)
            try:
                result=future.result()
                # Store bounded JPEGs immediately, rather than retaining all decoded images.
                key=sha(job['image_key'].encode())[:20]
                folder=out/'staging';folder.mkdir(exist_ok=True)
                target=folder/(key+'.jpg')
                if not cv2.imwrite(str(target),result[1]):raise ValueError('image_write_failed')
                results.append(dict(job=job,path=target,download=result[2],dhash=result[3],
                                    image_sha256=sha(target.read_bytes())))
            except Exception as error:
                failure=dict(source=job['source'],image_id=job['image_id'],stage='download',error=str(error)[:300])
                failures.append(failure)
                with (out/'download-failures.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(failure)+'\n')
            if index%100==0 or index==len(jobs):
                progress=dict(processed=index,total=len(jobs),downloaded=len(results),failures=len(failures))
                write(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    # The same resized bytes or near-identical appearance cannot cross splits.
    kept=[]; seen_sha=set();hashes=[]
    for row in sorted(results,key=lambda r:(r['job']['source']!='official',r['job']['image_key'])):
        near=next((previous for value,previous in hashes if (row['dhash']^value).bit_count()<=2),None)
        if row['image_sha256'] in seen_sha or near is not None:
            failures.append(dict(source=row['job']['source'],image_id=row['job']['image_id'],stage='deduplicate',
                                 error='exact_or_dhash_distance_le_2',matched_image=near));continue
        seen_sha.add(row['image_sha256']);hashes.append((row['dhash'],row['job']['image_key']));kept.append(row)
    counts=Counter(cid for row in kept for cid,_,_ in row['job']['polygons'])
    val_counts=Counter();remaining=counts.copy();manifest=[]
    # Validation comes only from the reviewed source. Preserve at least three
    # training instances, and report categories too rare to evaluate.
    for row in sorted(kept,key=lambda r:sha(r['job']['image_key'].encode())):
        job=row['job'];local=Counter(cid for cid,_,_ in job['polygons'])
        wants=any(val_counts[cid]<max(1,int(counts[cid]*.15)) and counts[cid]>=10 for cid in local)
        allowed=all(remaining[cid]-count>=3 for cid,count in local.items())
        split='val' if job['source']=='official' and wants and allowed else 'train'
        if split=='val':val_counts.update(local);remaining.subtract(local)
        image=out/'images'/split/row['path'].name
        label=out/'labels'/split/row['path'].with_suffix('.txt').name
        image.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True)
        row['path'].rename(image)
        label.write_text('\n'.join(str(cid)+' '+' '.join('%.8f'%v for v in points.flatten())
                            for cid,points,_ in job['polygons'])+'\n',encoding='utf-8',newline='\n')
        manifest.append(dict(image=str(image.relative_to(out)).replace('\\','/'),split=split,
            image_key=job['image_key'],source=job['source'],source_image_id=job['image_id'],
            source_url=row['download']['url'],source_image_sha256=row['download']['sha256'],
            image_sha256=sha(image.read_bytes()),label_sha256=sha(label.read_bytes()),
            source_license=job['license'],source_annotation_reviewed=job['source_reviewed'],
            target_classes=[cid for cid,_,_ in job['polygons']],source_annotation_ids=[a for _,_,a in job['polygons']],
            synthetic=False,physical_identity_verified=False,dhash64='%016x'%row['dhash']))
    yaml='path: '+out.as_posix()+'\ntrain: images/train\nval: images/val\nnames:\n'
    yaml+=''.join('  %d: %s\n'%(i,json.dumps(name)) for i,name in enumerate(names))
    (out/'data.yaml').write_text(yaml,encoding='utf-8')
    train_counts=Counter(cid for r in manifest if r['split']=='train' for cid in r['target_classes'])
    stats={names[i]:dict(train_instances=train_counts[i],val_instances=val_counts[i],
                        below_50_real_instances=train_counts[i]<50) for i in range(len(names))}
    write(out/'manifest.json',dict(classes=names,images=manifest,sources=source_records,
          split_note='Reviewed public validation photos; exact and near duplicates removed. Physical objects/brands not verified disjoint.',
          local_target_training_images=0,local_test_images_opened=False))
    write(out/'report.json',dict(images=len(manifest),splits=dict(Counter(r['split'] for r in manifest)),
          source_images=dict(Counter(r['source'] for r in manifest)),per_class=stats,failures=failures,
          count_note='Image and annotation counts are not verified distinct physical objects.',
          license_note='Keep source license fields and provenance; no dataset redistribution cleared.',
          community_quality_note='Source community labels are not individually reviewed; inspect previews before training.',
          all_26_have_train_examples=all(train_counts[i]>0 for i in range(len(names)))))
    # One contact sheet per class, including rare classes; polygons are shown.
    preview=out/'previews';preview.mkdir()
    for cid,name in enumerate(names):
        candidates=[r for r in manifest if cid in r['target_classes']][:12]
        sheet=np.full((3*190,4*240,3),245,np.uint8)
        for i,r in enumerate(candidates):
            image=cv2.imread(str(out/r['image']));h,w=image.shape[:2]
            for line in (out/'labels'/r['split']/Path(r['image']).with_suffix('.txt').name).read_text().splitlines():
                values=[float(v) for v in line.split()]
                if int(values[0])==cid:
                    points=np.asarray(values[1:]).reshape(-1,2)*[w,h]
                    cv2.polylines(image,[np.rint(points).astype(np.int32)],True,(0,255,0),2)
            scale=min(240/w,165/h);image=cv2.resize(image,(round(w*scale),round(h*scale)))
            y,x=(i//4)*190,(i%4)*240
            sheet[y:y+image.shape[0],x:x+image.shape[1]]=image
            cv2.putText(sheet,r['source']+' '+Path(r['image']).stem[:8],(x+3,y+182),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1)
        cv2.imwrite(str(preview/('%02d-'%cid+name.replace(' ','_')+'.jpg')),sheet)
    print(json.dumps(dict(images=len(manifest),splits=dict(Counter(r['split'] for r in manifest)),per_class=stats)),flush=True)


if __name__=='__main__':main()
