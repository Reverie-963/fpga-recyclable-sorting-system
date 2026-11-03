"""Tune material decision thresholds on existing train images; validate on val only."""
import argparse
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'pynq_km1_vision'))
from src.prompted_recyclables import read_policy, settings_for
from src.sorting_loop import mask_of, iou

MATERIALS=('metal','plastic','paper')


def score(images, thresholds):
    counts={m:dict(tp=0,fp=0,fn=0) for m in MATERIALS}
    empty_fp=0
    for row in images:
        used=set();predictions=[p for p in row['predictions'] if p['confidence']>=thresholds[p['material']]]
        if not row['truth'] and predictions:empty_fp+=1
        for p in sorted(predictions,key=lambda p:-p['confidence']):
            candidates=[(p['ious'][i],i) for i,t in enumerate(row['truth']) if t==p['material'] and i not in used]
            overlap,index=max(candidates,default=(0,-1))
            if index>=0 and overlap>=.5:counts[p['material']]['tp']+=1;used.add(index)
            else:counts[p['material']]['fp']+=1
        for i,m in enumerate(row['truth']):
            if i not in used:counts[m]['fn']+=1
    for v in counts.values():
        v['precision']=v['tp']/(v['tp']+v['fp']) if v['tp']+v['fp'] else None
        v['recall']=v['tp']/(v['tp']+v['fn']) if v['tp']+v['fn'] else None
    return dict(images=len(images),counts=counts,empty_false_positive_images=empty_fp)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--trained',action='store_true',help='Evaluate the existing supervised model without training')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    dataset=ROOT/'local-only/recyclables-final-16-v1'
    manifest=json.loads((dataset/'split_manifest.json').read_text(encoding='utf-8'))
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    url='http://127.0.0.1:8778' if args.trained else 'http://127.0.0.1:8811'
    with opener.open(url+'/health',timeout=5) as response:health=json.load(response)
    settings=health.get('settings',{'settings_sha256':health.get('runtime_id')});policy=read_policy()
    rows=[r for r in manifest['images'] if r['split'] in ('train','val')]
    results=[]
    for index,row in enumerate(rows):
        image_path=dataset/row['image'];image_bytes=image_path.read_bytes()
        assert hashlib.sha256(image_bytes).hexdigest()==row['image_sha256']
        frame=cv2.imdecode(np.frombuffer(image_bytes,np.uint8),cv2.IMREAD_COLOR)
        cache=out/(image_path.stem+'.json')
        if cache.exists():prediction=json.loads(cache.read_text(encoding='utf-8'))
        else:
            payload=json.dumps(dict(image_base64=base64.b64encode(image_bytes).decode(),
                                    prompt_mode='supervised' if args.trained else 'text')).encode()
            req=urllib.request.Request(url+'/detect',data=payload,headers={'Content-Type':'application/json'})
            with opener.open(req,timeout=30) as response:prediction=json.load(response)
            prediction.pop('overlay_base64',None)
            cache.write_text(json.dumps(prediction,ensure_ascii=False),encoding='utf-8')
        if args.trained:
            assert prediction['runtime_id']==settings['settings_sha256'] and prediction['model_version']==health['model_sha256']
            assert prediction['image_sha256']==hashlib.sha256(image_bytes).hexdigest()
        else:
            assert prediction['settings_sha256']==settings['settings_sha256']
            assert prediction['source_bgr_sha256']==hashlib.sha256(frame.tobytes()).hexdigest()
        label_path=dataset/'labels'/row['split']/Path(row['image']).with_suffix('.txt').name
        truth=[]
        for line in label_path.read_text(encoding='utf-8-sig').splitlines():
            values=[float(v) for v in line.split()]
            if not values:continue
            polygon=np.asarray(values[1:]).reshape(-1,2)*[frame.shape[1],frame.shape[0]]
            mask=np.zeros(frame.shape[:2],np.uint8)
            cv2.fillPoly(mask,[np.rint(polygon).astype(np.int32)],1)
            truth.append((MATERIALS[int(values[0])],mask.astype(bool)))
        predictions=[]
        for instance in prediction['instances']:
            mask=mask_of(instance,prediction['image_size'])
            predictions.append(dict(official_label=instance.get('official_label',instance['label']),material=instance['label'] if args.trained else instance['material'],
                confidence=instance['confidence'],ious=[iou(mask,m) for _,m in truth]))
        results.append(dict(image=row['image'],split=row['split'],scene=row['scene_type'],
            truth=[m for m,_ in truth],predictions=predictions))
        if (index+1)%20==0:print('images',index+1,'/',len(rows),flush=True)
    train=[r for r in results if r['split']=='train'];val=[r for r in results if r['split']=='val']
    chosen={};curves={}
    for material in MATERIALS:
        curve=[]
        for value in np.arange(max(.15,health.get('confidence_threshold',.1)),.951,.025):
            threshold=round(float(value),3)
            metrics=score(train,{m:threshold if m==material else .75 for m in MATERIALS})['counts'][material]
            p=metrics['precision'] or 0;r=metrics['recall'] or 0
            curve.append(dict(threshold=threshold,**metrics,f05=1.25*p*r/(.25*p+r) if p+r else 0))
        # Prefer >=90% train precision if available; otherwise report the best
        # observed tradeoff, without claiming the deployment is qualified.
        eligible=[r for r in curve if (r['precision'] or 0)>=.9 and r['tp']>=5]
        selected=max(eligible or curve,key=lambda r:(r['recall'] or 0,r['precision'] or 0,-r['threshold']) if eligible else (r['f05'],r['precision'] or 0,-r['threshold']))
        chosen[material]=selected['threshold'];curves[material]=curve
    report=dict(kind='trained-material-threshold-calibration-v1' if args.trained else 'prompted-material-threshold-calibration-v1',model_sha256=health['model_sha256'] if args.trained else policy['model']['sha256'],
        settings_sha256=settings['settings_sha256'],dataset_manifest_sha256=hashlib.sha256((dataset/'split_manifest.json').read_bytes()).hexdigest(),
        threshold_fit_split='train',validation_split='val',sealed_test_used=False,model_trained=False,
        original_fine_labels_annotated=False,mask_iou_threshold=.5,chosen=chosen,curves=curves,
        baseline_train=score(train,{m:.75 for m in MATERIALS}),baseline_val=score(val,{m:.75 for m in MATERIALS}),
        tuned_train=score(train,chosen),tuned_val=score(val,chosen),
        by_scene={s:score([r for r in val if r['scene']==s],chosen) for s in sorted({r['scene'] for r in val})})
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'scored-images.json').write_text(json.dumps(results,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('curves','by_scene')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
