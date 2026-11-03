"""Evaluate a separate 26-class candidate and baseline; never deploy or move."""
import argparse
from collections import defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='0')
    p.add_argument('--only',choices=['baseline','candidate'])
    args=p.parse_args();out=args.output.resolve();dataset=args.dataset.resolve()
    if not out.is_relative_to(ROOT/'local-only'):raise ValueError('Use local-only output')
    out.mkdir(parents=True,exist_ok=False)
    os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        YOLO_AUTOINSTALL='false',YOLO_CONFIG_DIR=str(out/'settings'))
    sys.path.insert(0,str(ROOT/'local-only/yoloe11m-26m-comparison/runtime'))
    sys.path.insert(0,str(ROOT/'pynq_km1_vision'))
    import cv2
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLOE
    from ultralytics.engine.model import Model as EngineModel
    from ultralytics.models.yolo.segment import SegmentationValidator
    from compare_material_models import truth_for,predictions_for,decode_mask,overlay,f1_for
    from src.train_recyclables_final import new_counts,merge_counts,describe_counts,label_path,match_instances
    from src.yoloe_inference import encode_mask
    from src.station_instance_filter import annotate_station
    torch.set_num_threads(4);torch.backends.cudnn.enabled=False
    if ultralytics.__version__!='8.4.193':raise ValueError('Use private pinned Ultralytics 8.4.193')
    taxonomy=read(dataset/'taxonomy.json')['classes'];names=[r['official_label'] for r in taxonomy]
    class_map={i:{'metal':0,'plastic':1,'paper':2}[r['material']] for i,r in enumerate(taxonomy)}
    local=ROOT/'local-only/recyclables-final-16-v1'
    manifest=read(local/'split_manifest.json')
    rows=sorted((r for r in manifest['images'] if r['split']=='val'),key=lambda r:r['image'])
    workcell=read(local/'workcell.json');reference=cv2.imread(str(local/workcell['baseline']))
    if sha(local/workcell['baseline'])!=workcell['baseline_sha256']:raise ValueError('Reference changed')
    reference_hash=hashlib.sha256(reference.tobytes()).hexdigest()
    filter_path=ROOT/'pynq_km1_vision/config/yoloe_station_filter.example.json'
    filter_config=read(filter_path)
    if (filter_config['workcell_id']!=workcell['workcell_id'] or filter_config['reference_image_sha256']!=workcell['baseline_sha256']
        or filter_config['image_size']!=workcell['image_size']):
        raise ValueError('Filter configuration is not bound to this empty reference')
    regions=filter_config['regions']
    before={str(local/r['image']):sha(local/r['image']) for r in rows}
    before.update({str(label_path(local,r)):sha(label_path(local,r)) for r in rows})
    summaries={}
    save(out/'plan.json',dict(dataset_manifest_sha256=sha(dataset/'manifest.json'),
       local_manifest_sha256=sha(local/'split_manifest.json'),local_split='val',test_images_scored=False,
       baseline_sha256=sha(args.baseline),candidate_sha256=sha(args.candidate),
       local_confidence=.3,local_mask_iou=.5,public_ap_confidence_floor=.001,
       precision='fp16' if args.device!='cpu' else 'fp32',
       runtime=dict(ultralytics=ultralytics.__version__,torch=torch.__version__),
       evaluated_subset=args.only,
       public_note='Reviewed public photo validation; physical/brand disjointness is not established',
       filter_note='Historical matching workcell only; not current camera commissioning',
       filter_config_sha256=sha(filter_path),
       phase_alignment=dict(min_response=.3,max_shift_px=2.),deploy=False))
    for name,path in [('baseline',args.baseline),('candidate',args.candidate)]:
        if args.only and name!=args.only:continue
        folder=out/name;folder.mkdir()
        model=YOLOE(str(path.resolve()))
        if list(model.names.values())!=names:raise ValueError('Taxonomy mismatch')
        metrics=EngineModel.val(model,validator=SegmentationValidator,data=str(dataset/'data.yaml'),
            device=args.device,batch=2,imgsz=640,workers=0,plots=False,save_json=False,
            conf=.001,iou=.7,max_det=100,half=args.device!='cpu',project=str(out),name=name+'-public-val',verbose=False)
        report=read(dataset/'report.json')
        valid=[i for i,n in enumerate(names) if report['public_only_per_class'][n]['val_instances']>0]
        public=dict(metrics={k:float(v) for k,v in metrics.results_dict.items()},
            per_class_mask_ap50_95={names[i]:float(metrics.seg.maps[i]) if i in valid else None for i in range(26)},
            classes_with_ground_truth=len(valid),classes_without_ground_truth=[names[i] for i in range(26) if i not in valid])
        save(folder/'public.json',public)
        policy=dict(device=args.device,imgsz=640,conf=.3,iou=.7,max_det=20,
                    agnostic_nms=True,retina_masks=True,verbose=False,save=False,batch=1,half=args.device!='cpu')
        total,source_total=new_counts(),new_counts();scenes=defaultdict(new_counts);alignments=[]
        records=[]
        with (folder/'local-predictions.jsonl').open('x',encoding='utf-8') as stream:
            for index,row in enumerate(rows,1):
                image=cv2.imread(str(local/row['image']));truth=truth_for(local,row,image.shape[:2])
                raw=model.predict(image,**policy)[0]
                predictions=predictions_for(raw,image.shape[:2],class_map)
                counts=match_instances(truth,predictions,.5);merge_counts(total,counts);merge_counts(scenes[row['scene_type']],counts)
                encoded=[dict(**{k:v for k,v in pred.items() if k!='mask'},mask_rle=encode_mask(pred['mask'])) for pred in predictions]
                shift,response=cv2.phaseCorrelate(cv2.cvtColor(reference,cv2.COLOR_BGR2GRAY).astype(np.float32),
                                                 cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32))
                aligned=response>=.3 and max(abs(v) for v in shift)<=2 and row['workcell_id']==workcell['workcell_id']
                accepted=[]
                if aligned:
                    filtered=annotate_station(image,reference,encoded,regions=regions,workcell_id=row['workcell_id'],
                        reference_workcell_id=workcell['workcell_id'],alignment_verified=True,reference_sha256=reference_hash,
                        inside_fraction=filter_config['inside_fraction'],changed_fraction=filter_config['changed_fraction'],
                        pixel_difference=filter_config['pixel_difference'])
                    accepted=[p for p,r in zip(predictions,filtered['instances']) if r['station_grasp_candidate']]
                else:filtered=dict(instances=encoded,filter_unavailable=True,automatic_motion_allowed=False)
                source_truth=truth if row['scene_type'] in ('source','multi') else []
                filtered_counts=match_instances(source_truth,accepted,.5);merge_counts(source_total,filtered_counts)
                record=dict(image=row['image'],image_sha256=sha(local/row['image']),shape=list(image.shape[:2]),
                    predictions=encoded,counts=counts,source_filter_counts=filtered_counts,
                    alignment=dict(verified=aligned,shift=list(shift),response=response),station=filtered)
                stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();records.append(record)
                alignments.append(aligned)
                if index in [1,2,26,74,110]:cv2.imwrite(str(folder/('sample-%03d.jpg'%index)),overlay(image,truth,predictions))
                if index%30==0:print(name,'local validation',index,'/',len(rows),flush=True)
        # Every serialized raw mask is decoded and scored again.
        recounted=new_counts()
        for row,record in zip(rows,records,strict=True):
            decoded=[dict(class_id=r['class_id'],confidence=r['confidence'],mask=decode_mask(r['mask_rle'],record['shape'])) for r in record['predictions']]
            checked=match_instances(truth_for(local,row,record['shape']),decoded,.5)
            assert checked==record['counts'];merge_counts(recounted,checked)
        assert recounted==total
        overall=describe_counts(total,['can','plastic','cardboard'])
        source=describe_counts(source_total,['can','plastic','cardboard'])
        summary=dict(weight_sha256=sha(path),public=public,local=dict(overall=overall,
            macro_f1=sum(f1_for(v) for v in overall['per_class'].values())/3,
            scenes={k:describe_counts(v,['can','plastic','cardboard']) for k,v in scenes.items()},
            source_filter=source,source_filter_macro_f1=sum(f1_for(v) for v in source['per_class'].values())/3,
            aligned_images=sum(alignments),total_images=len(rows),saved_mask_audit=True),deployed=False)
        save(folder/'summary.json',summary);summaries[name]=summary
        del model,raw;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    assert before=={path:sha(path) for path in before}
    save(out/'comparison.json',dict(candidates=summaries,local_inputs_unchanged=True,test_images_scored=False,
                                  deployed=False,evaluated_subset=args.only,finished_at=time.time()))


if __name__=='__main__':main()
