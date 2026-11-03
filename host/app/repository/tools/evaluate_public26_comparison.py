"""Evaluate one frozen 26-class candidate under the common public/local policy.

Public validation gives fine-class mask AP. Existing workcell validation gives
three-material F1 and errors; it never selects checkpoints. Test is not scored.
"""
import argparse
from collections import defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--kind', choices=['yolo','yoloe'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='0')
    args = p.parse_args()
    out, dataset = args.output.resolve(), args.dataset.resolve()
    if not out.is_relative_to(ROOT / 'local-only'):
        raise ValueError('Use local-only output')
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
        YOLO_AUTOINSTALL='false', YOLO_CONFIG_DIR=str(out / 'settings'))
    sys.path.insert(0, str(ROOT / 'local-only/yoloe11m-26m-comparison/runtime'))
    sys.path.insert(0, str(ROOT / 'pynq_km1_vision'))
    import cv2
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO, YOLOE
    from ultralytics.engine.model import Model as EngineModel
    from ultralytics.models.yolo.segment import SegmentationValidator
    from compare_material_models import truth_for, predictions_for, decode_mask, overlay, f1_for
    from src.train_recyclables_final import new_counts, merge_counts, describe_counts, label_path, match_instances
    from src.yoloe_inference import encode_mask
    from src.station_instance_filter import annotate_station
    if ultralytics.__version__ != '8.4.193':
        raise ValueError('Pinned comparison runtime required')
    torch.set_num_threads(4)
    torch.backends.cudnn.enabled = False
    taxonomy = read(dataset / 'taxonomy.json')['classes']
    names = [row['official_label'] for row in taxonomy]
    class_map = {i:{'metal':0, 'plastic':1, 'paper':2}[row['material']] for i,row in enumerate(taxonomy)}
    local = ROOT / 'local-only/recyclables-final-16-v1'
    manifest = read(local / 'split_manifest.json')
    rows = sorted((r for r in manifest['images'] if r['split']=='val'), key=lambda r:r['image'])
    if len(rows) != 121:
        raise ValueError('Expected the same 121 workcell validation images')
    workcell = read(local / 'workcell.json')
    reference = cv2.imread(str(local / workcell['baseline']))
    if sha(local / workcell['baseline']) != workcell['baseline_sha256']:
        raise ValueError('Empty reference changed')
    reference_hash = hashlib.sha256(reference.tobytes()).hexdigest()
    filter_path = ROOT / 'pynq_km1_vision/config/yoloe_station_filter.example.json'
    config = read(filter_path)
    if (config['workcell_id'] != workcell['workcell_id'] or
        config['reference_image_sha256'] != workcell['baseline_sha256'] or
        config['image_size'] != workcell['image_size']):
        raise ValueError('Filter/reference identity mismatch')
    before = {str(local/r['image']):sha(local/r['image']) for r in rows}
    before.update({str(label_path(local,r)):sha(label_path(local,r)) for r in rows})
    policy = dict(device=args.device, imgsz=640, conf=.3, iou=.7, max_det=20,
        agnostic_nms=True, retina_masks=True, verbose=False, save=False, batch=1, half=args.device!='cpu')
    save(out / 'plan.json', dict(weight_sha256=sha(args.weights), kind=args.kind,
        dataset_manifest_sha256=sha(dataset/'manifest.json'), local_manifest_sha256=sha(local/'split_manifest.json'),
        filter_config_sha256=sha(filter_path), torch=torch.__version__, ultralytics=ultralytics.__version__,
        public=dict(conf=.001, iou=.7, max_det=100, imgsz=640, batch=2, half=args.device!='cpu'),
        local=policy, local_mask_iou=.5, local_split='val', test_images_scored=False,
        repeated_validation=True, selects_checkpoints=False, deployed=False))
    model = (YOLOE if args.kind=='yoloe' else YOLO)(str(args.weights.resolve()))
    if list(model.names.values()) != names:
        raise ValueError('Candidate taxonomy/order mismatch')
    metrics = EngineModel.val(model, validator=SegmentationValidator, data=str(dataset/'data.yaml'),
        device=args.device, batch=2, imgsz=640, workers=0, plots=False, save_json=False,
        conf=.001, iou=.7, max_det=100, half=args.device!='cpu',
        project=str(out), name='public-val', verbose=False)
    report = read(dataset/'report.json')
    valid = [i for i,name in enumerate(names) if report['public_only_per_class'][name]['val_instances']>0]
    public = dict(metrics={k:float(v) for k,v in metrics.results_dict.items()},
        per_class_mask_ap50_95={name:float(metrics.seg.maps[i]) if i in valid else None for i,name in enumerate(names)},
        classes_with_ground_truth=len(valid), classes_without_ground_truth=[names[i] for i in range(26) if i not in valid])
    save(out/'public.json', public)
    total, source_total = new_counts(), new_counts()
    scenes = defaultdict(new_counts)
    records, alignments, timings = [], [], []
    # Warm up the same local inference policy before recording call durations.
    warmup = cv2.imread(str(local/rows[0]['image']))
    for _ in range(3):
        model.predict(warmup, **policy)
    with (out/'local-predictions.jsonl').open('x', encoding='utf-8') as stream:
        for index,row in enumerate(rows,1):
            image = cv2.imread(str(local/row['image']))
            truth = truth_for(local,row,image.shape[:2])
            if args.device!='cpu': torch.cuda.synchronize()
            started = time.perf_counter()
            raw = model.predict(image, **policy)[0]
            if args.device!='cpu': torch.cuda.synchronize()
            timings.append((time.perf_counter()-started)*1000)
            predictions = predictions_for(raw,image.shape[:2],class_map)
            counts = match_instances(truth,predictions,.5)
            merge_counts(total,counts)
            merge_counts(scenes[row['scene_type']],counts)
            encoded = [dict(**{k:v for k,v in pred.items() if k!='mask'},mask_rle=encode_mask(pred['mask'])) for pred in predictions]
            shift,response = cv2.phaseCorrelate(cv2.cvtColor(reference,cv2.COLOR_BGR2GRAY).astype(np.float32),
                cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32))
            aligned = response>=.3 and max(abs(v) for v in shift)<=2 and row['workcell_id']==workcell['workcell_id']
            accepted = []
            if aligned:
                filtered = annotate_station(image,reference,encoded,regions=config['regions'],workcell_id=row['workcell_id'],
                    reference_workcell_id=workcell['workcell_id'],alignment_verified=True,reference_sha256=reference_hash,
                    inside_fraction=config['inside_fraction'],changed_fraction=config['changed_fraction'],
                    pixel_difference=config['pixel_difference'])
                accepted = [pred for pred,details in zip(predictions,filtered['instances']) if details['station_grasp_candidate']]
            else:
                filtered = dict(filter_unavailable=True,automatic_motion_allowed=False)
            source_truth = truth if row['scene_type'] in ('source','multi') else []
            source_counts = match_instances(source_truth,accepted,.5)
            merge_counts(source_total,source_counts)
            record = dict(image=row['image'],image_sha256=before[str(local/row['image'])],shape=list(image.shape[:2]),
                predictions=encoded,counts=counts,source_filter_counts=source_counts,
                alignment=dict(verified=aligned,shift=list(shift),response=response),station=filtered,call_ms=timings[-1])
            stream.write(json.dumps(record,allow_nan=False)+'\n')
            stream.flush()
            records.append(record)
            alignments.append(aligned)
            if index in (1,2,26,74,110): cv2.imwrite(str(out/f'sample-{index:03d}.jpg'),overlay(image,truth,predictions))
            if index%30==0: print('Local validation',index,'/',len(rows),flush=True)
    recounted = new_counts()
    for row,record in zip(rows,records,strict=True):
        decoded = [dict(class_id=pred['class_id'],confidence=pred['confidence'],mask=decode_mask(pred['mask_rle'],record['shape']))
                   for pred in record['predictions']]
        checked = match_instances(truth_for(local,row,record['shape']),decoded,.5)
        if checked != record['counts']: raise ValueError('Serialized mask score mismatch')
        merge_counts(recounted,checked)
    if recounted!=total or before!={path:sha(path) for path in before}:
        raise ValueError('Score audit or source integrity failed')
    overall = describe_counts(total,['can','plastic','cardboard'])
    source = describe_counts(source_total,['can','plastic','cardboard'])
    save(out/'summary.json',dict(weight_sha256=sha(args.weights),public=public,local=dict(overall=overall,
        macro_f1=sum(f1_for(v) for v in overall['per_class'].values())/3,
        scenes={key:describe_counts(value,['can','plastic','cardboard']) for key,value in scenes.items()},
        source_filter=source,source_filter_macro_f1=sum(f1_for(v) for v in source['per_class'].values())/3,
        aligned_images=sum(alignments),total_images=len(rows),saved_mask_audit=True,
        inference_call_median_ms=float(np.median(timings)),inference_call_p95_ms=float(np.percentile(timings,95)),
        timing_note='Same-machine services remain running; image IO, scoring, station filter, camera and motion excluded'),
        local_inputs_unchanged=True,test_images_scored=False,deployed=False))
    del model,raw
    gc.collect()


if __name__=='__main__':
    main()
