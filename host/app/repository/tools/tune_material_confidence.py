"""Validation-only confidence search for five frozen material candidates.

Never loads test images, trains, edits labels, deploys or contacts live services.
Only confidence changes; image size, class mapping, box NMS and mask IoU stay fixed.
"""
import argparse
from collections import defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import compare_material_models as c
import cv2
import numpy as np

GRID = [round(x/100, 2) for x in range(5, 96)]


def keep(score, threshold):
    # Installed Ultralytics uses a strict > comparison in FP32, not >=.
    return bool(np.float32(score) > np.float32(threshold))


def prepare_prefix(truth, predictions):
    """Greedy matches are unchanged when a lower-confidence suffix is removed."""
    used, flags = set(), []
    for pred in sorted(predictions, key=lambda p: p['confidence'], reverse=True):
        candidates = []
        for index, target in enumerate(truth):
            if index in used or target['class_id'] != pred['class_id']:
                continue
            union = np.logical_or(pred['mask'], target['mask']).sum()
            iou = float(np.logical_and(pred['mask'], target['mask']).sum()/union) if union else 0.
            candidates.append((iou, index))
        score, index = max(candidates, default=(0., -1))
        matched = index >= 0 and score >= .5
        if matched:
            used.add(index)
        flags.append(dict(class_id=pred['class_id'], confidence=pred['confidence'], tp=matched))
    return dict(truth_counts=[sum(t['class_id'] == cid for t in truth) for cid in range(3)], flags=flags)


def counts_at(prefix, threshold):
    counts = c.new_counts()
    counts.update(images=1, empty_images=int(sum(prefix['truth_counts']) == 0))
    for flag in prefix['flags']:
        if keep(flag['confidence'], threshold):
            counts['tp' if flag['tp'] else 'fp'][flag['class_id']] += 1
    counts['fn'] = [n-tp for n, tp in zip(prefix['truth_counts'], counts['tp'])]
    counts['empty_false_positive_images'] = int(counts['empty_images'] and any(counts['fp']))
    return counts


def aggregate(prepared, threshold):
    total, scenes = c.new_counts(), defaultdict(c.new_counts)
    for row, prefix in prepared:
        counts = counts_at(prefix, threshold)
        c.merge_counts(total, counts)
        c.merge_counts(scenes[row['scene_type']], counts)
    result = dict(overall=c.describe_counts(total, c.NAMES),
                  scenes={s:c.describe_counts(v, c.NAMES) for s, v in scenes.items()})
    result['macro_f1'] = sum(c.f1_for(v) for v in result['overall']['per_class'].values())/3
    result['macro_precision'] = sum((v['precision'] or 0) for v in result['overall']['per_class'].values())/3
    result['macro_recall'] = sum((v['recall'] or 0) for v in result['overall']['per_class'].values())/3
    return result


def choose(sweep):
    # Same scene-aware validation ranking as the preceding comparison. Ties prefer
    # a value near the existing .3 setting, then the smaller threshold.
    return max(sweep, key=lambda v: (*c.rank(v['fixed']), -abs(v['confidence']-.3), -v['confidence']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT/'local-only'):
        raise ValueError('Evidence stays local-only')
    output.mkdir(parents=True, exist_ok=False)
    baseline = c.read_json(args.baseline)
    assert baseline['policy'] == c.POLICY and baseline['dataset_sha256'] == c.DATA_HASH
    root = ROOT/'local-only/recyclables-final-16-v1'
    manifest = c.read_json(root/'split_manifest.json')
    rows = sorted((r for r in manifest['images'] if r['split']=='val'), key=lambda r:r['image'])
    assert len(rows) == 121 and {r['split'] for r in rows} == {'val'}
    digest = hashlib.sha256()
    for row in rows:
        digest.update(Path(row['image']).as_posix().encode())
        digest.update((root/row['image']).read_bytes())
        digest.update(c.label_path(root,row).read_bytes())
    assert digest.hexdigest() == baseline['split_sha256']['val']
    lock = c.read_json(c.ASSETS/'deployment.json')
    reference = c.read_json(c.ASSETS/'visual_references.json')
    assert c.config_hash(reference) == lock['visual_references_sha256']
    assert not {oid for r in rows for oid in r['object_ids']}.intersection(r['object_id'] for r in reference)
    tracked = [args.baseline.resolve(), root/'split_manifest.json', root/'test-use.json',
        ROOT/'tools/compare_material_models.py', Path(__file__),
        ROOT/'pynq_km1_vision/src/train_recyclables_final.py',
        c.ASSETS/'deployment.json', c.ASSETS/'visual_references.json']
    before = {str(p):c.sha256(p) for p in tracked}
    specs=[]
    for spec in baseline['models']:
        assert c.sha256(spec['frozen']) == spec['sha256']
        frozen=output/'weights'/(spec['name']+'.pt')
        frozen.parent.mkdir(exist_ok=True)
        shutil.copy2(spec['frozen'], frozen)
        assert c.sha256(frozen) == spec['sha256']
        specs.append(dict(spec, frozen=str(frozen)))
    plan=dict(schema_version='material-confidence-validation-v1', started_at=c.now(), models=specs,
        baseline_sha256=c.sha256(args.baseline), validation_sha256=digest.hexdigest(),
        fixed_policy={k:v for k,v in c.POLICY.items() if k!='conf'}, mask_iou=.5,
        confidence_grid=GRID, inference_floor=.05, split='val', test_images_read=False,
        test_model_queries=0, test_used_for_selection=False, trained=False, deployed=False,
        selection='Existing validation rank: overall P/R >= .6 and empty0; source/bin minimum P/R; balanced group F1; fewer FP; tie nearest .3.',
        alternative='Highest overall macro F1 with no empty false-positive image; descriptive only.',
        cache_audit='All .3 counts must reproduce preceding baseline; recommended threshold re-run natively on all 121 validation images.',
        limitations=['Four validation objects, already used for model development; confidence tuning is development only.',
            'Single global threshold per model; no per-class or scene thresholds searched.',
            'Range .05-.95 in .01 steps, max_det20; not a global guarantee or calibrated probability.',
            'No historical test scores used to select a threshold.'], immutable_before=before)
    c.write_json(output/'plan.json',plan)
    shutil.copy2(Path(__file__),output/'runner.frozen.py')
    os.environ.update(YOLO_CONFIG_DIR=str(output/'ultralytics-settings'),YOLO_AUTOINSTALL='false',YOLO_OFFLINE='true')
    import torch
    import ultralytics
    from ultralytics import YOLO, YOLOE, settings
    assert ultralytics.__version__=='8.3.230' and torch.__version__=='2.12.0+rocm7.14.0'
    torch.set_num_threads(4)
    torch.backends.cudnn.enabled=False
    cv2.setNumThreads(1)
    settings.update({k:False for k in ['sync','clearml','comet','dvc','hub','mlflow','neptune','raytune','tensorboard','wandb'] if k in settings})
    summaries={}
    calls=0
    for spec in specs:
        name=spec['name']; folder=output/name;folder.mkdir()
        model=(YOLOE if name=='yoloe-visual' else YOLO)(spec['frozen']).to('cpu')
        if name=='yoloe-visual':
            assert list(model.names.values()) == [p['text'] for p in lock['visual_prompts']]
            mapping={i:dict(metal=0,plastic=1,paper=2)[p['category']] for i,p in enumerate(lock['visual_prompts'])}
        else:
            assert list(model.names.values()) == c.NAMES
            mapping={i:i for i in range(3)}
        prepared=[]
        previous_run=Path(baseline['new_report'] if name=='yolo11n' else baseline['prior_report']).parent
        baseline_image_counts={r['image']:r['counts'] for r in
            c.read_json(previous_run/name/'val/result.json')['images']}
        cache_path=folder/'validation-low-confidence.jsonl'
        with cache_path.open('x',encoding='utf-8') as stream:
            for index,row in enumerate(rows,1):
                image=cv2.imread(str(root/row['image']))
                truth=c.truth_for(root,row,image.shape[:2])
                raw=model.predict(image,**dict(c.POLICY,conf=.05))[0];calls+=1
                assert str(next(model.model.parameters()).device)=='cpu'
                predictions=c.predictions_for(raw,image.shape[:2],mapping)
                prefix=prepare_prefix(truth,predictions)
                assert counts_at(prefix,.3)==baseline_image_counts[row['image']], 'Low-threshold cache differs from preceding .3 evaluation'
                assert counts_at(prefix,.05)==c.match_instances(truth,predictions,.5)
                prepared.append((row,prefix))
                encoded=[dict(**{k:v for k,v in p.items() if k!='mask'},mask_rle=c.encode_mask(p['mask'])) for p in predictions]
                stream.write(json.dumps(dict(image=row['image'],shape=list(image.shape[:2]),predictions=encoded,prefix=prefix))+'\n')
                if index%10==0 or index==121:
                    c.write_json(output/'progress.json',dict(stage='low_confidence_inference',model=name,
                        completed=index,total=121,model_calls=calls,updated_at=c.now(),pid=os.getpid()))
        sweep=[dict(confidence=t,fixed=aggregate(prepared,t)) for t in GRID]
        selected=choose(sweep)
        alternative=max((v for v in sweep if v['fixed']['overall']['empty_false_positive_images']==0),
            key=lambda v:(v['fixed']['macro_f1'],-abs(v['confidence']-.3)) ,default=None)
        c.write_json(folder/'sweep.json',sweep)
        c.write_json(folder/'selected-before-native-audit.json',selected)
        # Independent native inference at the recommendation, not a test query.
        native_total,native_groups=c.new_counts(),defaultdict(c.new_counts)
        native_path=folder/'recommended-native-validation.jsonl'
        with native_path.open('x',encoding='utf-8') as stream:
            for index,(row,prefix) in enumerate(prepared,1):
                image=cv2.imread(str(root/row['image'])); truth=c.truth_for(root,row,image.shape[:2])
                raw=model.predict(image,**dict(c.POLICY,conf=selected['confidence']))[0];calls+=1
                predictions=c.predictions_for(raw,image.shape[:2],mapping)
                counts=c.match_instances(truth,predictions,.5)
                assert counts==counts_at(prefix,selected['confidence']), 'Recommended native threshold differs from cached curve'
                c.merge_counts(native_total,counts); c.merge_counts(native_groups[row['scene_type']],counts)
                stream.write(json.dumps(dict(image=row['image'],shape=list(image.shape[:2]),counts=counts,
                    predictions=[dict(**{k:v for k,v in p.items() if k!='mask'},mask_rle=c.encode_mask(p['mask'])) for p in predictions]))+'\n')
                if index%10==0 or index==121:
                    c.write_json(output/'progress.json',dict(stage='native_recommendation_audit',model=name,
                        confidence=selected['confidence'],completed=index,total=121,model_calls=calls,updated_at=c.now(),pid=os.getpid()))
        assert c.describe_counts(native_total,c.NAMES)==selected['fixed']['overall']
        assert {s:c.describe_counts(v,c.NAMES) for s,v in native_groups.items()}==selected['fixed']['scenes']
        # Re-read stored low-confidence masks and audit the full 91-point curves.
        rebuilt=[]
        for row,line in zip(rows,cache_path.read_text(encoding='utf-8').splitlines(),strict=True):
            value=json.loads(line); assert value['image']==row['image']
            predictions=[dict(class_id=p['class_id'],confidence=p['confidence'],mask=c.decode_mask(p['mask_rle'],value['shape'])) for p in value['predictions']]
            prefix=prepare_prefix(c.truth_for(root,row,value['shape']),predictions)
            assert prefix==value['prefix'];rebuilt.append((row,prefix))
        assert [dict(confidence=t,fixed=aggregate(rebuilt,t)) for t in GRID]==sweep
        summaries[name]=dict(model_sha256=spec['sha256'],recommended=selected,
            overall_f1_peak=alternative,at_03=next(v for v in sweep if v['confidence']==.3),
            at_08=next(v for v in sweep if v['confidence']==.8),baseline_03_all_images_match=True,
            native_recommendation_121_counts_match=True,all_91_stored_mask_curves_match=True,
            cache_sha256=c.sha256(cache_path),native_sha256=c.sha256(native_path))
        c.write_json(folder/'result.json',summaries[name])
        print(json.dumps(dict(model=name,confidence=selected['confidence'],f1=selected['fixed']['macro_f1'])),flush=True)
        del model,raw; gc.collect()
    after={str(p):c.sha256(p) for p in tracked}; assert before==after
    digest_after=hashlib.sha256()
    for row in rows:
        digest_after.update(Path(row['image']).as_posix().encode());digest_after.update((root/row['image']).read_bytes());digest_after.update(c.label_path(root,row).read_bytes())
    assert digest_after.hexdigest()==digest.hexdigest()
    for spec in specs: assert c.sha256(spec['source'])==c.sha256(spec['frozen'])==spec['sha256']
    report=dict(plan,completed_at=c.now(),results=summaries,model_calls=calls,
        immutable_inputs_unchanged=True,validation_unchanged=True,all_native_audits_passed=True)
    c.write_json(output/'confidence-report.json',report)
    c.write_json(output/'progress.json',dict(stage='complete',model_calls=calls,updated_at=c.now(),pid=os.getpid()))
    print(json.dumps(dict(complete=True,model_calls=calls)),flush=True)


if __name__=='__main__':
    main()
