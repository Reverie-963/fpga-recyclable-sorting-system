"""Explicit retrospective comparison on the already-used fixed16 test set.

Does not train, deploy, contact hardware, reserve a fresh test, or erase test-use
receipts. The user must explicitly request reuse of the already-opened test set.
All candidates use CPU FP32 and the same class-agnostic NMS before scoring.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pynq_km1_vision'))
import cv2
import numpy as np
from src.train_recyclables_final import (check_final_dataset, describe_counts,
    label_path, match_instances, merge_counts, new_counts, parse_labels, read_json,
    sha256)
from src.yoloe_inference import encode_mask, config_hash

NAMES = ['can', 'plastic', 'cardboard']
DATA_HASH = '2d929f1a4aad581d296d9513a7837d2dab66cc94e7075d2ca6d48576f68c0d91'
POLICY = dict(device='cpu', half=False, imgsz=640, conf=.3, iou=.7,
              max_det=20, agnostic_nms=True, retina_masks=True, verbose=False,
              save=False, batch=1)
FOLLOWUP = ROOT / 'local-only/recyclables-v8-followup/20261029-144923'
CAMPAIGN = ROOT / 'local-only/recyclables-model-comparison/20261029-104817'
ASSETS = ROOT / 'local-only/yoloe-deployment/assets'
SPECS = [
    ('v8-parent', FOLLOWUP / 'tested-parent/model.pt',
     '9288e697042d14813d7d47abbe2e413f314be59c3069c2302f2dffee5912dd0e'),
    ('v8-continued', FOLLOWUP / 'continued-candidate/model.pt',
     '78425124e1bee96ec514ab1b4dc634c607da0c5a5047575f23b7d581d78b87ce'),
    ('yolo11s', CAMPAIGN / 'models/yolo11s-seg/training/weights/best.pt',
     'd7436017eeb2aa1cc414c2afd23b666b545287fdbcd032536800f1c2998d0ba9'),
    ('yoloe-visual', ASSETS / 'yoloe-materials-visual.pt',
     '671411af722566cdd57e93ed2623f01a8c275bbd34719aeb16d0aa424732f120'),
]
NANO_SPEC = ('yolo11n', ROOT / 'local-only/recyclables-final-runs/final16-improved-20261029-033656/training/weights/epoch70.pt',
             '5ca3df98922bcd5015b10b80e6022be651b2f0f8e7444361d0da425c0fb6aee1')


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8')
    for attempt in range(10):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.1)


def decode_mask(runs, shape):
    if any(type(x) is not int or x < 0 for x in runs) or sum(runs) != int(np.prod(shape)):
        raise ValueError('Invalid full-frame RLE')
    return np.repeat(np.arange(len(runs), dtype=np.uint32) % 2,
                     runs).reshape(shape).astype(bool)


def truth_for(root, row, shape):
    h, w = shape
    result = []
    for cid, polygon in parse_labels(label_path(root, row)):
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [np.rint(polygon * [w, h]).astype(np.int32)], 1)
        result.append(dict(class_id=cid, mask=mask.astype(bool)))
    return result


def predictions_for(raw, shape, class_map):
    if raw.boxes is None or len(raw.boxes) == 0:
        return []
    if raw.masks is None or len(raw.masks.data) != len(raw.boxes):
        raise ValueError('Missing instance masks')
    masks = raw.masks.data.detach().cpu().numpy()
    if tuple(masks.shape[1:]) != tuple(shape):
        raise ValueError('Masks must use original image coordinates')
    result = []
    for index, (raw_id, score) in enumerate(zip(raw.boxes.cls.tolist(), raw.boxes.conf.tolist())):
        source_id = int(raw_id)
        if raw_id != source_id or source_id not in class_map:
            raise ValueError('Unmapped model class')
        result.append(dict(class_id=class_map[source_id], source_class_id=source_id,
            confidence=float(score), mask=masks[index] > .5,
            detection_bbox_xyxy=raw.boxes.xyxy[index].tolist()))
    return result


def rank(fixed):
    values = fixed['overall']['per_class']
    reference = all((values[k][m] or 0) >= .6 for k in NAMES for m in ['precision', 'recall'])
    reference = reference and fixed['overall']['empty_false_positive_images'] == 0
    primary = [(fixed['scenes'][s]['per_class'][k][m] or 0)
        for s in ['source', 'bin'] for k in NAMES for m in ['precision', 'recall']]
    f1 = [f1_for(v) for group in [fixed['overall']] + [fixed['scenes'][s]
        for s in ['source', 'bin', 'wrong_bin', 'multi']]
        for v in group['per_class'].values()]
    return [int(reference), min(primary), sum(f1)/len(f1),
            -sum(v['fp'] for v in values.values())]


def f1_for(value):
    p, r = value['precision'] or 0, value['recall'] or 0
    return 2*p*r/(p+r) if p+r else 0


def error_details(truth, predictions):
    """Explain errors without changing confidence-ordered class-aware scoring."""
    used, errors = set(), []
    for pred in sorted(predictions, key=lambda p: p['confidence'], reverse=True):
        overlaps = []
        for index, target in enumerate(truth):
            union = np.logical_or(pred['mask'], target['mask']).sum()
            iou = float(np.logical_and(pred['mask'], target['mask']).sum()/union) if union else 0.
            overlaps.append((iou, index))
        score, best = max((v for v in overlaps if v[1] not in used and
            truth[v[1]]['class_id'] == pred['class_id']), default=(0., -1))
        if best >= 0 and score >= .5:
            used.add(best)
            continue
        overlap, target_index = max(overlaps, default=(0., -1))
        kind = 'background_or_localization'
        if target_index >= 0 and overlap >= .5:
            kind = 'duplicate' if truth[target_index]['class_id'] == pred['class_id'] else 'wrong_material'
        elif target_index >= 0 and overlap > 0 and truth[target_index]['class_id'] == pred['class_id']:
            kind = 'mask_iou_below_0.5'
        errors.append(dict(type='fp', kind=kind, predicted=NAMES[pred['class_id']],
            best_truth=NAMES[truth[target_index]['class_id']] if target_index >= 0 else None,
            best_iou=overlap, confidence=pred['confidence']))
    for index, target in enumerate(truth):
        if index not in used:
            errors.append(dict(type='fn', material=NAMES[target['class_id']]))
    return errors


def overlay(frame, truth, predictions):
    out = frame.copy()
    for collection, color in [(truth, (0, 220, 0)), (predictions, (0, 80, 255))]:
        for item in collection:
            contours, _ = cv2.findContours(item['mask'].astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, color, 1)
            y, x = np.nonzero(item['mask'])
            if len(x):
                text = NAMES[item['class_id']]
                if 'confidence' in item:
                    text += ' %.2f' % item['confidence']
                cv2.putText(out, text, (int(x.min()), max(15, int(y.min())-3)),
                            cv2.FONT_HERSHEY_SIMPLEX, .42, color, 1)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--acknowledge-used-test', action='store_true', required=True)
    parser.add_argument('--only-yolo11n', action='store_true',
                        help='Score the existing nano checkpoint, then compare with a frozen prior run.')
    parser.add_argument('--baseline-comparison', type=Path)
    args = parser.parse_args()
    baseline = None
    if args.only_yolo11n:
        if args.baseline_comparison is None:
            parser.error('--only-yolo11n requires --baseline-comparison')
        baseline = read_json(args.baseline_comparison)
        if baseline['policy'] != POLICY or baseline['dataset_sha256'] != DATA_HASH:
            raise ValueError('Baseline policy or dataset differs')
        if not baseline['immutable_inputs_and_old_test_receipts_unchanged']:
            raise ValueError('Baseline integrity audit did not pass')
    elif args.baseline_comparison is not None:
        parser.error('--baseline-comparison is for the nano extension only')
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'local-only'):
        raise ValueError('Outputs must stay local-only')
    output.mkdir(parents=True, exist_ok=False)
    root = ROOT / 'local-only/recyclables-final-16-v1'
    recipe = read_json(CAMPAIGN / 'comparison-recipe.json')
    dataset = check_final_dataset(root, recipe)
    assert dataset['dataset_sha256'] == DATA_HASH
    write_json(output / 'dataset-check.json', dataset)
    manifest = read_json(root / 'split_manifest.json')
    shutil.copy2(root / 'split_manifest.json', output / 'manifest.frozen.json')
    tracked = [root / 'test-use.json',
        root / 'annotation-draft-v1/finalization/sealed-test-v1-reservation.json',
        FOLLOWUP / 'sealed-evaluation/result.json',
        FOLLOWUP / 'sealed-evaluation/predictions.jsonl',
        ASSETS / 'visual_references.json', ASSETS / 'deployment.json',
        ROOT / 'pynq_km1_vision/src/train_recyclables_final.py', Path(__file__)]
    if baseline is not None:
        tracked.append(args.baseline_comparison.resolve())
    before = {str(p): sha256(p) for p in tracked}
    receipts = read_json(root / 'test-use.json')
    lock = read_json(ASSETS / 'deployment.json')
    references = read_json(ASSETS / 'visual_references.json')
    assert config_hash(references) == lock['visual_references_sha256']
    evaluated_ids = {oid for r in manifest['images'] if r['split'] in ('val', 'test') for oid in r['object_ids']}
    assert not evaluated_ids.intersection(r['object_id'] for r in references)
    assert all(r['source_split'] == 'train' for r in references)
    specs = []
    for name, path, expected in ([NANO_SPEC] if args.only_yolo11n else SPECS):
        assert sha256(path) == expected, name + ' weights changed'
        destination = output / 'weights' / (name + '.pt')
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(path, destination)
        assert sha256(destination) == expected
        specs.append(dict(name=name, source=str(path), frozen=str(destination), sha256=expected))
    plan = dict(schema_version='material-retrospective-comparison-v1', started_at=now(),
        authorization=('User explicitly requested adding the existing YOLO11n to the same already-opened four-object comparison; no new photos exist.'
            if args.only_yolo11n else 'User explicitly requested all four models on the original held-out four objects; no new photos exist.'),
        models=specs, policy=POLICY, mask_iou=.5, cpu_threads=4,
        class_names=NAMES, runner_sha256=sha256(Path(__file__)),
        dataset_sha256=DATA_HASH, split_sha256=dataset['split_sha256'],
        test_status='already_opened_retrospective_not_fresh_independent_acceptance',
        prior_test_use=receipts, immutable_before=before,
        validation_ranking=['overall 60% reference and empty FP=0', 'source/bin minimum P/R',
                            'mean F1 across overall and four scenes', 'fewer FP'],
        test_used_for_selection=False, new_physical_objects_or_photos=False,
        policy_change_from_previous='Class-agnostic box NMS at IoU .7 is common to every candidate, removing YOLOE prompt-ID NMS differences.',
        limitations=['Four objects per split, same-day fixed workcell; not arbitrary new-object generalization.',
            'All validation data have previously been used for development; old test is explicitly retrospective.',
            'Models differ in training and prompt preparation; this is candidate comparison, not architecture ablation.',
            'CPU offline timings cannot rank live GPU throughput.'],
        deployed=False, trained=False, physical_motion=False)
    if baseline is not None:
        assert baseline['split_sha256'] == dataset['split_sha256']
        plan['baseline_comparison'] = str(args.baseline_comparison.resolve())
        plan['baseline_sha256'] = sha256(args.baseline_comparison)
        plan['baseline_policy_identical'] = True
    write_json(output / 'plan.json', plan)
    shutil.copy2(Path(__file__), output / 'runner.frozen.py')
    for filename in ['deployment.json', 'visual_references.json']:
        shutil.copy2(ASSETS / filename, output / filename)
    os.environ['YOLO_CONFIG_DIR'] = str(output / 'ultralytics-settings')
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ['YOLO_OFFLINE'] = 'true'
    import torch
    import ultralytics
    from ultralytics import YOLO, YOLOE, settings
    assert ultralytics.__version__ == '8.3.230'
    assert torch.__version__ == '2.12.0+rocm7.14.0'
    torch.backends.cudnn.enabled = False
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    settings.update({key: False for key in ['sync', 'clearml', 'comet', 'dvc', 'hub',
        'mlflow', 'neptune', 'raytune', 'tensorboard', 'wandb'] if key in settings})
    write_json(output / 'environment.json', dict(torch=torch.__version__,
        ultralytics=ultralytics.__version__, numpy=np.__version__, cv2=cv2.__version__,
        device='cpu', threads=torch.get_num_threads(), half=False))
    completed, summaries, audit = 0, {}, []
    total = len(specs) * (dataset['images']['val'] + dataset['images']['test'])
    for spec in specs:
        name = spec['name']
        model = (YOLOE if name == 'yoloe-visual' else YOLO)(spec['frozen']).to('cpu')
        if name == 'yoloe-visual':
            prompts = lock['visual_prompts']
            assert list(model.names.values()) == [p['text'] for p in prompts]
            materials = dict(metal=0, plastic=1, paper=2)
            class_map = {i: materials[p['category']] for i, p in enumerate(prompts)}
        else:
            assert list(model.names.values()) == NAMES
            class_map = {i: i for i in range(3)}
        summaries[name] = {}
        for split in ('val', 'test'):
            rows = sorted((r for r in manifest['images'] if r['split'] == split), key=lambda r: r['image'])
            folder = output / name / split
            folder.mkdir(parents=True)
            totals, groups, details, times, kinds = new_counts(), defaultdict(new_counts), [], [], Counter()
            saved_errors = 0
            predictions_path = folder / 'predictions.jsonl'
            with predictions_path.open('x', encoding='utf-8') as stream:
                for index, row in enumerate(rows, 1):
                    frame = cv2.imread(str(root / row['image']))
                    shape = frame.shape[:2]
                    assert hashlib.sha256(frame.tobytes()).hexdigest() not in {
                        r['decoded_bgr_sha256'] for r in references}
                    truth = truth_for(root, row, shape)
                    started = time.perf_counter()
                    raw = model.predict(frame, **POLICY)[0]
                    elapsed = (time.perf_counter()-started)*1000
                    assert str(next(model.model.parameters()).device) == 'cpu'
                    predictions = predictions_for(raw, shape, class_map)
                    counts = match_instances(truth, predictions, .5)
                    errors = error_details(truth, predictions)
                    assert sum(e['type'] == 'fp' for e in errors) == sum(counts['fp'])
                    assert sum(e['type'] == 'fn' for e in errors) == sum(counts['fn'])
                    kinds.update(e['kind'] for e in errors if e['type'] == 'fp')
                    encoded = [dict(**{k:v for k,v in p.items() if k != 'mask'},
                        mask_rle=encode_mask(p['mask'])) for p in predictions]
                    stream.write(json.dumps(dict(image=row['image'], image_sha256=sha256(root / row['image']),
                        label_sha256=sha256(label_path(root, row)), shape=list(shape),
                        predictions=encoded, counts=counts, inference_ms=elapsed), allow_nan=False)+'\n')
                    stream.flush()
                    merge_counts(totals, counts)
                    merge_counts(groups[row['scene_type']], counts)
                    times.append(elapsed)
                    details.append(dict(image=row['image'], scene_type=row['scene_type'],
                        object_ids=row['object_ids'], counts=counts, errors=errors))
                    if errors and saved_errors < 6:
                        cv2.imwrite(str(folder / ('error-%03d.jpg' % index)), overlay(frame, truth, predictions))
                        saved_errors += 1
                    completed += 1
                    if index % 10 == 0 or index == len(rows):
                        progress = dict(stage='evaluating', model=name, split=split, split_completed=index,
                            split_total=len(rows), completed=completed, total=total, updated_at=now(), pid=os.getpid())
                        write_json(output / 'progress.json', progress)
                        print(json.dumps(progress), flush=True)
            fixed = dict(overall=describe_counts(totals, NAMES),
                scenes={key:describe_counts(value, NAMES) for key, value in groups.items()})
            fixed['macro_f1'] = sum(f1_for(v) for v in fixed['overall']['per_class'].values())/3
            result = dict(model=name, model_sha256=spec['sha256'], split=split,
                retrospective=(split == 'test'), fixed=fixed, fp_diagnosis=dict(kinds),
                images=details, prediction_sha256=sha256(predictions_path),
                inference_ms=dict(median=float(np.median(times)), max=max(times),
                    note='CPU offline; cold start included; live monitor concurrent; not GPU FPS.'))
            write_json(folder / 'result.json', result)
            summaries[name][split] = {k:v for k,v in result.items() if k != 'images'}
            # Re-score every saved RLE without querying a model again.
            recounted = new_counts()
            regroups = defaultdict(new_counts)
            for row, line in zip(rows, predictions_path.read_text(encoding='utf-8').splitlines(), strict=True):
                record = json.loads(line)
                assert record['image'] == row['image']
                decoded = [dict(class_id=p['class_id'], confidence=p['confidence'],
                    mask=decode_mask(p['mask_rle'], record['shape'])) for p in record['predictions']]
                checked = match_instances(truth_for(root, row, record['shape']), decoded, .5)
                assert checked == record['counts']
                merge_counts(recounted, checked)
                merge_counts(regroups[row['scene_type']], checked)
            assert recounted == totals and dict(regroups) == dict(groups)
            audit.append(dict(model=name, split=split, saved_rle_images=len(rows), counts_identical=True))
        del model, raw
        gc.collect()
    after = {str(p): sha256(p) for p in tracked}
    assert before == after, 'Input, code, or old test evidence changed'
    post_dataset = check_final_dataset(root, recipe)
    assert post_dataset == dataset
    for spec in specs:
        assert sha256(spec['source']) == sha256(spec['frozen']) == spec['sha256']
    ranking = sorted(summaries, key=lambda n: rank(summaries[n]['val']['fixed']), reverse=True)
    report = dict(plan, completed_at=now(), candidates=summaries,
        validation_ranking=ranking, validation_recommendation=ranking[0],
        selection_evidence='validation_only_not_retrospective_test',
        immutable_inputs_and_old_test_receipts_unchanged=True, original_data_unchanged=True,
        model_calls=completed, audit=audit)
    write_json(output / 'comparison-report.json', report)
    write_json(output / 'audit.json', dict(passed=True, before=before, after=after,
        dataset_unchanged=True, models_unchanged=True, rle_rescores=audit))
    write_json(output / 'progress.json', dict(stage='complete', completed=completed,
        total=total, updated_at=now(), validation_recommendation=ranking[0], pid=os.getpid()))
    print(json.dumps(dict(complete=True, validation_ranking=ranking)), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
