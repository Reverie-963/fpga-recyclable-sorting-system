"""Read-only validation of frozen YOLOE prompts against an existing YOLO11 report.

Run with the recorded ROCm environment. Never trains, reads test images, deploys,
or contacts camera/actuator services. All evidence stays in a new local-only folder.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pynq_km1_vision'))

import cv2
import numpy as np
from src.train_recyclables_final import (describe_counts, label_path, match_instances,
    merge_counts, new_counts, parse_labels, read_json, save_json, sha256)
from src.yoloe_inference import YoloeDetector, annotate, convert_instances


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, campaign, output = (p.resolve() for p in (args.dataset, args.comparison, args.output))
    if not output.is_relative_to(ROOT / 'local-only'):
        raise ValueError('Evidence must stay in local-only')
    output.mkdir(parents=True, exist_ok=False)
    document = read_json(root / 'split_manifest.json')
    recipe = read_json(campaign / 'comparison-recipe.json')
    baseline_path = campaign / 'models/yolo11s-seg/result.json'
    baseline = read_json(baseline_path)['selected']
    frozen_data = read_json(campaign / 'models/yolo11s-seg/dataset-check.json')
    assert sha256(root / 'split_manifest.json') == frozen_data['manifest_sha256']
    rows = sorted((r for r in document['images'] if r['split'] == 'val'), key=lambda r: r['image'])
    assert len(rows) == 121
    assert [r['image'] for r in rows] == [r['image'] for r in baseline['fixed']['images']]
    assert document['class_names'] == recipe['class_names'] == ['can', 'plastic', 'cardboard']
    assert sha256(Path(baseline['checkpoint'])) == baseline['model_sha256']
    references = read_json(ROOT / 'local-only/yoloe-deployment/assets/visual_references.json')
    val_ids = {oid for row in rows for oid in row['object_ids']}
    assert not val_ids.intersection(r['object_id'] for r in references)
    assert all(r['source_split'] == 'train' for r in references)
    digest = hashlib.sha256()
    prepared = []
    stale_label_metadata = []
    for row in rows:
        image_path, labels = root / row['image'], label_path(root, row)
        assert sha256(image_path) == row['image_sha256']
        if sha256(labels) != row['label_sha256']:
            stale_label_metadata.append(row['image'])
        digest.update(Path(row['image']).as_posix().encode())
        digest.update(image_path.read_bytes())
        digest.update(labels.read_bytes())
        frame = cv2.imread(str(image_path))
        h, w = frame.shape[:2]
        truth = []
        for cid, polygon in parse_labels(labels):
            mask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(mask, [np.rint(polygon * [w, h]).astype(np.int32)], 1)
            truth.append(dict(class_id=cid, mask=mask.astype(bool)))
        prepared.append((row, truth))
    assert digest.hexdigest() == frozen_data['split_sha256']['val'], 'Validation differs from YOLO11s frozen data'
    policy = recipe['evaluation']
    assert policy['confidence'] == .3 and policy['mask_iou'] == .5
    profiles = [
        dict(name='matched_yolo11_policy', half=False, conf=.3, iou=policy['nms_iou'],
             max_det=policy['max_det'], agnostic_nms=False),
        dict(name='current_deployment', half=True, conf=.3, iou=.5,
             max_det=30, agnostic_nms=True),
    ]
    assets = ROOT / 'local-only/yoloe-deployment/assets'
    tracked = [assets / 'yoloe-materials-visual.pt', assets / 'yoloe-materials-prompted.pt',
               assets / 'visual_references.json', assets / 'deployment.json',
               ROOT / 'pynq_km1_vision/config/yoloe_materials.json']
    before = {str(p.relative_to(ROOT)): sha256(p) for p in tracked}
    plan = dict(schema_version='yoloe-validation-comparison-v1',
        started_at=datetime.now(timezone.utc).isoformat(), profiles=profiles,
        prompt_mode='visual', validation_images=121, validation_physical_objects=sorted(val_ids),
        validation_content_sha256=digest.hexdigest(), manifest_sha256=sha256(root / 'split_manifest.json'),
        baseline_validation_content_match=True, stale_per_row_label_hashes=stale_label_metadata,
        baseline_report_sha256=sha256(baseline_path), baseline_model_sha256=baseline['model_sha256'],
        assets_before=before, runner_sha256=sha256(Path(__file__)),
        scoring_source_sha256=sha256(ROOT / 'pynq_km1_vision/src/train_recyclables_final.py'),
        class_mapping=dict(metal='can', plastic='plastic', paper='cardboard'),
        evaluation=dict(confidence=.3, mask_iou=.5, matching='confidence_ordered_same_class_one_to_one'),
        reference_validation_object_overlap=False, trained=False, test_evaluated=False,
        deployed=False, physical_motion=False,
        limitations=['Same-day fixed workcell and only four validation objects.',
                    'YOLO11s checkpoint was selected using this validation set; YOLOE prompts remain frozen.',
                    'Parallel training and monitoring make these timings unsuitable for speed ranking.',
                    'Matched class-aware NMS operates on 24 visual prompt IDs before material mapping.'])
    save_json(output / 'plan.json', plan)
    summaries = {}
    mapping = {'metal': 0, 'plastic': 1, 'paper': 2}
    for profile in profiles:
        folder = output / profile['name']
        folder.mkdir()
        detector = YoloeDetector(device='cuda:0', half=profile['half'])
        assert detector.model_ids['visual'] == before[str((assets / 'yoloe-materials-visual.pt').relative_to(ROOT))]
        model = detector.models['visual']
        totals, groups, details, times = new_counts(), defaultdict(new_counts), [], []
        with (folder / 'observations.jsonl').open('w', encoding='utf-8') as evidence:
            for index, (row, truth) in enumerate(prepared, 1):
                frame = cv2.imread(str(root / row['image']))
                detector.torch.cuda.synchronize()
                start = time.perf_counter()
                raw = model.predict(frame, device='cuda:0', imgsz=640, retina_masks=True,
                    verbose=False, save=False, **{k:v for k,v in profile.items() if k != 'name'})[0]
                instances = convert_instances(raw, detector.configs['visual'], frame.shape[:2])
                detector.torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
                assert str(next(model.model.parameters()).device).startswith('cuda:')
                assert hashlib.sha256(frame.tobytes()).hexdigest() not in {
                    r['decoded_bgr_sha256'] for r in references}
                predictions = []
                for instance in instances:
                    runs = instance['mask_rle']
                    flat = np.repeat(np.arange(len(runs), dtype=np.uint32) % 2, runs)
                    mask = flat.reshape(frame.shape[:2]).astype(bool)
                    assert int(mask.sum()) == instance['area_px']
                    predictions.append(dict(class_id=mapping[instance['label']],
                        confidence=instance['confidence'], mask=mask))
                counts = match_instances(truth, predictions, .5)
                merge_counts(totals, counts)
                merge_counts(groups[row['scene_type']], counts)
                times.append(elapsed)
                details.append(dict(image=row['image'], scene_type=row['scene_type'],
                    object_ids=row['object_ids'], counts=counts,
                    predictions=[{k:v for k,v in p.items() if k != 'mask'} for p in predictions]))
                evidence.write(json.dumps(dict(image=row['image'], instances=instances,
                    inference_ms=elapsed), ensure_ascii=False) + '\n')
                if index in (1, 26, 50, 74, 98, 110):
                    cv2.imwrite(str(folder / f'overlay-{index:03d}.jpg'), annotate(frame, dict(instances=instances)))
                if index % 10 == 0 or index == len(rows):
                    progress = dict(profile=profile['name'], completed=index, total=len(rows),
                                    updated_at=datetime.now(timezone.utc).isoformat())
                    save_json(output / 'progress.json', progress)
                    print(json.dumps(progress), flush=True)
        summary = dict(profile=profile, execution=detector.execution,
            model_sha256=detector.model_ids['visual'],
            overall=describe_counts(totals, recipe['class_names']),
            scenes={k:describe_counts(v, recipe['class_names']) for k,v in groups.items()}, images=details,
            inference_ms=dict(median=float(np.median(times)), min=min(times), max=max(times),
                              note='Includes first cold call; concurrent GPU workloads; not a speed comparison.'))
        save_json(folder / 'result.json', summary)
        summaries[profile['name']] = {k:v for k,v in summary.items() if k != 'images'}
        del detector, model, raw
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    after = {str(p.relative_to(ROOT)): sha256(p) for p in tracked}
    assert before == after
    report = dict(plan, completed_at=datetime.now(timezone.utc).isoformat(),
        assets_unchanged=True, yoloe=summaries,
        yolo11s=dict(model_sha256=baseline['model_sha256'],
            overall=baseline['fixed']['overall'], scenes=baseline['fixed']['scenes']))
    save_json(output / 'comparison.json', report)
    print(json.dumps({k:v['overall'] for k,v in summaries.items()}), flush=True)


if __name__ == '__main__':
    main()
