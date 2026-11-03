"""Complete three-material training, frozen validation gates and held-out evaluation.

No hardware, camera, serial, deployment or automatic motion entry points.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from .train_segment import check_dataset

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_RECIPE = PROJECT / 'config/recyclables_final_training.json'
SCENES = ('source', 'bin', 'empty', 'wrong_bin', 'multi')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def label_path(root, row):
    relative = Path(row['image'])
    return root / 'labels' / row['split'] / Path(*relative.parts[2:]).with_suffix('.txt')


def parse_labels(path):
    items = []
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        if line.strip():
            values = [float(v) for v in line.split()]
            items.append((int(values[0]), np.asarray(values[1:], dtype=float).reshape(-1, 2)))
    return items


def check_final_dataset(root, recipe):
    """Audit real objects, full capture sessions, scene coverage and reviewed masks."""
    root = Path(root).resolve()
    basic = check_dataset(root)
    document = read_json(root / 'split_manifest.json')
    names = recipe['class_names']
    need(document.get('schema_version') == 'recyclable-final-dataset-v1',
         'Need final dataset metadata: schema_version=recyclable-final-dataset-v1')
    need(document.get('_template_only') is not True, 'Replace the empty manifest template with actual reviewed data')
    need(basic['labels'] == names, 'Final class order must be can, plastic, cardboard')
    objects = document.get('object_classes')
    need(isinstance(objects, dict) and objects, 'object_classes must map actual physical IDs to material names')
    need(all(isinstance(k, str) and k.strip() and v in names for k, v in objects.items()),
         'Invalid object_classes')
    owners, sessions, hashes, poses = {}, {}, set(), set()
    scene_counts, scene_classes, views, split_hashes = Counter(), Counter(), Counter(), {}
    for split in ('train', 'val', 'test'):
        split_hashes[split] = hashlib.sha256()
    for row in sorted(document['images'], key=lambda r: r['image']):
        split, scene = row['split'], row.get('scene_type')
        need(scene in SCENES, 'scene_type must be source, bin, empty, wrong_bin or multi')
        need(row.get('annotation_status') == 'reviewed', 'Every image needs reviewed polygons or reviewed empty labels')
        session = row.get('capture_session_id')
        need(isinstance(session, str) and session.strip(), 'Need actual capture_session_id, not a per-object segment')
        need(session not in sessions or sessions[session] == split, 'Capture session crosses dataset splits: ' + session)
        sessions[session] = split
        workcell = row.get('workcell_id')
        need(isinstance(workcell, str) and workcell.strip(), 'Missing workcell_id')
        relative = Path(row['image'])
        image_path = root / relative
        image_hash = sha256(image_path)
        need(image_hash not in hashes, 'Duplicate image bytes, including duplicates within one split')
        hashes.add(image_hash)
        image = cv2.imread(str(image_path))
        need(list(image.shape[:2][::-1]) == recipe['image_size'], 'Full-frame size mismatch: ' + row['image'])
        labels = parse_labels(label_path(root, row))
        ids = row['object_ids']
        need(len(ids) == len(set(ids)) and all(x in objects for x in ids), 'Unknown or duplicate physical object IDs')
        expected = Counter(names.index(objects[x]) for x in ids)
        need(Counter(cid for cid, _ in labels) == expected, 'Physical IDs and polygon classes/counts disagree: ' + row['image'])
        for object_id in ids:
            need(object_id not in owners or owners[object_id] == split, 'Physical object crosses splits: ' + object_id)
            owners[object_id] = split
        if scene == 'empty':
            need(not ids and not labels, 'Empty image contains annotated target objects')
        elif scene == 'multi':
            need(2 <= len(ids) <= 3, 'multi scenes must contain 2 or 3 distinct target objects')
        else:
            need(len(ids) == 1, 'Single-object scene must have exactly one physical object')
            material = objects[ids[0]]
            if scene in ('bin', 'wrong_bin'):
                need(row.get('bin_material') in names, 'bin/wrong_bin requires actual bin_material')
                need((row['bin_material'] == material) == (scene == 'bin'), 'Material/bin scene type disagrees')
            if scene in ('source', 'bin'):
                pose = row.get('pose_id')
                need(isinstance(pose, str) and pose.strip(), 'source/bin requires a distinct pose_id')
                pose_key = (ids[0], scene, pose)
                need(pose_key not in poses, 'Repeated pose_id for the same object and scene')
                poses.add(pose_key)
                views[(ids[0], scene)] += 1
        scene_counts[(split, scene)] += 1
        for object_id in ids:
            scene_classes[(split, scene, objects[object_id])] += 1
        digest = split_hashes[split]
        digest.update(relative.as_posix().encode())
        digest.update(image_path.read_bytes())
        digest.update(label_path(root, row).read_bytes())
    need(set(owners) == set(objects), 'object_classes must contain exactly the photographed physical objects')
    object_counts = {}
    for split, minimum in recipe['dataset_minimums'].items():
        for material in names:
            members = [x for x in objects if owners[x] == split and objects[x] == material]
            object_counts[split + ':' + material] = len(members)
            required = minimum['objects_per_class']
            if isinstance(required, dict):
                need(set(required) == set(names), 'objects_per_class must specify all three materials')
                required = required[material]
            need(type(required) is int and required > 0, 'Object minimum must be a positive integer')
            need(len(members) >= required, f'Need {required} different {material} objects in {split}')
            for object_id in members:
                for scene in ('source', 'bin'):
                    target = minimum[scene + '_views_per_object']
                    need(views[(object_id, scene)] >= target, f'Need {target} distinct {scene} views for {object_id}')
        for scene in ('empty', 'wrong_bin', 'multi'):
            need(scene_counts[(split, scene)] >= minimum[scene + '_images'], f'Not enough {scene} images in {split}')
        for material in names:
            need(scene_classes[(split, 'wrong_bin', material)] >= minimum['wrong_bin_images'] // 3,
                 f'Not enough wrong-bin {material} images in {split}')
            need(scene_classes[(split, 'multi', material)] >= minimum['multi_images'] * 2 // 3,
                 f'Not enough multi-object {material} instances in {split}')
    basic.update(object_counts=object_counts,
                 scene_counts={split: {scene: scene_counts[(split, scene)] for scene in SCENES}
                               for split in ('train', 'val', 'test')},
                 split_sha256={k: v.hexdigest() for k, v in split_hashes.items()},
                 manifest_sha256=sha256(root / 'split_manifest.json'),
                 metadata_is_producer_declaration=True)
    return basic


def new_counts():
    return {'tp': [0, 0, 0], 'fp': [0, 0, 0], 'fn': [0, 0, 0],
            'images': 0, 'empty_images': 0, 'empty_false_positive_images': 0}


def match_instances(truth, predictions, iou_threshold):
    """Confidence-ordered one-to-one matches; wrong class and duplicates count as FP."""
    counts, used = new_counts(), set()
    counts['images'] = 1
    counts['empty_images'] = int(not truth)
    counts['empty_false_positive_images'] = int(not truth and bool(predictions))
    for prediction in sorted(predictions, key=lambda d: d['confidence'], reverse=True):
        cid = prediction['class_id']
        candidates = []
        for index, target in enumerate(truth):
            if index in used or target['class_id'] != cid:
                continue
            union = np.logical_or(prediction['mask'], target['mask']).sum()
            overlap = float(np.logical_and(prediction['mask'], target['mask']).sum() / union) if union else 0.0
            candidates.append((overlap, index))
        score, best = max(candidates, default=(0.0, -1))
        if score >= iou_threshold and best >= 0:
            counts['tp'][cid] += 1
            used.add(best)
        else:
            counts['fp'][cid] += 1
    for index, target in enumerate(truth):
        if index not in used:
            counts['fn'][target['class_id']] += 1
    return counts


def merge_counts(total, item):
    for key in ('tp', 'fp', 'fn'):
        total[key] = [a + b for a, b in zip(total[key], item[key])]
    for key in ('images', 'empty_images', 'empty_false_positive_images'):
        total[key] += item[key]


def describe_counts(counts, names):
    result = {k: v for k, v in counts.items() if k not in ('tp', 'fp', 'fn')}
    result['per_class'] = {}
    for cid, name in enumerate(names):
        tp, fp, fn = (counts[k][cid] for k in ('tp', 'fp', 'fn'))
        result['per_class'][name] = dict(tp=tp, fp=fp, fn=fn,
            precision=tp/(tp+fp) if tp+fp else None,
            recall=tp/(tp+fn) if tp+fn else None)
    return result


def fixed_threshold_evaluation(model, root, split, recipe, device):
    """Full multi-instance mask evaluation at the frozen runtime confidence."""
    document = read_json(root / 'split_manifest.json')
    policy = recipe['evaluation']
    totals, groups, details = new_counts(), defaultdict(new_counts), []
    for row in sorted((r for r in document['images'] if r['split'] == split), key=lambda r: r['image']):
        image = cv2.imread(str(root / row['image']))
        h, w = image.shape[:2]
        truth = []
        for cid, polygon in parse_labels(label_path(root, row)):
            mask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(mask, [np.rint(polygon * [w, h]).astype(np.int32)], 1)
            truth.append(dict(class_id=cid, mask=mask.astype(bool)))
        prediction = model.predict(image, device=device, imgsz=recipe['training']['imgsz'],
            conf=policy['confidence'], iou=policy['nms_iou'], max_det=policy['max_det'],
            retina_masks=True, half=False, verbose=False)[0]
        predictions = []
        for i, box in enumerate(prediction.boxes):
            need(prediction.masks is not None, 'Segmentation output has no masks')
            mask = prediction.masks.data[i].cpu().numpy() > .5
            need(mask.shape == (h, w), 'Prediction mask is not in original image coordinates')
            cid = int(box.cls.item())
            need(0 <= cid < 3, 'Unexpected predicted material ID')
            predictions.append(dict(class_id=cid, confidence=float(box.conf.item()), mask=mask))
        counts = match_instances(truth, predictions, policy['mask_iou'])
        merge_counts(totals, counts)
        merge_counts(groups[row['scene_type']], counts)
        details.append(dict(image=row['image'], scene_type=row['scene_type'], counts=counts,
                            predictions=[{k: v for k, v in p.items() if k != 'mask'} for p in predictions]))
    return dict(split=split, confidence=policy['confidence'], mask_iou=policy['mask_iou'],
                overall=describe_counts(totals, recipe['class_names']),
                scenes={k: describe_counts(v, recipe['class_names']) for k, v in groups.items()},
                images=details)


def quality_gate(metrics, recipe):
    policy, reasons = recipe['evaluation'], []
    for group, summary in [('overall', metrics['overall'])] + [
            (scene, metrics['scenes'].get(scene)) for scene in policy['required_scene_groups']]:
        if summary is None:
            reasons.append(group + ':missing_scene')
            continue
        for material, result in summary['per_class'].items():
            # Overall and source/bin must cover every material. Other scenes
            # are checked when that material has truth or predictions.
            if group not in ('overall', 'source', 'bin') and not (result['tp']+result['fp']+result['fn']):
                continue
            for key in ('precision', 'recall'):
                value = result[key]
                if value is None or value < policy[key + '_min']:
                    reasons.append(f'{group}:{material}:{key}_below_target')
    empty = metrics['scenes'].get('empty')
    if empty is None or not empty['empty_images']:
        reasons.append('empty:missing_scene')
    elif empty['empty_false_positive_images'] > policy['empty_false_positive_images_max']:
        reasons.append('empty:false_positive_images')
    return dict(passed=not reasons, reasons=reasons, thresholds=policy)


def ap_evaluation(model, data_path, split, recipe, device, output):
    result = model.val(data=str(data_path), split=split, device=device,
        imgsz=recipe['training']['imgsz'], batch=recipe['training']['batch'], workers=0,
        half=False, conf=.001, iou=recipe['evaluation']['nms_iou'],
        project=str(output), name='ap-' + split, plots=True, save_json=False)
    per_class = {}
    for i, cid in enumerate(result.seg.ap_class_index):
        per_class[recipe['class_names'][int(cid)]] = dict(
            mask_map50=float(result.seg.ap50[i]), mask_map50_95=float(result.seg.ap[i]))
    return dict(aggregate={k: float(v) for k, v in result.results_dict.items()}, per_class=per_class,
                note='AP uses a confidence sweep; the separate fixed-threshold report controls quality gates.')


def evaluate_split(model, root, data_path, split, recipe, device, output):
    fixed = fixed_threshold_evaluation(model, root, split, recipe, device)
    save_json(output / ('fixed-' + split + '.json'), fixed)
    ap = ap_evaluation(model, data_path, split, recipe, device, output)
    save_json(output / ('ap-' + split + '.json'), ap)
    return dict(fixed=fixed, ap=ap, gate=quality_gate(fixed, recipe))


def evaluate_and_package(model, best, root, data_path, recipe, device, output, dataset_report):
    """Freeze model before test. Failed validation never evaluates test."""
    best_hash = sha256(best)
    result = dict(model_sha256=best_hash, dataset_sha256=dataset_report['dataset_sha256'],
        split_sha256=dataset_report['split_sha256'], test_evaluated=False,
        automatic_motion_allowed=False, physical_success_verified=False, deployed=False,
        test_previously_reserved=(Path(root) / 'test-use.json').exists())
    result['validation'] = evaluate_split(model, root, data_path, 'val', recipe, device, output)
    if not result['validation']['gate']['passed']:
        result.update(status='validation_failed', next_action='Fix using training/validation data; this run did not evaluate held-out data.')
        save_json(output / 'final-report.json', result)
        return result
    need(check_final_dataset(root, recipe)['dataset_sha256'] == dataset_report['dataset_sha256'],
         'Dataset changed during training/evaluation')
    need(sha256(best) == best_hash, 'Selected model changed before held-out evaluation')
    # Reserve at dataset scope as well as run scope: a new output directory must
    # not silently turn an already consulted test split into a fresh holdout.
    reservation = dict(model_sha256=best_hash, test_sha256=dataset_report['split_sha256']['test'])
    try:
        with (Path(root) / 'test-use.json').open('x', encoding='utf-8') as handle:
            json.dump(reservation, handle)
    except FileExistsError as error:
        result.update(status='heldout_previously_used_do_not_repeat',
                      next_action='Use training/validation for development and prepare new independent test data.')
        save_json(output / 'final-report.json', result)
        raise FileExistsError('Held-out data already reserved or scored; a new run requires new independent test data.') from error
    # Both exclusive markers persist even on failure; no automatic test replay.
    with (output / 'test-reservation.json').open('x', encoding='utf-8') as handle:
        json.dump(reservation, handle)
    result['test_evaluated'] = True
    try:
        result['test'] = evaluate_split(model, root, data_path, 'test', recipe, device, output)
    except Exception:
        result['status'] = 'test_interrupted_do_not_retry_automatically'
        save_json(output / 'final-report.json', result)
        raise
    result['status'] = 'quality_targets_met' if result['test']['gate']['passed'] else 'test_failed'
    if result['status'] == 'quality_targets_met':
        need(sha256(best) == best_hash and check_final_dataset(root, recipe)['dataset_sha256'] ==
             dataset_report['dataset_sha256'], 'Model or dataset changed before packaging')
        package = output / 'package'
        package.mkdir(exist_ok=False)
        shutil.copy2(best, package / 'model.pt')
        save_json(package / 'inference.json', dict(class_names=recipe['class_names'],
            confidence=recipe['evaluation']['confidence'], imgsz=recipe['training']['imgsz'],
            image_size=recipe['image_size'], retina_masks=True, model_sha256=best_hash,
            automatic_motion_allowed=False, physical_success_verified=False))
        save_json(package / 'training-recipe.json', recipe)
    save_json(output / 'final-report.json', result)
    return result


def run_training(root, weights, output, device, recipe):
    root, weights, output = Path(root).resolve(), Path(weights).resolve(), Path(output).resolve()
    dataset_report = check_final_dataset(root, recipe)
    need(weights.is_file(), 'Local segmentation checkpoint missing; no automatic model download')
    need(not output.exists(), 'Output already exists; use a new run directory')
    import torch
    import ultralytics
    from ultralytics import YOLO
    torch.backends.cudnn.enabled = False
    torch.set_num_threads(4)
    if device != 'cpu':
        need(device.isdigit() and torch.cuda.is_available() and int(device) < torch.cuda.device_count(),
             'Requested GPU unavailable; no silent CPU fallback')
    model = YOLO(str(weights))
    need(model.task == 'segment', 'Use a local YOLO segmentation checkpoint')
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / 'dataset-check.json', dataset_report)
    save_json(output / 'training-recipe.json', recipe)
    shutil.copy2(root / 'split_manifest.json', output / 'split_manifest.frozen.json')
    provenance = dict(source_weights_sha256=sha256(weights), torch=torch.__version__,
        hip=torch.version.hip, ultralytics=ultralytics.__version__, requested_device=device,
        entry_sha256=sha256(__file__), recipe_sha256=sha256(DEFAULT_RECIPE),
        miopen_cudnn_enabled=torch.backends.cudnn.enabled,
        started_at=datetime.now(timezone.utc).isoformat(), physical_motion=False, deployed=False)
    save_json(output / 'provenance.json', provenance)
    training_data = dict(path=str(root), train='images/train', val='images/val',
                         names=dict(enumerate(recipe['class_names'])))
    data_path = output / 'dataset-train-val.yaml'
    save_json(data_path, training_data)  # JSON is a YAML subset.
    try:
        model.train(data=str(data_path), device=device, project=str(output), name='training',
                    exist_ok=False, **recipe['training'])
        need(check_final_dataset(root, recipe)['dataset_sha256'] == dataset_report['dataset_sha256'],
             'Dataset changed while training')
        best = Path(model.trainer.best)
        trained = YOLO(str(best))
        need(trained.task == 'segment' and list(trained.names.values()) == recipe['class_names'],
             'Saved model task or class order mismatch')
        provenance.update(epochs_completed=model.trainer.epoch + 1,
                          actual_device=str(model.trainer.device), best_sha256=sha256(best),
                          completed_at=datetime.now(timezone.utc).isoformat())
        save_json(output / 'provenance.json', provenance)
        evaluation_data = dict(training_data, test='images/test')
        evaluation_path = output / 'dataset-evaluation.yaml'
        save_json(evaluation_path, evaluation_data)
        return evaluate_and_package(trained, best, root, evaluation_path, recipe, device,
                                    output, dataset_report)
    except Exception as exc:
        save_json(output / 'run-error.json', dict(error_type=type(exc).__name__, error=str(exc),
                                                automatic_retry=False, deployed=False))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--weights', type=Path, default=PROJECT / 'yolo11n-seg.pt')
    parser.add_argument('--output', type=Path, default=PROJECT.parent / 'local-only/recyclables-final-runs' /
                        datetime.now(timezone.utc).strftime('final-%Y%m%d-%H%M%S-%f'))
    parser.add_argument('--device', default='0')
    parser.add_argument('--check-only', action='store_true', help='Validate complete final data without model loading')
    args = parser.parse_args(argv)
    recipe = read_json(DEFAULT_RECIPE)
    try:
        if args.check_only:
            result = check_final_dataset(args.dataset, recipe)
        else:
            result = run_training(args.dataset, args.weights, args.output, args.device, recipe)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, 'Final training stopped: ' + str(exc) + '\n')
    print(json.dumps(result if args.check_only else dict(status=result['status'], output=str(args.output)),
                     ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if args.check_only or result['status'] == 'quality_targets_met' else 3


if __name__ == '__main__':
    raise SystemExit(main())
