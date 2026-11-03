"""Fixed-threshold comparison on a reused validation set with at most one target per image."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .train_segment import check_dataset


def summarize_frame(target, detections):
    counts = Counter(images=1, detections=len(detections))
    if target is None:
        counts.update(empty_frames=1, empty_frame_false_detections=len(detections))
        return counts
    top = max(detections, key=lambda d: d['confidence']) if detections else None
    matched = any(d['class_id'] == target and d['mask_iou'] >= 0.5 for d in detections)
    top_matched = bool(top and top['class_id'] == target and top['mask_iou'] >= 0.5)
    counts.update(top_class_and_mask_iou_ge_0_5=int(top_matched),
                  correct_class_and_mask_iou_ge_0_5=int(matched),
                  additional_or_unmatched_detections=len(detections) - int(matched))
    counts[f'class_{target}_images'] = 1
    counts[f'class_{target}_success'] = int(matched)
    counts[f'class_{target}_top_success'] = int(top_matched)
    return counts


def evaluate(dataset, weights, output, device='0', native_backend=False):
    dataset, weights, output = Path(dataset).resolve(), Path(weights).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Output exists; preserve previous comparisons')
    if not weights.is_file():
        raise ValueError('Use an existing local checkpoint')
    report = check_dataset(dataset)
    manifest = json.loads((dataset / 'split_manifest.json').read_text(encoding='utf-8-sig'))
    rows = sorted((r for r in manifest['images'] if r['split'] == 'val'), key=lambda r: r['image'])
    prepared, validation_digest = [], hashlib.sha256()
    for row in rows:
        image = dataset / row['image']
        relative = Path(row['image'])
        label = dataset / 'labels/val' / Path(*relative.parts[2:]).with_suffix('.txt')
        text = label.read_text(encoding='utf-8-sig')
        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) > 1:
            raise ValueError('Pilot comparison requires at most one labeled target per validation image')
        prepared.append((row, image, lines))
        validation_digest.update(row['image'].encode())
        validation_digest.update(image.read_bytes())
        validation_digest.update(text.encode())
    import torch
    from ultralytics import YOLO
    if native_backend:
        torch.backends.cudnn.enabled = False
    model = YOLO(str(weights))
    if model.task != 'segment' or list(model.names.values()) != report['labels']:
        raise ValueError('Checkpoint task or class names mismatch')
    totals, details = Counter(), []
    for row, image, lines in prepared:
        h, w = cv2.imread(str(image)).shape[:2]
        truth, target = np.zeros((h, w), np.uint8), None
        if lines:
            values = [float(v) for v in lines[0].split()]
            target = int(values[0])
            points = np.rint(np.array(values[1:]).reshape(-1, 2) * [w, h]).astype(np.int32)
            cv2.fillPoly(truth, [points], 1)
        result = model.predict(str(image), device=device, imgsz=640, conf=0.25, iou=0.7,
                               max_det=20, retina_masks=True, half=False, verbose=False)[0]
        detections = []
        for i, box in enumerate(result.boxes):
            mask = np.zeros_like(truth)
            if result.masks is not None:
                points = np.rint(result.masks.xy[i]).astype(np.int32)
                if len(points) >= 3:
                    cv2.fillPoly(mask, [points], 1)
            union = np.logical_or(mask, truth).sum()
            overlap = float(np.logical_and(mask, truth).sum() / union) if union else 0.0
            detections.append({'class_id': int(box.cls.item()),
                               'confidence': float(box.conf.item()), 'mask_iou': overlap})
        counts = summarize_frame(target, detections)
        totals.update(counts)
        details.append({'image': row['image'], 'target': target, 'summary': dict(counts),
                        'detections': detections})
    result = {'checkpoint': str(weights), 'sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
              'labels': report['labels'], 'confidence_threshold': 0.25, 'mask_iou_threshold': 0.5,
              'validation_sha256': validation_digest.hexdigest(), 'summary': dict(totals), 'images': details,
              'scope': 'Reused single-target validation for checkpoint comparison; not mAP, independent test or physical acceptance.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='0')
    p.add_argument('--native-backend', action='store_true')
    a = p.parse_args()
    print(json.dumps(evaluate(a.dataset, a.weights, a.output, a.device, a.native_backend)['summary']))


if __name__ == '__main__':
    main()
