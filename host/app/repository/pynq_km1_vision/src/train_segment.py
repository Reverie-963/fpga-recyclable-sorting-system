"""Train the three recyclable classes with instance polygons in a separate run directory."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import math

LABELS = ['can', 'plastic_bottle', 'cardboard']
MATERIAL_LABELS = ['can', 'plastic', 'cardboard']
IMAGE_EXTENSIONS = {'.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png', '.tif', '.tiff', '.webp', '.pfm', '.heic'}


def check_dataset(root):
    """Require explicit labels and grouped splits before downloading or training anything."""
    root = Path(root).resolve()
    manifest_path = root / 'split_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    labels = manifest.get('class_names', LABELS)
    if labels not in (LABELS, MATERIAL_LABELS):
        raise ValueError('Unsupported class_names or class order')
    rows = manifest['images']
    if not isinstance(rows, list) or not rows:
        raise ValueError('No annotated images: collect and label scenes first')
    ownership = {}
    files = set()
    counts = Counter()
    classes = {split: Counter() for split in ('train', 'val', 'test')}
    digest = hashlib.sha256(manifest_path.read_bytes())
    import cv2
    for row in rows:
        split = row['split']
        if split not in classes:
            raise ValueError('Invalid split')
        relative = Path(row['image'])
        if (relative.is_absolute() or relative.parts[:2] != ('images', split)
                or '..' in relative.parts or relative.suffix.lower() not in IMAGE_EXTENSIONS):
            raise ValueError('Images must be under images/<split>/')
        image = (root / relative).resolve()
        if not image.is_relative_to(root) or image in files:
            raise ValueError('Duplicate or outside image path')
        files.add(image)
        raw = image.read_bytes()
        decoded = cv2.imread(str(image))
        if decoded is None:
            raise ValueError('Unreadable image: ' + str(relative))
        label = root / 'labels' / split / Path(*relative.parts[2:]).with_suffix('.txt')
        text = label.read_text(encoding='utf-8-sig')  # Empty file explicitly labels a background.
        ids = row['object_ids']
        session = row['session_id']
        if not isinstance(ids, list) or any(not isinstance(x, str) or not x.strip() for x in ids):
            raise ValueError('Invalid physical object_ids')
        if not isinstance(session, str) or not session.strip() or (text.strip() and not ids):
            raise ValueError('Object and session identifiers are required')
        keys = [('image_hash', hashlib.sha256(raw).hexdigest()), ('session', session)]
        keys += [('object', x) for x in ids]
        for key in keys:
            if key in ownership and ownership[key] != split:
                raise ValueError('Train/validation/test leakage: ' + key[0] + ' ' + key[1])
            ownership[key] = split
        for line in text.splitlines():
            values = [float(x) for x in line.split()]
            if len(values) < 7 or len(values) % 2 != 1 or not all(math.isfinite(x) for x in values):
                raise ValueError('Need class and >=3 polygon vertices, not detection boxes')
            cid, coords = values[0], values[1:]
            if cid not in (0, 1, 2) or not all(0 <= x <= 1 for x in coords):
                raise ValueError('Invalid class or normalized coordinates')
            points = list(zip(coords[::2], coords[1::2]))
            area2 = sum(x*y1-x1*y for (x, y), (x1, y1) in zip(points, points[1:] + points[:1]))
            if abs(area2) <= 1e-10:
                raise ValueError('Degenerate polygon')
            classes[split][int(cid)] += 1
        digest.update(relative.as_posix().encode())
        digest.update(raw)
        digest.update(text.encode())
        counts[split] += 1
    for split in ('train', 'val'):
        if not counts[split] or set(classes[split]) != {0, 1, 2}:
            raise ValueError('Need all three annotated classes in ' + split)
    for split in classes:
        disk = {p.resolve() for p in (root / 'images' / split).rglob('*')
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS}
        expected = {p for p in files if p.is_relative_to(root / 'images' / split)}
        if disk != expected:
            raise ValueError('Manifest does not cover exactly the images in ' + split)
    return {'images': dict(counts), 'instances': {k: dict(v) for k, v in classes.items()},
            'dataset_sha256': digest.hexdigest(), 'labels': labels}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--weights', default='yolo11n-seg.pt')
    parser.add_argument('--device', default='0')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--lr0', type=float)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--augmentation', choices=['default', 'conservative'], default='default')
    parser.add_argument('--warmup-epochs', type=float, default=3.0)
    parser.add_argument('--warmup-bias-lr', type=float, default=0.1)
    parser.add_argument('--project', default='runs/segment')
    parser.add_argument('--name', default='recyclables-' + datetime.now().strftime('%Y%m%d-%H%M%S'))
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--native-backend', action='store_true',
                        help='Use native PyTorch GPU kernels when MIOpen training compilation fails')
    parser.add_argument('--evaluate-test', action='store_true', help='Only after model and thresholds are frozen')
    args = parser.parse_args()
    if min(args.epochs, args.batch, args.imgsz) < 1:
        parser.error('epochs, batch and imgsz must be positive')
    if args.patience < 0 or (args.lr0 is not None and (not math.isfinite(args.lr0) or args.lr0 <= 0)):
        parser.error('patience must be nonnegative and lr0 must be finite and positive')
    if any(not math.isfinite(v) or v < 0 for v in (args.warmup_epochs, args.warmup_bias_lr)):
        parser.error('warmup settings must be finite and nonnegative')
    try:
        report = check_dataset(args.dataset)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error('Dataset not ready: ' + str(error))
    print(json.dumps(report, indent=2), flush=True)
    if args.check_only:
        return
    if args.evaluate_test and set(report['instances']['test']) != {0, 1, 2}:
        parser.error('Independent test data must include all three classes')
    from ultralytics import YOLO
    import torch
    if args.native_backend:
        torch.backends.cudnn.enabled = False
    model = YOLO(args.weights)
    if model.task != 'segment':
        parser.error('Use segmentation weights such as yolo11n-seg.pt')
    run = Path(args.project).resolve() / args.name
    run.mkdir(parents=True, exist_ok=False)
    data = {'path': str(args.dataset.resolve()), 'train': 'images/train', 'val': 'images/val',
            'names': dict(enumerate(report['labels']))}
    if report['images'].get('test'):
        data['test'] = 'images/test'
    # JSON is valid YAML; keeping it explicit avoids machine paths in shared sources.
    data_path = run / 'dataset.yaml'
    data_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
    report.update(task='segment', source_weights=args.weights, torch=torch.__version__, hip=torch.version.hip,
                  requested_device=args.device, precision='FP32', dataset=str(args.dataset.resolve()),
                  miopen_cudnn_enabled=torch.backends.cudnn.enabled,
                  started_at=datetime.now().astimezone().isoformat())
    params = dict(epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
                device=args.device, amp=False, workers=0, seed=7, project=str(run.parent),
                name=run.name, exist_ok=True, patience=args.patience,
                warmup_epochs=args.warmup_epochs, warmup_bias_lr=args.warmup_bias_lr)
    if args.lr0 is not None:
        params.update(optimizer='AdamW', lr0=args.lr0)
    if args.augmentation == 'conservative':
        params.update(mosaic=0.0, close_mosaic=0, mixup=0.0, copy_paste=0.0,
                      degrees=15.0, translate=0.1, scale=0.2, fliplr=0.5)
    report['training_parameters'] = params
    source = Path(args.weights)
    if source.is_file():
        report['source_weights_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    (run / 'dataset-check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    model.train(data=str(data_path), **params)
    best = Path(model.trainer.best)
    trained = YOLO(str(best))
    if trained.task != 'segment' or list(trained.names.values()) != report['labels']:
        raise ValueError('Saved model task or labels mismatch')
    report.update(best=str(best), sha256=hashlib.sha256(best.read_bytes()).hexdigest(),
                  actual_device=str(model.trainer.device), test_evaluated=False,
                  epochs_completed=model.trainer.epoch + 1,
                  completed_at=datetime.now().astimezone().isoformat())
    if args.evaluate_test:
        result = trained.val(data=str(data_path), split='test', device=args.device,
                             imgsz=args.imgsz, batch=1, project=str(run), name='heldout-test')
        report.update(test_evaluated=True, test_metrics={k: float(v) for k, v in result.results_dict.items()})
    (run / 'model.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
