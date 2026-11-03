"""Sequential YOLOv8s/YOLO11s training on a frozen public segmentation dataset.

Run from the project directory with its existing ROCm Python. No deployment or
hardware access. Each worker has its own logs, settings, and checkpoints.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
BASES = {
    'yolov8s-seg': '0bac0770b55e5eb5b76a61bc535673288dcec36c2bc0cd25ee0d584c632f3413',
    'yolo11s-seg': '1caa81c0195412efa411b632bcfb8c184939dddb6ae41f6a80c41b211ff257c3',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    for attempt in range(10):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.1)


def now():
    return datetime.now(timezone.utc).isoformat()


def audit(dataset):
    manifest = read(dataset / 'manifest.json')
    names = manifest['classes']
    if len(names) != 26 or names != [v['official_label'] for v in read(dataset / 'taxonomy.json')['classes']]:
        raise ValueError('Expected identical 26-class taxonomy')
    split_hashes = {'train': set(), 'val': set()}
    counts = Counter()
    for row in manifest['images']:
        split = row['split']
        if split not in split_hashes:
            raise ValueError('Only frozen public train/val are permitted')
        image = dataset / row['image']
        if not image.resolve().is_relative_to(dataset) or Path(row['image']).parts[:2] != ('images', split):
            raise ValueError('Invalid image path')
        label = dataset / 'labels' / split / image.with_suffix('.txt').name
        if sha(image) != row['image_sha256'] or sha(label) != row['label_sha256']:
            raise ValueError('Dataset changed: ' + row['image'])
        split_hashes[split].add(row['image_sha256'])
        counts[split] += 1
    if split_hashes['train'] & split_hashes['val']:
        raise ValueError('Identical train/val image bytes')
    if counts != {'train': 2886, 'val': 389}:
        raise ValueError('Unexpected frozen data counts: ' + repr(counts))
    return dict(counts=counts, classes=names, metadata_sha256={name: sha(dataset / name)
        for name in ('manifest.json', 'taxonomy.json', 'data.yaml')}, all_image_label_hashes_verified=True)


def setup_runtime(out):
    os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
        YOLO_AUTOINSTALL='false', YOLO_CONFIG_DIR=str(out / 'settings'), PYTHONUTF8='1')
    sys.path.insert(0, str(ROOT / 'local-only/yoloe11m-26m-comparison/runtime'))
    sys.path.insert(0, str(ROOT / 'pynq_km1_vision'))
    import torch
    import ultralytics
    if ultralytics.__version__ != '8.4.193':
        raise ValueError('Use the existing private Ultralytics 8.4.193 runtime')
    if not torch.cuda.is_available() or not torch.version.hip:
        raise ValueError('Existing ROCm GPU is required')
    torch.set_num_threads(4)
    torch.backends.cudnn.enabled = False
    from src.yoloe_training_runtime import install_cpu_target_assignment
    install_cpu_target_assignment()
    return torch, ultralytics


def train(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.smoke and (out / 'complete.json').exists():
        receipt = read(out / 'complete.json')
        if (receipt['model'] != args.model or not receipt['smoke_only'] or
            not receipt.get('backbone_updated') or receipt.get('optimizer_steps', 0) < 2 or
            receipt['base_sha256'] != sha(args.weights) or
            receipt['dataset_manifest_sha256'] != sha(args.dataset / 'manifest.json') or
            not read(out / 'actual-trainable.json')['amp'] or
            sha(receipt['best']) != receipt['best_sha256']):
            raise ValueError('Existing AMP smoke receipt does not match this run')
        print('Reused verified real AMP training smoke receipt for', args.model, flush=True)
        return
    torch, ultralytics = setup_runtime(out)
    from ultralytics import YOLO
    from ultralytics.models.yolo.segment import SegmentationTrainer
    import ultralytics.engine.trainer as trainer_module

    class MaskSelectionTrainer(SegmentationTrainer):
        def optimizer_step(self):
            nonlocal optimizer_steps
            previous_scale = float(self.scaler.get_scale())
            super().optimizer_step()
            # Fused optimizers can be called even on overflow; a post-hook
            # counts calls rather than actual updates. GradScaler lowers its
            # scale when it skips an update.
            if float(self.scaler.get_scale()) >= previous_scale:
                optimizer_steps += 1

        def validate(self):
            previous_best = self.best_fitness
            metrics, _ = super().validate()
            score = float(metrics['metrics/mAP50-95(M)'])
            if not math.isfinite(score):
                raise ValueError('Nonfinite validation score')
            self.best_fitness = score if previous_best is None else max(previous_best, score)
            return metrics, score

    weight = args.weights.resolve()
    if sha(weight) != BASES[args.model]:
        raise ValueError('Official pretrained weight hash mismatch')
    dataset = args.dataset.resolve()
    names = read(dataset / 'manifest.json')['classes']
    data = dataset / 'data.yaml'
    if args.smoke:
        small = out / 'smoke-data'
        (small / 'images/train').mkdir(parents=True, exist_ok=False)
        (small / 'labels/train').mkdir(parents=True)
        rows = [r for r in read(dataset / 'manifest.json')['images'] if r['split'] == 'train'][:2]
        for row in rows:
            image = Path(row['image'])
            shutil.copy2(dataset / image, small / 'images/train' / image.name)
            shutil.copy2(dataset / 'labels/train' / image.with_suffix('.txt').name,
                         small / 'labels/train' / image.with_suffix('.txt').name)
        data = small / 'data.yaml'
        data.write_text('path: ' + small.as_posix() + '\ntrain: images/train\nval: images/train\nnames: '
                        + json.dumps(names) + '\n', encoding='utf-8')
    model = YOLO(str(weight))
    if not args.smoke:
        # Both architectures completed a real AMP forward/backward/checkpoint
        # smoke run first; avoid the upstream unrelated YOLO26 download probe.
        receipt = read(out.parent / 'smoke' / args.model / 'complete.json')
        if not receipt['smoke_only'] or receipt['base_sha256'] != sha(weight):
            raise ValueError('Missing matching AMP integration receipt')
        if (not receipt.get('backbone_updated') or receipt.get('optimizer_steps', 0) < 2 or
            receipt['dataset_manifest_sha256'] != sha(dataset / 'manifest.json') or
            receipt['best_sha256'] != sha(receipt['best'])):
            raise ValueError('AMP integration update/dataset/checkpoint evidence does not match')
        if not read(out.parent / 'smoke' / args.model / 'actual-trainable.json')['amp']:
            raise ValueError('AMP integration did not run with AMP enabled')
    trainer_module.check_amp = lambda _: True
    # AMP may skip the first steps while its dynamic loss scale settles. A
    # single-batch check can therefore finish without any optimizer update.
    config = dict(data=str(data), epochs=12 if args.smoke else args.epochs,
        batch=2 if args.smoke else 4, imgsz=640, device='0', workers=0,
        amp=True, cache=False, optimizer='AdamW', lr0=.001, lrf=.1,
        weight_decay=.0005, patience=12 if args.smoke else 4,
        warmup_epochs=0 if args.smoke else 1, warmup_bias_lr=0.,
        nbs=2 if args.smoke else 64,
        seed=20261030, deterministic=True, mosaic=0., mixup=0., copy_paste=0.,
        degrees=0., translate=0., scale=0., fliplr=0., flipud=0.,
        hsv_h=0., hsv_s=0., hsv_v=0., close_mosaic=0,
        project=str(out), name='training', exist_ok=False, plots=False,
        save=True, save_period=1, val=True, freeze=0,
        conf=.001, iou=.7, max_det=100, rect=False)
    if (out / 'plan.json').exists():
        raise FileExistsError('Keep existing run; choose a new output')
    save(out / 'plan.json', dict(model=args.model, config=config, base_sha256=sha(weight),
        dataset_manifest_sha256=sha(dataset / 'manifest.json'), torch=torch.__version__,
        ultralytics=ultralytics.__version__, hip=torch.version.hip,
        gpu=torch.cuda.get_device_name(0), started_at=now(), smoke_only=args.smoke,
        checkpoint_selection='public validation mask mAP50-95 only',
        amp_check='Actual architecture AMP training integration; upstream unrelated inference probe bypassed',
        training_scope='full fine-tuning; YOLOE comparison uses classifier-only training',
        augmentation='fixed supplied views; no additional stochastic image augmentation', deployed=False))
    batch_count = 0
    first_parameter = None
    optimizer_steps = 0
    optimizer_audit = []
    backbone_updated = False
    def started(trainer):
        nonlocal first_parameter
        if list(trainer.data['names'].values()) != names:
            raise ValueError('Trainer taxonomy changed')
        first_parameter = next(trainer.model.parameters()).detach().cpu().clone()
        save(out / 'actual-trainable.json', dict(parameters=[n for n, p in trainer.model.named_parameters() if p.requires_grad],
            count=sum(p.numel() for p in trainer.model.parameters() if p.requires_grad), amp=bool(trainer.amp)))
    def batch_done(trainer):
        nonlocal batch_count, backbone_updated
        batch_count += 1
        if not torch.isfinite(trainer.loss).all():
            raise ValueError('Nonfinite training loss')
        if args.smoke or batch_count <= 10:
            backbone_updated = not torch.equal(first_parameter,
                next(trainer.model.parameters()).detach().cpu())
            optimizer_audit.append(dict(batch=batch_count, optimizer_steps=optimizer_steps,
                loss_scale=float(trainer.scaler.get_scale()), backbone_updated=backbone_updated))
            save(out / 'optimizer-audit.json', optimizer_audit)
        if args.smoke or batch_count <= 10 or batch_count % 50 == 0:
            save(out / 'progress.json', dict(epoch=trainer.epoch + 1, epochs=trainer.epochs,
                batches=batch_count, loss=float(trainer.loss.detach().sum()),
                optimizer_steps=optimizer_steps, loss_scale=float(trainer.scaler.get_scale()),
                updated_at=now()))
    def epoch_done(trainer):
        if args.smoke and optimizer_steps >= 2 and backbone_updated:
            trainer.stop = True
        save(out / 'progress.json', dict(epoch=min(trainer.epoch + 1, trainer.epochs), epochs=trainer.epochs,
            batches=batch_count, optimizer_steps=optimizer_steps,
            metrics={k:float(v) for k,v in trainer.metrics.items()}, updated_at=now()))
    model.add_callback('on_train_start', started)
    model.add_callback('on_train_batch_end', batch_done)
    model.add_callback('on_fit_epoch_end', epoch_done)
    model.train(trainer=MaskSelectionTrainer, **config)
    if batch_count < 1 or not model.trainer.best.exists():
        raise ValueError('No completed training batches/checkpoint')
    changed = not torch.equal(first_parameter, next(model.trainer.model.parameters()).detach().cpu())
    if not changed or optimizer_steps < 1:
        raise ValueError('Backbone did not update during full fine-tuning')
    save(out / 'complete.json', dict(model=args.model, smoke_only=args.smoke,
        best=str(model.trainer.best), best_sha256=sha(model.trainer.best),
        last=str(model.trainer.last), last_sha256=sha(model.trainer.last),
        base_sha256=sha(weight), batches=batch_count, backbone_updated=changed,
        optimizer_steps=optimizer_steps, optimizer_audit=optimizer_audit,
        dataset_manifest_sha256=sha(dataset / 'manifest.json'),
        best_mask_map50_95=float(model.trainer.best_fitness), completed_at=now(), deployed=False))


def run_worker(command, logfile, state, out):
    save(out / 'progress.json', dict(stage=state, command=command, updated_at=now(), deployed=False))
    with logfile.open('x', encoding='utf-8') as stream:
        result = subprocess.run(command, cwd=ROOT / 'pynq_km1_vision', stdout=stream, stderr=subprocess.STDOUT,
                                env=dict(os.environ, PYTHONUTF8='1'))
    if result.returncode:
        raise RuntimeError(f'{state} failed with exit code {result.returncode}; see {logfile}')


def campaign(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'authorization.json', dict(request='那就让这两个也跑这批试一下，比较一下',
        started_at=now(), models=list(BASES), sequential=True, deployed=False))
    inputs = audit(args.dataset.resolve())
    save(out / 'dataset-audit.json', inputs)
    save(out / 'plan.json', dict(dataset=str(args.dataset.resolve()), audit=inputs, epochs=args.epochs,
        batch=4, seed=20261030, checkpoint_selection='public validation mask mAP50-95',
        local_split='val only; never used for checkpoint selection', test_images_scored=False,
        yoloe_training='Existing classifier-only 12-epoch/patience-4 candidate; different training scope and model size',
        public_validation='Same 389 images; repeated development validation, not independent final acceptance',
        deployed=False))
    shutil.copy2(__file__, out / Path(__file__).name)
    shutil.copy2(ROOT / 'tools/evaluate_public26_comparison.py', out / 'evaluate_public26_comparison.py')
    failures = []
    yoloe = args.yoloe_run.resolve()
    # The already-running YOLOE process uses the GPU for validation each epoch.
    # Wait for its completion receipt before starting either new GPU worker.
    while not (yoloe / 'complete.json').exists():
        if args.wait_pid:
            alive = subprocess.run(['powershell', '-NoProfile', '-Command',
                f'if (Get-Process -Id {args.wait_pid} -ErrorAction SilentlyContinue) {{ exit 0 }} else {{ exit 1 }}'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if alive.returncode and not (yoloe / 'complete.json').exists():
                raise RuntimeError('Existing YOLOE worker exited without a completion receipt; queue stopped')
        save(out / 'progress.json', dict(stage='waiting_for_existing_yoloe', updated_at=now(),
            yoloe_progress=read(yoloe / 'training-progress.json'), deployed=False))
        time.sleep(15)
    yoloe_receipt = read(yoloe / 'complete.json')
    if sha(yoloe / 'best.pt') != yoloe_receipt['best_sha256']:
        raise ValueError('YOLOE completed checkpoint hash changed')
    for name in BASES:
        try:
            if args.reuse_smoke:
                previous_smoke = args.reuse_smoke.resolve() / name
                if (previous_smoke / 'complete.json').exists():
                    # Keep checkpoint paths pointing at their original evidence;
                    # the train worker rechecks hashes, AMP and actual updates.
                    current_smoke = out / 'smoke' / name
                    current_smoke.mkdir(parents=True, exist_ok=False)
                    for receipt_name in ('complete.json', 'actual-trainable.json'):
                        shutil.copy2(previous_smoke / receipt_name, current_smoke / receipt_name)
                    save(current_smoke / 'reused-smoke.json', dict(
                        source=str(previous_smoke), reused_at=now()))
            common = [sys.executable, '-u', str(Path(__file__).resolve()), '--stage', 'train',
                '--dataset', str(args.dataset.resolve()), '--model', name,
                '--weights', str(args.weights_dir.resolve() / (name + '.pt')), '--epochs', str(args.epochs)]
            run_worker(common + ['--smoke', '--output', str(out / 'smoke' / name)],
                out / (name + '-smoke.log'), 'smoke_' + name, out)
            run_worker(common + ['--output', str(out / name)],
                out / (name + '-train.log'), 'training_' + name, out)
        except Exception as exc:
            failures.append(dict(model=name, error=str(exc), traceback=traceback.format_exc()))
            save(out / 'failures.json', failures)
    candidates = [('yoloe-11m-base', 'yoloe', args.yoloe_base.resolve()),
                  ('yoloe-11m-trained', 'yoloe', yoloe / 'best.pt')]
    for name in BASES:
        if (out / name / 'complete.json').exists():
            candidates.append((name, 'yolo', Path(read(out / name / 'complete.json')['best'])))
    for name, kind, weight in candidates:
        try:
            run_worker([sys.executable, '-u', str(ROOT / 'tools/evaluate_public26_comparison.py'),
                '--dataset', str(args.dataset.resolve()), '--weights', str(weight), '--kind', kind,
                '--output', str(out / 'evaluation' / name)], out / (name + '-evaluate.log'), 'evaluating_' + name, out)
        except Exception as exc:
            failures.append(dict(model=name, error=str(exc), traceback=traceback.format_exc()))
            save(out / 'failures.json', failures)
    after = audit(args.dataset.resolve())
    if after != inputs:
        raise ValueError('Dataset changed during the campaign')
    summaries = {name:read(out / 'evaluation' / name / 'summary.json') for name,_,_ in candidates
                 if (out / 'evaluation' / name / 'summary.json').exists()}
    save(out / 'comparison.json', dict(candidates=summaries, failures=failures,
        finished_at=now(), dataset_unchanged=True, test_images_scored=False, deployed=False,
        complete=not failures and len(summaries)==4))
    save(out / 'progress.json', dict(stage='complete' if not failures else 'finished_with_failures',
        updated_at=now(), failures=failures, deployed=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['audit','train','campaign'], required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', choices=list(BASES))
    p.add_argument('--weights', type=Path)
    p.add_argument('--weights-dir', type=Path)
    p.add_argument('--yoloe-run', type=Path)
    p.add_argument('--yoloe-base', type=Path)
    p.add_argument('--wait-pid', type=int)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--reuse-smoke', type=Path,
        help='Reuse matching completed training checks from this smoke directory; all receipt checks still apply')
    args = p.parse_args()
    if args.epochs < 1 or not args.output.resolve().is_relative_to(ROOT / 'local-only'):
        p.error('Use positive epochs and a local-only output')
    if args.reuse_smoke and not args.reuse_smoke.resolve().is_relative_to(ROOT / 'local-only'):
        p.error('Reuse only local-only training evidence')
    if args.stage == 'audit':
        args.output.mkdir(parents=True, exist_ok=False)
        save(args.output / 'dataset-audit.json', audit(args.dataset.resolve()))
    elif args.stage == 'train':
        train(args)
    else:
        try:
            campaign(args)
        except Exception:
            if args.output.exists():
                save(args.output / 'fatal-error.json', dict(traceback=traceback.format_exc(), failed_at=now()))
            raise


if __name__ == '__main__':
    main()
