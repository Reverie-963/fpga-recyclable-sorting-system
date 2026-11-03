"""Paired warm CPU/ROCm latency measurements on validation images, never test.

No camera, monitor, training or actuator access. Keeps all raw measurements.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pynq_km1_vision'))
import cv2
import numpy as np
from src.yolo11s_inference import convert_materials, POLICY


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    return dict(samples=len(values), median_ms=statistics.median(values), mean_ms=statistics.mean(values),
        p95_ms=float(np.percentile(values, 95)), p99_ms=float(np.percentile(values, 99)),
        min_ms=min(values), max_ms=max(values), sequential_fps=1000/statistics.mean(values))


def training_state():
    path = ROOT / 'local-only/recyclables-v8-followup/20261029-144923/progress.json'
    return read(path) if path.is_file() else None


def compare(observed, baseline):
    count_matches, ious, score_deltas = [], [], []
    for row, other in zip(baseline, observed):
        assert row['image'] == other['image']
        a, b = row['instances'], other['instances']
        same = Counter(x['label'] for x in a) == Counter(x['label'] for x in b)
        count_matches.append(same)
        if not same:
            continue
        used = set()
        for target in a:
            mask = np.repeat(np.arange(len(target['mask_rle'])) % 2, target['mask_rle']).astype(bool)
            choices = []
            for i, item in enumerate(b):
                if i in used or item['label'] != target['label']:
                    continue
                candidate = np.repeat(np.arange(len(item['mask_rle'])) % 2, item['mask_rle']).astype(bool)
                union = np.logical_or(mask, candidate).sum()
                iou = float(np.logical_and(mask, candidate).sum()/union) if union else 1.
                choices.append((iou, i))
            iou, index = max(choices)
            used.add(index)
            ious.append(iou)
            score_deltas.append(abs(target['confidence']-b[index]['confidence']))
    return dict(images=len(baseline), images_with_same_class_counts=sum(count_matches),
        matched_instances=len(ious), minimum_mask_iou=min(ious) if ious else None,
        max_score_delta=max(score_deltas) if score_deltas else None,
        note='Numerical sample check, not a full accuracy evaluation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    output, dataset = args.output.resolve(), args.dataset.resolve()
    if not output.is_relative_to(ROOT / 'local-only') or not 2 <= args.repeats <= 20 or not 1 <= args.threads <= 8:
        raise ValueError('Invalid local output or measurement settings')
    output.mkdir(parents=True, exist_ok=False)
    import torch
    import ultralytics
    from ultralytics import YOLO
    assert torch.__version__ == '2.12.0+rocm7.14.0' and ultralytics.__version__ == '8.3.230'
    assert torch.version.hip and torch.cuda.is_available()
    torch.backends.cudnn.enabled = False
    torch.set_num_threads(args.threads)
    manifest = read(dataset / 'split_manifest.json')
    rows = sorted((r for r in manifest['images'] if r['split'] == 'val'), key=lambda r:r['image'])
    assert len(rows) == 121
    rows = [rows[i] for i in (0, 1, 25, 49, 73, 97, 109, 119)]
    frames = []
    for row in rows:
        assert Path(row['image']).parts[:2] == ('images', 'val')
        assert digest(dataset / row['image']) == row['image_sha256']
        frames.append(cv2.imread(str(dataset / row['image'])))
    sources = {}
    for family, directory in [('yolo11s','yolo11s-deployment'), ('yolov8s','yolov8s-deployment')]:
        assets = ROOT / 'local-only' / directory / 'assets'
        lock = read(assets / 'deployment.json')
        weights = assets / 'model.pt'
        assert digest(weights) == lock['model_sha256']
        sources[family] = dict(weights=str(weights), model_sha256=lock['model_sha256'])
    variants = [('cpu_fp32','cpu',False), ('gpu_fp32','cuda:0',False), ('gpu_fp16','cuda:0',True)]
    plan = dict(started_at=datetime.now(timezone.utc).isoformat(), models=sources,
        variants=variants, batch=1, repeats=args.repeats, warmup_images_per_profile=len(rows),
        cpu_threads=args.threads, image_paths=[r['image'] for r in rows],
        image_hashes=[r['image_sha256'] for r in rows], policy=POLICY,
        torch=torch.__version__, hip=torch.version.hip, ultralytics=ultralytics.__version__,
        gpu=torch.cuda.get_device_name(0), miopen_enabled=False,
        runner_sha256=digest(__file__), training_at_start=training_state(),
        measurement_scope='In-memory BGR -> preprocessing -> network -> NMS/original masks -> CPU RLE. GPU synchronized.',
        excluded=['disk decode','weight loading','warmup','camera/network','annotation','browser','mechanical action'],
        sequence='Alternating forward/reverse profile blocks each round; one unmeasured rewarm call per block.',
        test_evaluated=False, trained=False, monitor_started=False)
    save(output / 'plan.json', plan)
    models, samples, observations = {}, {}, {}

    def predict(model, frame, device, half):
        if device != 'cpu':
            torch.cuda.synchronize()
        start = time.perf_counter_ns()
        raw = model.predict(frame, device=device, half=half, imgsz=640, conf=.3,
            iou=.7, max_det=20, agnostic_nms=False, retina_masks=True, save=False, verbose=False)[0]
        instances = convert_materials(raw, frame.shape[:2])
        if device != 'cpu':
            torch.cuda.synchronize()
        elapsed = (time.perf_counter_ns()-start)/1e6
        return elapsed, {k:float(v) for k,v in raw.speed.items()}, instances

    for family, source in sources.items():
        for variant, device, half in variants:
            key = family + '/' + variant
            model = YOLO(source['weights']).to(device)
            assert model.task == 'segment' and list(model.names.values()) == ['can','plastic','cardboard']
            models[key] = model
            samples[key], observations[key] = [], []
            for frame in frames:
                predict(model, frame, device, half)
            print('Warmed ' + key, flush=True)
    keys = list(models)
    variant_options = {name:(device,half) for name,device,half in variants}
    with (output / 'measurements.jsonl').open('w', encoding='utf-8') as stream:
        for round_index in range(args.repeats):
            order = keys if round_index % 2 == 0 else keys[::-1]
            for key in order:
                device, half = variant_options[key.split('/')[1]]
                model = models[key]
                predict(model, frames[0], device, half)
                for index, (row, frame) in enumerate(zip(rows, frames)):
                    elapsed, stages, instances = predict(model, frame, device, half)
                    actual_device = str(next(model.model.parameters()).device)
                    assert actual_device == device
                    measurement = dict(profile=key, round=round_index, image=row['image'],
                        pipeline_ms=elapsed, stages_ms=stages, instance_count=len(instances),
                        tensor_device=actual_device, tensor_dtype=str(next(model.model.parameters()).dtype))
                    samples[key].append(measurement)
                    stream.write(json.dumps(measurement)+'\n')
                    if round_index == 0:
                        observations[key].append(dict(image=row['image'], instances=instances))
                stream.flush()
                print(json.dumps(dict(round=round_index+1, profile=key, samples=len(samples[key]))), flush=True)
    results, speedups, consistency = {}, {}, {}
    for key, rows_measured in samples.items():
        results[key] = dict(pipeline=stats([r['pipeline_ms'] for r in rows_measured]),
            model_inference=stats([r['stages_ms']['inference'] for r in rows_measured]),
            tensor_device=rows_measured[0]['tensor_device'], tensor_dtype=rows_measured[0]['tensor_dtype'])
        save(output / (key.replace('/','-')+'-observations.json'), observations[key])
    for family in sources:
        cpu = results[family+'/cpu_fp32']['pipeline']['median_ms']
        speedups[family] = {variant:dict(median_latency_speedup=cpu/results[family+'/'+variant]['pipeline']['median_ms'],
            latency_reduction_percent=(1-results[family+'/'+variant]['pipeline']['median_ms']/cpu)*100)
            for variant in ('gpu_fp32','gpu_fp16')}
        consistency[family] = {variant:compare(observations[family+'/'+variant], observations[family+'/cpu_fp32'])
            for variant in ('gpu_fp32','gpu_fp16')}
    for source in sources.values():
        assert digest(source['weights']) == source['model_sha256']
    report = dict(plan, completed_at=datetime.now(timezone.utc).isoformat(), profiles=results,
        speedups=speedups, sample_consistency=consistency, training_at_end=training_state(),
        note='Concurrent unrelated training remains active. Speedups apply to these warm samples and four CPU threads, not camera FPS or an idle-device upper bound.')
    save(output / 'result.json', report)
    print(json.dumps(dict(speedups=speedups, consistency=consistency)), flush=True)


if __name__ == '__main__':
    main()
