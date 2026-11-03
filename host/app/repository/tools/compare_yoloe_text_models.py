"""Compare official YOLOE-11m/26m text prompts on the existing validation split.

Uses a private Ultralytics installation without changing the active runtime.
Never trains, deploys, opens the test split, or contacts a hardware interface.
Encode text with the dedicated CPU interpreter, then evaluate with the desired
interpreter. Predictions are saved and independently rescored from their RLE.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'local-only/yoloe11m-26m-comparison'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'encode', 'evaluate'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', choices=['yoloe-11m', 'yoloe-26m'])
    parser.add_argument('--runtime', type=Path, default=BASE / 'runtime')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--prompt-set', choices=['seven', 'taco24'], default='seven')
    parser.add_argument('--head-mode', choices=['nms', 'native'], default='nms')
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'local-only'):
        raise ValueError('Keep experiment output in local-only')
    if args.stage == 'prepare':
        output.mkdir(parents=True, exist_ok=False)
        config = ROOT / 'pynq_km1_vision/config' / (
            'yoloe_materials.json' if args.prompt_set == 'seven' else 'yoloe_taco_recyclables.json')
        document = json.loads(config.read_text(encoding='utf-8'))
        prompts = document['prompts'] if args.prompt_set == 'seven' else [
            dict(text=r['official_label'], category=r['material']) for r in document['classes']]
        if args.prompt_set == 'taco24' and (len(prompts) != 24 or any(p['category'] is None for p in prompts)):
            raise ValueError('Expected the current 24 sortable prompts; review policy changes')
        dataset = ROOT / 'local-only/recyclables-final-16-v1'
        manifest = json.loads((dataset / 'split_manifest.json').read_text(encoding='utf-8'))
        rows = sorted((r for r in manifest['images'] if r['split'] == 'val'), key=lambda r: r['image'])
        if len(rows) != 121:
            raise ValueError('Unexpected validation split; review before comparing')
        model_paths = {'yoloe-11m': ROOT / 'local-only/yoloe-deployment/assets/yoloe-11m-seg.pt',
                       'yoloe-26m': BASE / 'assets/yoloe-26m-seg.pt'}
        plan = dict(schema_version='yoloe-text-validation-comparison-v1',
                    created_at=datetime.now(timezone.utc).isoformat(),
                    dataset=str(dataset.relative_to(ROOT)), split='val', rows=rows,
                    manifest_sha256=sha(dataset / 'split_manifest.json'),
                    prompt_set=args.prompt_set, prompt_config_sha256=sha(config), prompts=prompts,
                    class_names=['can', 'plastic', 'cardboard'],
                    class_map={str(i): {'metal': 0, 'plastic': 1, 'paper': 2}[p['category']]
                               for i, p in enumerate(prompts)},
                    policy=dict(imgsz=640, conf=.3, iou=.7, max_det=20,
                                agnostic_nms=True, retina_masks=True, verbose=False,
                                save=False, batch=1, half=False),
                    head_mode=args.head_mode,
                    comparison_head=('NMS head for both models (nms=True)' if args.head_mode == 'nms'
                                     else '26m default end-to-end head; 11m ordinary NMS head'),
                    cpu_threads=4, mask_iou=.5, training=False, test_split_opened=False,
                    label_hash_policy='Compare manifest hash after CRLF-to-LF normalization; also freeze actual bytes',
                    runtime_version='8.4.193', code_sha256=sha(__file__),
                    models={name: dict(path=str(path.relative_to(ROOT)), sha256=sha(path))
                            for name, path in model_paths.items()})
        save(output / 'plan.json', plan)
        (output / 'code-snapshot.py').write_bytes(Path(__file__).read_bytes())
        print('Frozen 121 validation images,', len(prompts), 'prompts, confidence .3, mask IoU .5', flush=True)
        return

    plan = json.loads((output / 'plan.json').read_text(encoding='utf-8'))
    if sha(__file__) != plan['code_sha256']:
        raise ValueError('Experiment code changed after freezing plan')
    for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
        os.environ[key] = '1' if key == 'OPENBLAS_NUM_THREADS' else '4'
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ['YOLO_CONFIG_DIR'] = str(output / 'settings')
    sys.path.insert(0, str(args.runtime.resolve()))
    sys.path.insert(0, str(ROOT / 'pynq_km1_vision'))
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLOE
    if ultralytics.__version__ != plan['runtime_version']:
        raise ValueError('Unexpected Ultralytics version')
    torch.set_num_threads(plan['cpu_threads'])
    names = [p['text'] for p in plan['prompts']]
    selected = [args.model] if args.model else list(plan['models'])
    if args.stage == 'encode':
        if args.device != 'cpu':
            raise ValueError('Text encoding must use the dedicated CPU interpreter')
        for name in selected:
            spec = plan['models'][name]
            path = ROOT / spec['path']
            assert sha(path) == spec['sha256']
            os.chdir(path.parent)
            model = YOLOE(str(path)).to('cpu')
            started = time.perf_counter()
            with torch.inference_mode():
                values = model.get_text_pe(names).detach().cpu().float().numpy()
            assert values.shape == (1, len(names), 512) and np.isfinite(values).all()
            destination = output / (name + '-text.npy')
            with destination.open('xb') as stream:
                np.save(stream, values, allow_pickle=False)
            save(output / (name + '-encoding.json'), dict(model_sha256=spec['sha256'],
                 names=names, sha256=sha(destination), shape=list(values.shape),
                 torch=torch.__version__, ultralytics=ultralytics.__version__,
                 elapsed_s=time.perf_counter()-started, text_model=getattr(model.model, 'text_model', None)))
            print('Encoded', name, list(values.shape), flush=True)
            del model
            gc.collect()
        return

    import cv2
    from src.train_recyclables_final import (label_path, match_instances, merge_counts, new_counts,
                                             describe_counts)
    from compare_material_models import (truth_for, predictions_for, decode_mask, error_details, overlay, f1_for)
    from src.yoloe_inference import encode_mask
    if args.device != 'cpu':
        if not torch.cuda.is_available() or not torch.version.hip:
            raise ValueError('Requested ROCm GPU is unavailable')
        torch.backends.cudnn.enabled = False
    dataset = ROOT / plan['dataset']
    assert sha(dataset / 'split_manifest.json') == plan['manifest_sha256']
    before = {}
    for row in plan['rows']:
        assert row['split'] == 'val'
        for path in [dataset / row['image'], label_path(dataset, row)]:
            before[str(path)] = sha(path)
        assert sha(dataset / row['image']) == row['image_sha256']
        label_bytes = label_path(dataset, row).read_bytes()
        assert hashlib.sha256(label_bytes.replace(b'\r\n', b'\n')).hexdigest() == row['label_sha256']
    save(output / 'input-byte-hashes.json', before)
    save(output / ('environment-' + args.device.replace(':', '-') + '.json'),
         dict(torch=torch.__version__, ultralytics=ultralytics.__version__,
              ultralytics_file=ultralytics.__file__, numpy=np.__version__, cv2=cv2.__version__,
              hip=torch.version.hip, device=args.device, half=False, threads=plan['cpu_threads'],
              device_name=torch.cuda.get_device_name(0) if args.device != 'cpu' else 'CPU',
              timing_note='Same machine with other services running; not isolated throughput benchmark'))
    class_map = {int(k): v for k, v in plan['class_map'].items()}
    summaries = {}
    for name in selected:
        spec = plan['models'][name]
        assert sha(ROOT / spec['path']) == spec['sha256']
        model = YOLOE(str(ROOT / spec['path'])).to(args.device)
        embedding_path = output / (name + '-text.npy')
        encoding = json.loads((output / (name + '-encoding.json')).read_text(encoding='utf-8'))
        assert sha(embedding_path) == encoding['sha256'] and encoding['names'] == names
        values = np.load(embedding_path, allow_pickle=False)
        model.set_classes(names, torch.from_numpy(values).to(args.device))
        assert list(model.names.values()) == names
        policy = dict(plan['policy'], device=args.device,
                      nms=plan.get('head_mode', 'nms') == 'nms' or name == 'yoloe-11m')
        warm_frame = cv2.imread(str(dataset / plan['rows'][0]['image']))
        for _ in range(3):
            model.predict(warm_frame, **policy)
        model_head = model.model.model[-1]
        folder = output / args.device.replace(':', '-') / name
        folder.mkdir(parents=True, exist_ok=False)
        save(folder / 'model.json', dict(model_sha256=spec['sha256'],
             text_embedding_sha256=sha(embedding_path), parameters=sum(p.numel() for p in model.model.parameters()),
             head_type=type(model_head).__name__, end2end=bool(getattr(model_head, 'end2end', False)),
             predictor_end2end=bool(getattr(model.predictor.model, 'end2end', False)),
             names=model.names, policy=policy))
        total, groups, times, details = new_counts(), defaultdict(new_counts), [], []
        with (folder / 'predictions.jsonl').open('x', encoding='utf-8') as stream:
            for index, row in enumerate(plan['rows'], 1):
                frame = cv2.imread(str(dataset / row['image']))
                truth = truth_for(dataset, row, frame.shape[:2])
                if args.device != 'cpu':
                    torch.cuda.synchronize()
                started = time.perf_counter()
                raw = model.predict(frame, **policy)[0]
                if args.device != 'cpu':
                    torch.cuda.synchronize()
                elapsed = (time.perf_counter()-started)*1000
                predictions = predictions_for(raw, frame.shape[:2], class_map)
                counts = match_instances(truth, predictions, plan['mask_iou'])
                errors = error_details(truth, predictions)
                assert sum(e['type'] == 'fp' for e in errors) == sum(counts['fp'])
                assert sum(e['type'] == 'fn' for e in errors) == sum(counts['fn'])
                encoded = [dict(**{k: v for k, v in p.items() if k != 'mask'},
                                mask_rle=encode_mask(p['mask'])) for p in predictions]
                stream.write(json.dumps(dict(image=row['image'], image_sha256=sha(dataset / row['image']),
                     label_sha256=sha(label_path(dataset, row)), shape=list(frame.shape[:2]),
                     predictions=encoded, counts=counts, inference_ms=elapsed), allow_nan=False) + '\n')
                stream.flush()
                merge_counts(total, counts)
                merge_counts(groups[row['scene_type']], counts)
                times.append(elapsed)
                details.append(dict(image=row['image'], scene_type=row['scene_type'], errors=errors, counts=counts))
                if index <= 3 or row['scene_type'] == 'empty':
                    cv2.imwrite(str(folder / ('example-%03d.jpg' % index)), overlay(frame, truth, predictions))
                if index % 20 == 0 or index == len(plan['rows']):
                    progress = dict(model=name, device=args.device, complete=index, total=len(plan['rows']))
                    save(output / 'progress.json', progress)
                    print(json.dumps(progress), flush=True)
        fixed = dict(overall=describe_counts(total, plan['class_names']),
                     scenes={k: describe_counts(v, plan['class_names']) for k, v in groups.items()})
        fixed['macro_f1'] = sum(f1_for(v) for v in fixed['overall']['per_class'].values()) / 3
        save(folder / 'details.json', details)
        recounted, regroups = new_counts(), defaultdict(new_counts)
        records = (folder / 'predictions.jsonl').read_text(encoding='utf-8').splitlines()
        for row, line in zip(plan['rows'], records, strict=True):
            record = json.loads(line)
            assert record['image'] == row['image']
            decoded = [dict(class_id=p['class_id'], confidence=p['confidence'],
                            mask=decode_mask(p['mask_rle'], record['shape'])) for p in record['predictions']]
            checked = match_instances(truth_for(dataset, row, record['shape']), decoded, plan['mask_iou'])
            assert checked == record['counts']
            merge_counts(recounted, checked)
            merge_counts(regroups[row['scene_type']], checked)
        assert recounted == total and dict(regroups) == dict(groups)
        summary = dict(model=name, fixed=fixed, prediction_sha256=sha(folder / 'predictions.jsonl'),
                       median_ms=float(np.median(times)), p95_ms=float(np.percentile(times, 95)),
                       saved_mask_audit_images=len(records), saved_mask_audit_passed=True)
        save(folder / 'summary.json', summary)
        summaries[name] = summary
        print(name, json.dumps(summary), flush=True)
        del model, raw
        gc.collect()
        if args.device != 'cpu':
            torch.cuda.empty_cache()
    assert before == {path: sha(path) for path in before}
    assert sha(dataset / 'split_manifest.json') == plan['manifest_sha256']
    save(output / ('comparison-' + args.device.replace(':', '-') + '.json'),
         dict(plan_sha256=sha(output / 'plan.json'), candidates=summaries,
              input_hashes_unchanged=True, test_split_opened=False, trained=False, deployed=False))


if __name__ == '__main__':
    main()
