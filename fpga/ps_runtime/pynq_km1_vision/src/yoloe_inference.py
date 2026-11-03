"""Prompted YOLOE material observations. No training, camera, or actuator access."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / 'pynq_km1_vision/config/yoloe_materials.json'


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_config(path=DEFAULT_CONFIG):
    cfg = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if cfg['schema_version'] != 'yoloe-materials-v1' or cfg['physical_motion_enabled'] is not False:
        raise ValueError('Only observation mode is supported')
    if [x['id'] for x in cfg['categories']] != ['plastic', 'metal', 'paper']:
        raise ValueError('Material order must be plastic, metal, paper')
    if cfg['device'] != 'cpu':
        raise ValueError('This deployment is verified for CPU only')
    if type(cfg['cpu_threads']) is not int or not 1 <= cfg['cpu_threads'] <= 8:
        raise ValueError('Invalid CPU thread count')
    for key in ('confidence', 'nms_iou'):
        value = cfg[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < 1:
            raise ValueError('Invalid ' + key)
    if type(cfg['image_size']) is not int or cfg['image_size'] != 640:
        raise ValueError('This preview uses image_size=640')
    if type(cfg['max_detections']) is not int or not 1 <= cfg['max_detections'] <= 100:
        raise ValueError('Invalid detection limit')
    prompts = cfg['prompts']
    if (not isinstance(prompts, list) or not 3 <= len(prompts) <= 30 or
        any(not isinstance(x.get('text'), str) or not x['text'].strip() or
            x['category'] not in ('plastic', 'metal', 'paper') for x in prompts) or
        len({x['text'] for x in prompts}) != len(prompts) or
        {x['category'] for x in prompts} != {'plastic', 'metal', 'paper'}):
        raise ValueError('Invalid material prompts')
    assets = (ROOT / cfg['asset_dir']).resolve()
    if not assets.is_relative_to(ROOT / 'local-only'):
        raise ValueError('Model assets must stay in local-only')
    return cfg


def encode_mask(mask):
    """Full-image row-major alternating 0/1 runs, starting with zeros."""
    import numpy as np
    values = np.asarray(mask, dtype=np.uint8).reshape(-1)
    if not len(values) or not np.isin(values, [0, 1]).all():
        raise ValueError('Expected a nonempty binary mask')
    edges = np.flatnonzero(values[1:] != values[:-1]) + 1
    runs = np.diff(np.r_[0, edges, len(values)]).tolist()
    return ([0] if values[0] else []) + runs


def convert_instances(raw, cfg, shape):
    """Require original-image masks; do not stretch letterboxed masks."""
    import numpy as np
    instances = []
    if raw.boxes is None or len(raw.boxes) == 0:
        return instances
    if raw.masks is None or len(raw.masks.data) != len(raw.boxes):
        raise ValueError('Instance masks are missing')
    masks = raw.masks.data.detach().cpu().numpy()
    if masks.shape[1:] != tuple(shape):
        raise ValueError('Mask shape differs from original image; retina_masks is required')
    ids = {x['id']: i for i, x in enumerate(cfg['categories'])}
    categories = {x['id']: x for x in cfg['categories']}
    for i, (class_value, score) in enumerate(zip(raw.boxes.cls.tolist(), raw.boxes.conf.tolist())):
        prompt_id = int(class_value)
        if class_value != prompt_id or not 0 <= prompt_id < len(cfg['prompts']):
            raise ValueError('Unexpected prompt class ID')
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid model score')
        prompt = cfg['prompts'][prompt_id]
        category = categories[prompt['category']]
        mask = masks[i] > 0.5
        ys, xs = np.nonzero(mask)
        if not len(xs):
            continue
        instances.append(dict(instance_id=len(instances), class_id=ids[category['id']],
            label=category['id'], display_name=category['name'],
            prompt_id=prompt_id, prompt=prompt.get('description', prompt['text']),
            prompt_key=prompt['text'], confidence=float(score),
            bbox_xyxy=[int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1],
            detection_bbox_xyxy=[float(x) for x in raw.boxes.xyxy[i].tolist()],
            mask_rle=encode_mask(mask), area_px=int(len(xs)), sorting_bin=None))
    return instances


class YoloeDetector:
    def __init__(self, config=DEFAULT_CONFIG, device='cpu', half=False):
        if device not in ('cpu', 'cuda:0') or type(half) is not bool or (device == 'cpu' and half):
            raise ValueError('Use cpu/FP32 or cuda:0 with an explicit precision')
        self.device, self.half = device, half
        self.cfg = read_config(config)
        self.runtime_id = config_hash(self.cfg)
        assets = ROOT / self.cfg['asset_dir']
        lock = json.loads((assets / 'deployment.json').read_text(encoding='utf-8'))
        if lock['config_sha256'] != self.runtime_id or lock['ultralytics'] != '8.3.230':
            raise ValueError('Deployment differs from config; run setup again')
        self.weights = assets / 'yoloe-materials-prompted.pt'
        self.model_id = file_hash(self.weights)
        if self.model_id != lock['prompted_sha256']:
            raise ValueError('Prompted model SHA256 mismatch')
        os.environ['YOLO_AUTOINSTALL'] = 'false'
        os.environ['YOLO_CONFIG_DIR'] = str(assets.parent / 'ultralytics-settings')
        (assets.parent / 'ultralytics-settings/Ultralytics').mkdir(parents=True, exist_ok=True)
        import torch
        import ultralytics
        from ultralytics import YOLOE
        expected_torch = lock['torch'] if device == 'cpu' else '2.12.0+rocm7.14.0'
        if ultralytics.__version__ != lock['ultralytics'] or torch.__version__ != expected_torch:
            raise ValueError('Use the recorded CPU runtime or the verified ROCm runtime')
        self.torch = torch
        self.device_name = 'CPU'
        if device != 'cpu':
            if not torch.version.hip or not torch.cuda.is_available():
                raise RuntimeError('ROCm GPU is required; CPU fallback is not allowed')
            self.device_name = torch.cuda.get_device_name(0)
            # This installed ROCm build cannot compile its MIOpen BN kernel.
            # Native GPU kernels remain active; this setting affects this process only.
            torch.backends.cudnn.enabled = False
        torch.set_num_threads(self.cfg['cpu_threads'])
        self.model = YOLOE(str(self.weights)).to(device)
        if list(self.model.names.values()) != [x['text'] for x in self.cfg['prompts']]:
            raise ValueError('Prompt order in model differs from config')
        self.models = {'text': self.model}
        self.model_ids = {'text': self.model_id}
        self.configs = {'text': self.cfg}
        self.visual_references = []
        if lock.get('visual_sha256'):
            visual_path = assets / 'yoloe-materials-visual.pt'
            references = json.loads((assets / 'visual_references.json').read_text(encoding='utf-8'))
            if file_hash(visual_path) != lock['visual_sha256'] or config_hash(references) != lock['visual_references_sha256']:
                raise ValueError('Visual model or reference manifest hash mismatch')
            visual = YOLOE(str(visual_path)).to(device)
            if list(visual.names.values()) != [p['text'] for p in lock['visual_prompts']]:
                raise ValueError('Visual prompt order mismatch')
            self.models['visual'] = visual
            self.model_ids['visual'] = lock['visual_sha256']
            self.configs['visual'] = dict(self.cfg, prompts=lock['visual_prompts'])
            self.visual_references = references
        self.default_prompt_mode = 'visual' if 'visual' in self.models else 'text'
        self.backend = 'pytorch_' + torch.__version__ + ('_rocm_gpu' if device != 'cpu' else '_cpu')
        self.execution = dict(device=device, device_name=self.device_name,
            precision='fp16' if half else 'fp32', torch=torch.__version__, hip=torch.version.hip,
            miopen_enabled=bool(torch.backends.cudnn.enabled) if device != 'cpu' else None)

    def execution_id(self, mode):
        return config_hash(dict(config=self.runtime_id, prompt_mode=mode,
                                model=self.model_ids[mode], execution=self.execution))

    def health(self):
        return dict(model_loaded=True, backend=self.backend, model_name=self.cfg['model_name'],
            model_sha256=self.model_ids[self.default_prompt_mode],
            runtime_id=self.execution_id(self.default_prompt_mode), execution=self.execution,
            categories=self.cfg['categories'], prompts=self.cfg['prompts'],
            prompt_modes=list(self.models), default_prompt_mode=self.default_prompt_mode,
            model_versions=self.model_ids, visual_reference_count=len(self.visual_references),
            confidence_threshold=self.cfg['confidence'], automatic_motion_allowed=False,
            quality_gate_passed=False, camera_connected=False)

    def predict(self, bgr, prompt_mode=None):
        import numpy as np
        if (not isinstance(bgr, np.ndarray) or bgr.dtype != np.uint8 or
            bgr.ndim != 3 or bgr.shape[2] != 3):
            raise ValueError('Expected uint8 BGR image')
        height, width = bgr.shape[:2]
        if not (1 <= width <= 1920 and 1 <= height <= 1080):
            raise ValueError('Image exceeds 1920x1080 preview limit')
        mode = prompt_mode or self.default_prompt_mode
        if mode not in self.models:
            raise ValueError('Requested prompt mode is not prepared')
        cfg = self.configs[mode]
        tensor_device = str(next(self.models[mode].model.parameters()).device)
        if self.device != 'cpu' and not tensor_device.startswith('cuda:'):
            raise RuntimeError('Model has left the requested ROCm GPU')
        if self.device != 'cpu':
            self.torch.cuda.synchronize()
        started = time.perf_counter()
        raw = self.models[mode].predict(bgr, device=self.device, half=self.half, imgsz=self.cfg['image_size'],
            conf=self.cfg['confidence'], iou=self.cfg['nms_iou'],
            max_det=self.cfg['max_detections'], retina_masks=True,
            agnostic_nms=True, verbose=False, save=False)[0]
        instances = convert_instances(raw, cfg, (height, width))
        if self.device != 'cpu':
            self.torch.cuda.synchronize()
        image_fingerprint = hashlib.sha256(bgr.tobytes()).hexdigest()
        return dict(schema_version='yoloe-material-observation-v1',
            image_size=[width, height], instances=instances,
            model_version=self.model_ids[mode], runtime_id=self.execution_id(mode),
            backend=self.backend, execution=self.execution, tensor_device=tensor_device, prompt_mode=mode,
            model_name=self.cfg['model_name'], categories=self.cfg['categories'],
            prompts=cfg['prompts'], confidence_threshold=self.cfg['confidence'],
            visual_reference_count=len(self.visual_references) if mode == 'visual' else 0,
            reference_image_match=any(image_fingerprint == r.get('decoded_bgr_sha256')
                                      for r in self.visual_references) if mode == 'visual' else False,
            nms_iou=self.cfg['nms_iou'], class_agnostic_nms=True,
            inference_ms=(time.perf_counter()-started)*1000,
            automatic_motion_allowed=False, quality_gate_passed=False,
            material_identity_verified=False, sorting_mapping_status='preview_only')


def annotate(frame, result):
    import cv2
    import numpy as np
    output = frame.copy()
    colors = [(255, 196, 77), (92, 196, 255), (155, 223, 119)]
    for item in result['instances']:
        flat = np.concatenate([np.full(n, i % 2, np.uint8)
                               for i, n in enumerate(item['mask_rle'])])
        mask = flat.reshape(frame.shape[:2]).astype(bool)
        color = colors[item['class_id']]
        output[mask] = (output[mask]*0.55 + np.array(color)*0.45).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, color, 2)
        x0, y0, x1, y1 = item['bbox_xyxy']
        cv2.rectangle(output, (x0, y0), (x1-1, y1-1), color, 1)
        cv2.putText(output, '{} {:.2f}'.format(item['label'], item['confidence']),
                    (x0, max(18, y0-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--prompt-mode', choices=['text', 'visual'])
    parser.add_argument('--device', choices=['cpu', 'cuda:0'], default='cpu')
    parser.add_argument('--half', action='store_true')
    args = parser.parse_args()
    import cv2
    import numpy as np
    data = args.image.read_bytes()
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        parser.error('Cannot decode image')
    detector = YoloeDetector(args.config, device=args.device, half=args.half)
    result = detector.predict(frame, prompt_mode=args.prompt_mode)
    result.update(image_sha256=hashlib.sha256(data).hexdigest(), input_kind='recorded_image')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    ok, overlay = cv2.imencode('.jpg', annotate(frame, result))
    if not ok:
        raise RuntimeError('Cannot encode overlay')
    (args.output_dir / 'overlay.jpg').write_bytes(overlay.tobytes())
    print(json.dumps(dict(inference_ms=result['inference_ms'],
        instances=[{k:v for k,v in x.items() if k != 'mask_rle'} for x in result['instances']]),
        ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
