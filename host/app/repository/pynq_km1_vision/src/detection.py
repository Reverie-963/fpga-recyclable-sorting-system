"""Protocol-v2 YOLO11 detection on CUDA/ROCm/CPU; no actuator access."""
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np


def validate_detections(items, width, height, limit=100):
    if not isinstance(items, list) or len(items) > limit:
        raise ValueError('Invalid detection list')
    clean = []
    for item in items:
        box = item.get('xyxy'); score = item.get('confidence'); label = item.get('label')
        if not isinstance(label, str) or not label or len(label) > 100:
            raise ValueError('Invalid label')
        if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid confidence')
        if not isinstance(box, list) or len(box) != 4 or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in box):
            raise ValueError('Invalid box')
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError('Box outside source frame')
        clean.append({'label': label, 'confidence': float(score), 'xyxy': list(map(float, box))})
    return clean


class YOLODetector:
    def __init__(self, cfg):
        import torch
        from ultralytics import YOLO
        self.cfg = cfg; self.torch = torch
        path = Path(cfg['model'])
        if not path.is_file():
            raise FileNotFoundError('Train and deploy your own YOLO11n best.pt: ' + str(path))
        backend = cfg['accelerator']
        if backend not in ('cpu', 'cuda', 'rocm'):
            raise ValueError('accelerator must be cpu/cuda/rocm')
        hip = getattr(torch.version, 'hip', None)
        if backend != 'cpu' and not torch.cuda.is_available():
            raise RuntimeError('Requested GPU unavailable; no CPU fallback')
        if backend == 'rocm' and not hip:
            raise RuntimeError('ROCm PyTorch required, not a CUDA build')
        if backend == 'cuda' and (hip or not torch.version.cuda):
            raise RuntimeError('NVIDIA CUDA PyTorch required')
        if type(cfg['imgsz']) is not int or cfg['imgsz'] < 32 or cfg['imgsz'] % 32:
            raise ValueError('imgsz must be a positive multiple of 32')
        for k in ('confidence', 'iou'):
            if not 0 < cfg[k] <= 1:
                raise ValueError('Invalid ' + k)
        if not 1 <= cfg['max_det'] <= 100:
            raise ValueError('max_det must be 1..100')
        self.device = 'cpu' if backend == 'cpu' else int(cfg.get('device', 0))
        self.model_id = hashlib.sha256(path.read_bytes()).hexdigest()
        self.runtime_id = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
        self.providers = [backend + '_pytorch']
        self.model = YOLO(str(path), task='detect')
        if self.model.task != 'detect':
            raise ValueError('Detection weights required')
        self.labels = list(self.model.names.values())
        self.predict(np.zeros((cfg['imgsz'], cfg['imgsz'], 3), np.uint8))

    def predict(self, frame):
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError('Expected BGR image')
        if self.device != 'cpu':
            self.torch.cuda.synchronize(self.device)
        t = time.perf_counter()
        result = self.model.predict(source=frame, imgsz=self.cfg['imgsz'],
            device=self.device, conf=self.cfg['confidence'], iou=self.cfg['iou'],
            max_det=self.cfg['max_det'], half=False, verbose=False)[0]
        raw = result.boxes.data.detach().cpu().numpy()
        detections = [{'xyxy': row[:4].tolist(), 'confidence': float(row[4]),
                       'label': result.names[int(row[5])]} for row in raw]
        if self.device != 'cpu':
            self.torch.cuda.synchronize(self.device)
        timings = {k + '_ms': float(v) for k, v in result.speed.items()}
        timings['total_ms'] = (time.perf_counter() - t) * 1000
        h, w = frame.shape[:2]
        return {'detections': validate_detections(detections, w, h), 'timing': timings,
                'model_id': self.model_id, 'runtime_id': self.runtime_id,
                'providers': self.providers, 'image_size': [w, h]}
