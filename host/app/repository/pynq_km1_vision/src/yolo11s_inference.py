"""Frozen local YOLO11s three-material monitor adapter; ROCm GPU only."""
import json
from pathlib import Path
import time

from .yoloe_inference import ROOT, config_hash, file_hash, convert_instances

ASSETS = ROOT / 'local-only/yolo11s-deployment/assets'
CATEGORIES = [dict(id='plastic', name='塑料'), dict(id='metal', name='金属'),
              dict(id='paper', name='纸质')]
PROMPTS = [dict(text='can', category='metal'), dict(text='plastic', category='plastic'),
           dict(text='cardboard', category='paper')]
POLICY = dict(image_size=640, confidence=.3, nms_iou=.7, max_detections=20,
              class_agnostic_nms=False)


def convert_materials(raw, shape):
    return convert_instances(raw, dict(categories=CATEGORIES, prompts=PROMPTS), shape)


class Yolo11sDetector:
    MODEL_FAMILY = 'yolo11s'
    MODEL_NAME = 'YOLO11s-seg'
    LOCK_SCHEMA = 'yolo11s-monitor-deployment-v1'

    def __init__(self, assets=ASSETS, *, device='cuda:0', half=True):
        if device != 'cuda:0' or type(half) is not bool:
            raise ValueError('This deployment requires the verified ROCm cuda:0 runtime')
        assets = Path(assets).resolve()
        if not assets.is_relative_to(ROOT / 'local-only'):
            raise ValueError('Model assets must stay in local-only')
        lock = json.loads((assets / 'deployment.json').read_text(encoding='utf-8'))
        weights = assets / 'model.pt'
        self.model_id = file_hash(weights)
        if (lock['schema_version'] != self.LOCK_SCHEMA or
            self.model_id != lock['model_sha256'] or lock['class_names'] != [p['text'] for p in PROMPTS]):
            raise ValueError('Frozen model or material class order mismatch')
        import torch
        import ultralytics
        from ultralytics import YOLO
        if (torch.__version__ != '2.12.0+rocm7.14.0' or ultralytics.__version__ != '8.3.230' or
            not torch.version.hip or not torch.cuda.is_available()):
            raise RuntimeError('Use the recorded ROCm runtime; no CPU fallback')
        torch.backends.cudnn.enabled = False
        torch.set_num_threads(4)
        self.torch, self.device, self.half = torch, device, half
        self.model = YOLO(str(weights)).to(device)
        if self.model.task != 'segment' or list(self.model.names.values()) != lock['class_names']:
            raise ValueError('Expected local three-material segmentation weights')
        self.models = {'supervised': self.model}
        self.default_prompt_mode = 'supervised'
        self.execution = dict(device=device, device_name=torch.cuda.get_device_name(0),
            precision='fp16' if half else 'fp32', torch=torch.__version__, hip=torch.version.hip,
            miopen_enabled=False)
        self.backend = 'pytorch_' + torch.__version__ + '_rocm_gpu'
        self.runtime_id = config_hash(dict(model=self.model_id, execution=self.execution,
            policy=POLICY, categories=CATEGORIES, mapping=PROMPTS, schema='recyclable-material-observation-v1'))

    def health(self):
        return dict(model_loaded=True, model_family=self.MODEL_FAMILY, model_name=self.MODEL_NAME,
            model_display_name=self.MODEL_NAME + ' · 本地三类专项训练', backend=self.backend,
            model_sha256=self.model_id, model_versions={'supervised': self.model_id},
            runtime_id=self.runtime_id, execution=self.execution, categories=CATEGORIES,
            prompt_modes=['supervised'], default_prompt_mode='supervised',
            visual_reference_count=0, confidence_threshold=POLICY['confidence'],
            inference_policy=POLICY, automatic_motion_allowed=False, quality_gate_passed=False,
            camera_connected=False, local_supervised_training=True)

    def predict(self, bgr, prompt_mode=None):
        import numpy as np
        if prompt_mode not in (None, 'supervised'):
            raise ValueError('This model uses fixed trained classes; visual/text prompts are unavailable')
        if (not isinstance(bgr, np.ndarray) or bgr.dtype != np.uint8 or bgr.ndim != 3 or
            bgr.shape[2] != 3 or not 1 <= bgr.shape[1] <= 1920 or not 1 <= bgr.shape[0] <= 1080):
            raise ValueError('Expected a uint8 BGR image up to 1920x1080')
        tensor_device = str(next(self.model.model.parameters()).device)
        if not tensor_device.startswith('cuda:'):
            raise RuntimeError('Model left the ROCm GPU')
        self.torch.cuda.synchronize()
        start = time.perf_counter()
        raw = self.model.predict(bgr, device=self.device, half=self.half,
            imgsz=POLICY['image_size'], conf=POLICY['confidence'], iou=POLICY['nms_iou'],
            max_det=POLICY['max_detections'], agnostic_nms=False,
            retina_masks=True, verbose=False, save=False)[0]
        instances = convert_materials(raw, bgr.shape[:2])
        self.torch.cuda.synchronize()
        return dict(schema_version='recyclable-material-observation-v1',
            image_size=list(bgr.shape[:2][::-1]), instances=instances,
            model_version=self.model_id, runtime_id=self.runtime_id, model_name=self.MODEL_NAME,
            backend=self.backend, execution=self.execution, tensor_device=tensor_device,
            prompt_mode='supervised', categories=CATEGORIES, confidence_threshold=POLICY['confidence'],
            visual_reference_count=0, reference_image_match=False,
            nms_iou=POLICY['nms_iou'], class_agnostic_nms=False,
            inference_ms=(time.perf_counter()-start)*1000,
            automatic_motion_allowed=False, quality_gate_passed=False,
            material_identity_verified=False, sorting_mapping_status='preview_only')
