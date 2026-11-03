"""Fixed-checkpoint recognition with explicit ROCm GPU or CPU selection. No actuators."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def runtime_execution(torch, device):
    if device not in ('cuda:0', 'cpu'):
        raise ValueError('unsupported_candidate_device')
    if device == 'cuda:0':
        if not torch.version.hip or not torch.cuda.is_available():
            raise RuntimeError('ROCm GPU required; no silent CPU fallback')
        torch.backends.cudnn.enabled = False
    return dict(device=device, device_name=torch.cuda.get_device_name(0) if device != 'cpu' else 'CPU',
        precision='fp16' if device != 'cpu' else 'fp32', torch=torch.__version__, hip=torch.version.hip)


class CandidateDetector:
    def __init__(self, entry, device='cuda:0'):
        import torch
        from ultralytics import YOLO, YOLOE
        from .yoloe_inference import file_hash, config_hash
        from .recognition_profiles import configuration, configuration_hash
        weights = Path(entry['weights']).resolve()
        if not weights.is_file() or file_hash(weights) != entry['sha256']:
            raise ValueError('candidate_weight_hash_mismatch')
        torch.set_num_threads(2)
        self.entry, self.torch, self.device = entry, torch, device
        self.half = device != 'cpu'
        self.execution = runtime_execution(torch, device)
        self.configuration = configuration(entry)
        self.configuration_sha256 = configuration_hash(entry)
        if entry.get('configuration_sha256', self.configuration_sha256) != self.configuration_sha256:
            raise ValueError('candidate_configuration_identity_changed')
        self.confidence = self.configuration['confidence']
        self.reference = None
        if entry.get('reference_bank'):
            from .inventory_reference import ReferenceBank
            self.reference = ReferenceBank(entry['reference_bank'], entry['reference_bank_sha256'])
        self.model = (YOLOE if entry['family'] == 'yoloe' else YOLO)(str(weights)).to(device)
        if self.model.task != 'segment' or list(self.model.names.values()) != entry['classes']:
            raise ValueError('candidate_class_order_mismatch')
        if set(entry['mapping']) != set(entry['classes']) or not set(entry['mapping'].values()) <= {'metal', 'plastic', 'paper'}:
            raise ValueError('candidate_material_mapping_incomplete')
        self.models = {'supervised': self.model}
        self.default_prompt_mode = 'supervised'
        self.runtime_id = config_hash(dict(model=entry['sha256'], classes=entry['classes'],
            mapping=entry['mapping'], execution=self.execution, configuration_sha256=self.configuration_sha256))

    def health(self):
        tensor = self.inference_tensor()
        return dict(model_loaded=True, model_family=self.entry['family'], model_name=self.entry['name'],
            model_sha256=self.entry['sha256'], runtime_id=self.runtime_id, execution=self.execution,
            vocabulary_count=len(self.entry['classes']), default_prompt_mode='supervised',
            prompt_modes=['supervised'], confidence_threshold=self.confidence, automatic_motion_allowed=False,
            configuration_sha256=self.configuration_sha256, profile_id=self.entry['id'],
            reference_assistance=self.reference is not None,
            tensor_device=str(tensor.device), tensor_dtype=str(tensor.dtype),
            backend='pytorch_rocm_gpu' if self.half else 'pytorch_cpu', preview_only=True)

    def inference_tensor(self):
        # The predictor copies/fuses the source model. Check its execution copy.
        engine = getattr(getattr(self.model, 'predictor', None), 'model', None)
        module = getattr(getattr(engine, 'backend', engine), 'model', None)
        return next((module if module is not None else self.model.model).parameters())

    def predict(self, bgr, prompt_mode=None):
        return self._predict(bgr,prompt_mode,False)

    def predict_control(self,bgr,prompt_mode=None):
        return self._predict(bgr,prompt_mode,True)

    def _predict(self, bgr, prompt_mode=None, control_only=False):
        import numpy as np
        import cv2
        from .yoloe_inference import encode_mask, annotate
        from .monitor_grasp import display_candidates
        if prompt_mode not in (None, 'supervised'):
            raise ValueError('fixed_candidate_classes')
        if str(next(self.model.model.parameters()).device) != self.device:
            raise RuntimeError('candidate_model_device_changed')
        if self.half:
            self.torch.cuda.synchronize()
        started = time.perf_counter()
        raw = self.model.predict(bgr, device=self.device, quantize=16 if self.half else 32, imgsz=640, conf=self.confidence, iou=.7,
            max_det=100, retina_masks=True, agnostic_nms=True, save=False, verbose=False)[0]
        tensor = self.inference_tensor()
        if str(tensor.device) != self.device or tensor.dtype != (self.torch.float16 if self.half else self.torch.float32):
            raise RuntimeError('candidate_execution_device_or_precision_changed')
        instances = []
        if len(raw.boxes):
            if raw.masks is None or tuple(raw.masks.data.shape[1:]) != bgr.shape[:2]:
                raise ValueError('candidate_masks_missing_or_resized')
            masks = raw.masks.data.cpu().numpy() > .5
            for i, box in enumerate(raw.boxes):
                ys, xs = np.nonzero(masks[i])
                if not len(xs):
                    continue
                label = self.entry['classes'][int(box.cls.item())]
                material = self.entry['mapping'][label]
                original_material = material
                assistance = self.reference.assist(bgr, masks[i], hashlib.sha256(bgr.tobytes()).hexdigest()) if self.reference else {}
                if assistance.get('reference_assisted'):
                    material = assistance['reference_material']
                instances.append(dict(instance_id=len(instances), label=material, category_name=material,
                    official_label=label, prompt_key=label, confidence=float(box.conf.item()),
                    model_material=original_material, **assistance,
                    class_id={'plastic': 0, 'metal': 1, 'paper': 2}[material],
                    bbox_xyxy=[int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1],
                    mask_rle=encode_mask(masks[i]), area_px=int(len(xs)), sorting_bin=None))
        if self.half:
            self.torch.cuda.synchronize()
        result = dict(image_size=list(bgr.shape[1::-1]), instances=instances,
            inference_ms=(time.perf_counter()-started)*1000, model_version=self.entry['sha256'],
            model_name=self.entry['name'], runtime_id=self.runtime_id, execution=self.execution,
            configuration_sha256=self.configuration_sha256, profile_id=self.entry['id'],
            confidence_threshold=self.confidence, reference_assistance=self.reference is not None,
            source_bgr_sha256=hashlib.sha256(bgr.tobytes()).hexdigest(), automatic_motion_allowed=False)
        if control_only:
            result['control_only']=True
            return result
        result=display_candidates(result)
        ok, overlay = cv2.imencode('.jpg', annotate(bgr, result))
        if not ok:
            raise ValueError('candidate_overlay_encoding_failed')
        result['overlay_base64'] = base64.b64encode(overlay.tobytes()).decode('ascii')
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--device', choices=['cuda:0', 'cpu'])
    args = p.parse_args()
    config = json.loads(args.catalog.read_text(encoding='utf-8-sig'))
    os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
        YOLO_AUTOINSTALL='false', YOLO_CONFIG_DIR=str(args.catalog.parent/'worker-settings'))
    if config.get('runtime'):
        sys.path.insert(0, config['runtime'])
    entry = next(row for row in config['models'] if row['id'] == args.model)
    if entry['kind'] != 'checkpoint' or entry.get('disabled_reason'):
        raise ValueError('candidate_not_available')
    # Preserve explicitly registered legacy CPU catalogs until an operator
    # applies the GPU deployment; never relabel a still-running CPU service.
    detector = CandidateDetector(entry, args.device or entry.get('device', config.get('device', 'cpu')))
    from .yoloe_preview_server import make_server
    server = make_server(detector, args.port, args.catalog.parent/'uploads'/args.model,
                         camera_url=config['camera_url'])
    server.live.start()
    print('Candidate recognition ready: '+args.model, flush=True)
    try:
        server.serve_forever()
    finally:
        server.live.close()
        server.server_close()


if __name__ == '__main__':
    main()
