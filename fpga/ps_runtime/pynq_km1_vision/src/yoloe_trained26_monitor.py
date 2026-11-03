"""Fixed 26-class trained checkpoint through the existing local console protocol."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from .prompted_recyclables import read_policy, settings_for, adapt
from .yoloe_recyclable_monitor import RecyclablePlayground
from .yoloe_official_monitor import ROOT, make_server, jpeg64


class Trained26Detector:
    def __init__(self, weights, policy, filter_path, device, settings_dir):
        os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                          YOLO_AUTOINSTALL='false', YOLO_CONFIG_DIR=str(settings_dir))
        sys.path.insert(0, str(ROOT/'local-only/yoloe11m-26m-comparison/runtime'))
        import cv2
        import numpy as np
        import torch
        import ultralytics
        from ultralytics import YOLOE
        if ultralytics.__version__ != '8.4.193':
            raise ValueError('Trained candidate requires private Ultralytics 8.4.193')
        torch.set_num_threads(4)
        torch.backends.cudnn.enabled = False
        self.policy, self.device = policy, device
        self.weights = Path(weights).resolve()
        self.weight_hash = hashlib.sha256(self.weights.read_bytes()).hexdigest()
        if self.weight_hash != policy['model']['sha256'] or not policy['model'].get('fixed_classes'):
            raise ValueError('Candidate weight identity/fixed taxonomy does not match policy')
        self.model = YOLOE(str(self.weights))
        self.vocabulary = [r['official_label'] for r in policy['classes']]
        if len(self.vocabulary) != 26 or list(self.model.names.values()) != self.vocabulary:
            raise ValueError('Checkpoint must have the same 26 classes in the same order')
        self.embeddings = {}
        self.hashes = {policy['model']['name']: self.weight_hash}
        self.execution = dict(device=device, device_name=torch.cuda.get_device_name(0) if device != 'cpu' else 'CPU',
            precision='fp16' if device != 'cpu' else 'fp32', torch=torch.__version__,
            ultralytics=ultralytics.__version__, hip=torch.version.hip)
        self.filter_path = Path(filter_path).resolve()
        self.filter = json.loads(self.filter_path.read_text(encoding='utf-8'))
        self.filter_hash = hashlib.sha256(self.filter_path.read_bytes()).hexdigest()
        if policy.get('station_filter_sha256') != self.filter_hash:
            raise ValueError('Station filter identity does not match candidate policy')
        ref = (self.filter_path.parent/self.filter['reference_path']).resolve()
        if hashlib.sha256(ref.read_bytes()).hexdigest() != self.filter['reference_image_sha256']:
            raise ValueError('Station filter reference changed')
        self.reference = cv2.imread(str(ref))
        if self.reference is None or list(self.reference.shape[1::-1]) != self.filter['image_size']:
            raise ValueError('Invalid station filter reference geometry')
        self.reference_hash = hashlib.sha256(self.reference.tobytes()).hexdigest()

    def prepare(self, payload):
        return settings_for(self.policy, payload)

    def predict(self, frame, settings):
        import cv2
        import numpy as np
        from .yoloe_inference import encode_mask
        from .station_instance_filter import annotate_station
        started = time.perf_counter()
        raw = self.model.predict(frame, device=self.device, half=self.device != 'cpu', imgsz=640,
            conf=settings['confidence'], iou=.7, max_det=100, retina_masks=True,
            agnostic_nms=True, verbose=False, save=False)[0]
        instances = []
        if len(raw.boxes):
            if raw.masks is None or len(raw.masks.data) != len(raw.boxes):
                raise ValueError('Candidate segmentation masks missing')
            masks = raw.masks.data.detach().cpu().numpy() > .5
            if tuple(masks.shape[1:]) != frame.shape[:2]:
                raise ValueError('Candidate masks do not match input geometry')
            for i, box in enumerate(raw.boxes):
                ys, xs = np.nonzero(masks[i])
                if not len(xs):
                    continue
                cid = int(box.cls.item())
                instances.append(dict(class_id=cid, label=self.vocabulary[cid], confidence=float(box.conf.item()),
                    detection_bbox_xyxy=[float(x) for x in box.xyxy[0].tolist()],
                    bbox_xyxy=[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1],
                    mask_rle=encode_mask(masks[i]), area_px=len(xs)))
        result = adapt(self.policy, settings, dict(instances=instances, image_size=list(frame.shape[1::-1]),
            mode='text', fixed_classes=True, model_sha256=self.weight_hash,
            model_name='YOLOE-11m · 网上26类微调', classes=self.vocabulary, vocabulary_count=26,
            source_bgr_sha256=hashlib.sha256(frame.tobytes()).hexdigest(),
            execution=self.execution, confidence_threshold=settings['confidence'],
            overlay_base64=jpeg64(raw[raw.boxes.conf >= settings['display_confidence']].plot())
                if settings.get('return_overlay',True) else None,
            automatic_motion_allowed=False, generated_at=time.time()))
        registered = False
        shift, response = (None, None)
        if frame.shape == self.reference.shape:
            x0,y0,x1,y1 = self.filter['alignment_roi']
            ref = cv2.cvtColor(self.reference[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY).astype(np.float32)
            now = cv2.cvtColor(frame[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY).astype(np.float32)
            shift,response = cv2.phaseCorrelate(ref,now)
            registered = response >= .3 and max(abs(v) for v in shift) <= 2
        if registered:
            station = annotate_station(frame,self.reference,result['instances'],regions=self.filter['regions'],
                workcell_id=self.filter['workcell_id'],reference_workcell_id=self.filter['workcell_id'],
                alignment_verified=True,reference_sha256=self.reference_hash,
                inside_fraction=self.filter['inside_fraction'],changed_fraction=self.filter['changed_fraction'],
                pixel_difference=self.filter['pixel_difference'])
            result['instances'] = station['instances']
        else:
            for obj in result['instances']:
                obj.update(station_grasp_candidate=False,station_zone='unregistered',
                           station_filter_reasons=['station_reference_not_registered'])
        for obj in result['instances']:
            if not obj['station_grasp_candidate']:
                obj['semantic_gate_passed'] = False
                obj['decision_reasons'] = obj['decision_reasons'] + obj['station_filter_reasons']
        result.update(counts=dict(Counter(x['label'] for x in result['instances'])),
            inference_ms=(time.perf_counter()-started)*1000,
            station_filter=dict(registered=bool(registered),shift_px=shift,response=response,
                config_sha256=self.filter_hash,reference_bgr_sha256=self.reference_hash,
                raw_instances_preserved=True,automatic_motion_allowed=False))
        return result


class Trained26Playground(RecyclablePlayground):
    def health(self):
        value = super().health()
        value.update(model_name='YOLOE-11m · 网上26类微调', model_sha256=self.detector.weight_hash,
            trained_on_taco=True, fixed_classes=True, station_filter_config_sha256=self.detector.filter_hash)
        return value

    def backgrounds(self):
        value = super().backgrounds()
        value.update(model_trained=True, used_for_automatic_filtering=False,
                     source_filter_uses_separate_verified_reference=True)
        return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8811)
    parser.add_argument('--device',choices=['cpu','cuda:0'],default='cuda:0')
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--policy',type=Path,required=True)
    parser.add_argument('--filter',type=Path,required=True)
    parser.add_argument('--settings',type=Path,required=True)
    args=parser.parse_args()
    os.environ['YOLOE_RECYCLABLE_POLICY']=str(args.policy.resolve())
    detector=Trained26Detector(args.weights,read_policy(args.policy),args.filter,args.device,args.settings.parent/'runtime-settings')
    playground=Trained26Playground(detector,'http://127.0.0.1:8780/frame.jpg',args.settings,args.port)
    server=make_server(playground,args.port,'yoloe_trained26.html')
    playground.start()
    print('Fixed trained 26-class service ready on',args.port,flush=True)
    try:server.serve_forever()
    finally:playground.done.set();server.server_close()


if __name__=='__main__':main()
