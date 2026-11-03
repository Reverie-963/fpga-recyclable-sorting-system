"""Preview a fixed 26-class candidate and workstation annotations on one photo.

This offline tool writes evidence only. It never contacts the camera, board,
serial port or motion services. All raw masks are retained in the JSON result.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--taxonomy', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--workcell-id', required=True)
    parser.add_argument('--filter-config', type=Path,
                        default=ROOT / 'pynq_km1_vision/config/yoloe_station_filter.example.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--confidence', type=float, default=.3)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT / 'local-only'):
        raise ValueError('Keep photo evidence in local-only')
    if not 0 < args.confidence < 1:
        raise ValueError('Confidence must be between zero and one')
    for path in (args.weights, args.taxonomy, args.image, args.reference, args.filter_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = json.loads(args.filter_config.read_text(encoding='utf-8'))
    taxonomy = json.loads(args.taxonomy.read_text(encoding='utf-8'))['classes']
    names = [row['official_label'] for row in taxonomy]
    if len(names) != 26 or len(set(names)) != 26:
        raise ValueError('Expected the fixed 26-class taxonomy')
    if (config['workcell_id'] != args.workcell_id or
            config['reference_image_sha256'] != sha(args.reference)):
        raise ValueError('Workcell/reference does not match the filter configuration')
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                      YOLO_AUTOINSTALL='false', YOLO_CONFIG_DIR=str(out / 'settings'))
    sys.path.insert(0, str(ROOT / 'local-only/yoloe11m-26m-comparison/runtime'))
    sys.path.insert(0, str(ROOT / 'pynq_km1_vision'))
    import cv2
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLOE
    from src.station_instance_filter import annotate_station
    from src.yoloe_inference import encode_mask
    torch.set_num_threads(4)
    torch.backends.cudnn.enabled = False
    if ultralytics.__version__ != '8.4.193':
        raise ValueError('Use the private pinned Ultralytics 8.4.193 runtime')
    frame = cv2.imread(str(args.image))
    reference = cv2.imread(str(args.reference))
    if frame is None or reference is None:
        raise ValueError('Unable to decode an input photo')
    if frame.shape != reference.shape or list(frame.shape[1::-1]) != config['image_size']:
        raise ValueError('Photo geometry does not match the empty reference')
    shift, response = cv2.phaseCorrelate(
        cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32),
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32))
    registered = response >= .3 and max(abs(v) for v in shift) <= 2
    weights_before = sha(args.weights)
    model = YOLOE(str(args.weights.resolve()))
    if list(model.names.values()) != names:
        raise ValueError('Checkpoint class order differs from taxonomy')
    policy = dict(device=args.device, imgsz=640, conf=args.confidence, iou=.7,
                  max_det=100, agnostic_nms=True, retina_masks=True, half=args.device != 'cpu',
                  verbose=False, save=False)
    result = model.predict(frame, **policy)[0]
    instances = []
    masks = []
    if len(result.boxes):
        if result.masks is None or len(result.masks.data) != len(result.boxes):
            raise ValueError('Segmentation masks are missing')
        for i, box in enumerate(result.boxes):
            cid = int(box.cls.item())
            mask = result.masks.data[i].detach().cpu().numpy() > .5
            if mask.shape != frame.shape[:2]:
                raise ValueError('Expected full-frame masks')
            masks.append(mask)
            instances.append(dict(class_id=cid, label=names[cid], material=taxonomy[cid]['material'],
                                  confidence=float(box.conf.item()),
                                  box_xyxy=[float(v) for v in box.xyxy[0].tolist()],
                                  mask_rle=encode_mask(mask)))
    if registered:
        station = annotate_station(frame, reference, instances, regions=config['regions'],
            workcell_id=args.workcell_id, reference_workcell_id=config['workcell_id'],
            alignment_verified=True, reference_sha256=hashlib.sha256(reference.tobytes()).hexdigest(),
            inside_fraction=config['inside_fraction'], changed_fraction=config['changed_fraction'],
            pixel_difference=config['pixel_difference'])
    else:
        station = dict(instances=instances, raw_instances_preserved=True,
                       filter_unavailable='Image registration gate failed', automatic_motion_allowed=False)
    evidence = dict(weight_sha256=weights_before, image_sha256=sha(args.image),
        runtime=dict(ultralytics=ultralytics.__version__, torch=torch.__version__),
        reference_image_sha256=sha(args.reference), filter_config_sha256=sha(args.filter_config),
        taxonomy_sha256=sha(args.taxonomy), workcell_id=args.workcell_id, policy=policy,
        registration=dict(gate_passed=bool(registered), shift_px=list(shift), response=response,
                          minimum_response=.3, maximum_shift_px=2.),
        raw_instances=instances, station=station, automatic_motion_allowed=False, deployed=False,
        note='Offline observation regions are not mechanical reach or commissioning evidence')
    (out / 'result.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False),
                                   encoding='utf-8')
    preview = frame.copy()
    for row, mask in zip(station['instances'], masks):
        color = (20, 220, 20) if row.get('station_grasp_candidate') else (0, 165, 255)
        preview[mask] = (preview[mask] * .65 + np.asarray(color) * .35).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(preview, contours, -1, color, 2)
        x, y = [int(v) for v in row['box_xyxy'][:2]]
        text = '%s %.2f %s' % (row['label'], row['confidence'], row.get('station_zone', 'unregistered'))
        cv2.putText(preview, text, (max(0, x), max(14, y)), cv2.FONT_HERSHEY_SIMPLEX, .42, color, 1)
    if not cv2.imwrite(str(out / 'preview.jpg'), preview):
        raise ValueError('Failed to save preview')
    if sha(args.weights) != weights_before:
        raise ValueError('Candidate changed while predicting')
    print(json.dumps(dict(output=str(out), instances=len(instances), registration_passed=bool(registered),
                          source_candidates=sum(bool(row.get('station_grasp_candidate')) for row in station['instances']),
                          automatic_motion_allowed=False), ensure_ascii=False))


if __name__ == '__main__':
    main()
