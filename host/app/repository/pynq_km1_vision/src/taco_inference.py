"""Run the author's original TACO Mask R-CNN weights. No actuator access.

Launch by file path with the isolated Python 3.7 runtime. Paths in the config
are relative to the repository root, never to a machine-specific drive.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / 'pynq_km1_vision/config/taco_original.json'


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def encode_mask(mask):
    """Full-image row-major RLE, alternating 0/1, starting with zeros."""
    import numpy as np
    values = np.asarray(mask, dtype=np.uint8).reshape(-1)
    edges = np.flatnonzero(values[1:] != values[:-1]) + 1
    runs = np.diff(np.r_[0, edges, len(values)]).tolist()
    return ([0] if values[0] else []) + runs


def validate_sorting_mapping(cfg):
    """Require explicit coverage of all ten source labels; unknown is not Other."""
    mapping = cfg['sorting_mapping']
    classes = cfg['sorting_classes']
    if (not isinstance(mapping, dict) or set(mapping) != set(cfg['labels']) or
            not isinstance(classes, dict) or set(classes) != {'bottle', 'can', 'other'} or
            any(not isinstance(name, str) or not name.strip() for name in classes.values()) or
            any(not isinstance(target, str) or target not in classes for target in mapping.values()) or
            set(mapping.values()) != set(classes)):
        raise ValueError('Sorting mapping must cover all original labels and exactly three groups')


class TacoDetector:
    def __init__(self, config=DEFAULT_CONFIG):
        self.cfg = json.loads(Path(config).read_text(encoding='utf-8-sig'))
        cfg = self.cfg
        if cfg['schema_version'] != 'taco-original-v1' or cfg['physical_motion_enabled'] is not False:
            raise ValueError('This entry point supports observation only')
        validate_sorting_mapping(cfg)
        ratio = cfg['score_ratio_threshold']
        if type(ratio) not in (int, float) or not math.isfinite(ratio) or ratio <= 0:
            raise ValueError('Invalid score ratio threshold')
        threads = cfg['cpu_threads']
        if type(threads) is not int or not 1 <= threads <= 16:
            raise ValueError('Invalid CPU thread count')
        source = ROOT / cfg['source_dir']
        commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD']).decode().strip()
        status = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain']).decode()
        dirty = [line for line in status.splitlines() if line != '?? detector/__pycache__/']
        if commit != cfg['upstream_commit'] or dirty:
            raise ValueError('Upstream checkout differs from pinned unmodified source')
        with (source / 'detector/taco_config/map_10.csv').open(newline='') as stream:
            labels = sorted(set(row[1] for row in csv.reader(stream)))
        if labels != cfg['labels'] or len(labels) != 10:
            raise ValueError('Original class order mismatch')
        self.labels = ['BG'] + labels
        self.weights = ROOT / cfg['weights']
        self.model_id = file_hash(self.weights)
        if self.model_id != cfg['weights_sha256']:
            raise ValueError('Weight SHA256 mismatch')
        self.runtime_id = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
        os.environ['OMP_NUM_THREADS'] = str(threads)
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(source / 'detector'))
        import tensorflow as tf
        import keras
        from keras import backend as K
        from config import Config
        from model import MaskRCNN
        if tf.__version__ != '1.15.0' or keras.__version__ != '2.1.6':
            raise RuntimeError('Use the isolated legacy runtime')
        self.tf = tf
        self.graph = tf.Graph()
        session_config = tf.ConfigProto(device_count={'GPU': 0},
                                       intra_op_parallelism_threads=threads,
                                       inter_op_parallelism_threads=1)
        self.session = tf.Session(graph=self.graph, config=session_config)
        class InferenceConfig(Config):
            NAME = 'taco'
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            NUM_CLASSES = 11
            USE_OBJECT_ZOOM = False
            DETECTION_SCORE_RATIO = True
            DETECTION_MIN_CONFIDENCE = ratio
        with self.graph.as_default(), self.session.as_default():
            K.set_session(self.session)
            self.model = MaskRCNN(mode='inference', config=InferenceConfig(),
                                  model_dir=str(ROOT / 'local-only/taco-deployment/logs'))
            # The upstream entry point loads by name. Verify complete weighted
            # layer coverage and shapes first, so no head can be silently skipped.
            import h5py
            with h5py.File(str(self.weights), 'r') as saved:
                current = {layer.name: layer for layer in self.model.keras_model.layers if layer.weights}
                stored = {name.decode() if isinstance(name, bytes) else name
                          for name in saved.attrs['layer_names']
                          if len(saved[name].attrs['weight_names'])}
                if set(current) != stored:
                    raise ValueError('Weight file does not cover every model layer')
                for name, layer in current.items():
                    group = saved[name]
                    shapes = [tuple(group[key].shape) for key in group.attrs['weight_names']]
                    if shapes != [tuple(K.int_shape(weight)) for weight in layer.weights]:
                        raise ValueError('Weight shape mismatch: ' + name)
            self.model.load_weights(str(self.weights), str(self.weights), by_name=True)

    def predict(self, bgr):
        import numpy as np
        if (not isinstance(bgr, np.ndarray) or bgr.dtype != np.uint8 or
                bgr.ndim != 3 or bgr.shape[2] != 3):
            raise ValueError('Expected uint8 BGR image')
        height, width = bgr.shape[:2]
        if not (1 <= width <= 1920 and 1 <= height <= 1080):
            raise ValueError('Image exceeds 1920x1080 observation limit')
        started = time.perf_counter()
        with self.graph.as_default(), self.session.as_default():
            raw = self.model.detect([bgr[:, :, ::-1].copy()], verbose=0)[0]
        elapsed = (time.perf_counter() - started) * 1000
        instances = []
        for i, class_id in enumerate(raw['class_ids']):
            label = self.labels[int(class_id)]
            sorting_bin = self.cfg['sorting_mapping'][label]
            mask = raw['masks'][:, :, i].astype(bool)
            ys, xs = np.nonzero(mask)
            if not len(xs):
                continue
            y0, x0, y1, x1 = map(int, raw['rois'][i])
            probability = float(raw['scores'][i])
            # Keep upstream output probability and its background ratio separate.
            background = float(raw['full_scores'][i, 0])
            instances.append(dict(class_id=int(class_id), label=label,
                confidence=probability, background_score=background,
                score_ratio=probability/(background + 0.0001),
                detection_bbox_xyxy=[x0, y0, x1, y1],
                bbox_xyxy=[int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1],
                mask_rle=encode_mask(mask), area_px=int(len(xs)), sorting_bin=sorting_bin,
                sorting_label=self.cfg['sorting_classes'][sorting_bin]))
        return dict(schema_version='taco-observation-v1', instances=instances,
                    image_size=[width, height], model_version=self.model_id,
                    runtime_id=self.runtime_id, backend='tensorflow_1.15_cpu',
                    original_labels=self.labels[1:],
                    score_ratio_threshold=self.cfg['score_ratio_threshold'],
                    inference_ms=elapsed, automatic_motion_allowed=False,
                    quality_gate_passed=False, sorting_mapping_status='configured',
                    sorting_mapping=dict(self.cfg['sorting_mapping']),
                    sorting_classes=dict(self.cfg['sorting_classes']))


def annotate(image, result):
    import cv2
    import numpy as np
    output = image.copy()
    palette = {'bottle': (40, 210, 250), 'can': (70, 210, 80), 'other': (170, 80, 230)}
    for item in result['instances']:
        color = palette[item['sorting_bin']]
        flat = np.concatenate([np.full(n, i % 2, np.uint8)
                               for i, n in enumerate(item['mask_rle'])])
        mask = flat.reshape(image.shape[:2]).astype(bool)
        output[mask] = (output[mask] * .55 + np.array(color) * .45).astype(np.uint8)
        x0, y0, x1, y1 = item['bbox_xyxy']
        cv2.rectangle(output, (x0, y0), (x1-1, y1-1), color, 2)
        text = '{} / {} p={:.2f}'.format(item['sorting_bin'], item['label'], item['confidence'])
        cv2.putText(output, text, (x0, max(18, y0-7)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    import cv2
    import numpy as np
    image_bytes = args.image.read_bytes()
    frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        parser.error('Cannot decode input image')
    # A fresh output directory prevents accidental replacement of previous evidence.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    detector = TacoDetector(args.config)
    result = detector.predict(frame)
    result.update(input_image=str(args.image.resolve()), image_sha256=hashlib.sha256(image_bytes).hexdigest())
    (args.output_dir / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    cv2.imwrite(str(args.output_dir / 'overlay.jpg'), annotate(frame, result))
    print(json.dumps({k: v for k, v in result.items() if k != 'instances'}, indent=2))
    print('Instances:', [(d['label'], round(d['confidence'], 3)) for d in result['instances']])


if __name__ == '__main__':
    main()
