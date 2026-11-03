"""Board-only candidate array acceptance. Explicit load; never opens a camera/arm."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
if (ROOT/'src').is_dir():
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT/'board'))
else:
    sys.path.insert(0, str(ROOT.parent/'pynq_km1_vision'))
    sys.path.insert(0, str(ROOT.parent/'pynq_km1_vision/board'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image', type=Path)
    parser.add_argument('--load-overlay', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh_probe_output_required')
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    if config.get('pipeline_contract') != 'station-frame-pipeline-v1':
        raise ValueError('frame_pipeline_contract_required')
    from km1_frame_pipeline import StationFramePipeline
    from src.frame_pipeline_selftest import startup_selftest
    from src.frame_pipeline import process_frame, PLANES
    backend = StationFramePipeline((args.config.parent/config['bitfile']).resolve(), config['assets'],
        tuple(config['image_size']), load=args.load_overlay, preview_enabled=True)
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        report = startup_selftest(backend, config)
        if args.image:
            frame = cv2.imread(str(args.image))
            if frame is None or list(frame.shape[1::-1]) != config['image_size']:
                raise ValueError('real_scene_image_dimensions_mismatch')
            base = np.zeros_like(frame)
            expected = process_frame(base, frame, config['regions'], config['threshold'])
            backend.reset(base, config['regions'], config['threshold'])
            actual = backend.process(frame)
            if actual != expected['regions'] or not np.array_equal(backend.preview_image, expected['preview']):
                raise RuntimeError('real_scene_fpga_cpu_pixel_mismatch')
            for i, name in enumerate(PLANES):
                if not cv2.imwrite(str(args.output/(name+'.png')), backend.preview_image[:, :, i]):
                    raise RuntimeError('cannot_write_verified_plane')
            report['real_scene_replay'] = dict(input_bgr_sha256=hashlib.sha256(frame.tobytes()).hexdigest(),
                all_four_planes_equal=True, regions_equal=True, phases=backend.last_timing,
                source='recorded_scene_with_zero_diagnostic_baseline_not_task_evidence')
        (args.output/'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps({k: report[k] for k in ('all_equal', 'samples', 'cpu_median_ms',
            'pl_median_ms', 'median_speedup', 'kernel_only_measured', 'mechanical_commands_sent')}))
    finally:
        backend.close()


if __name__ == '__main__':
    main()
