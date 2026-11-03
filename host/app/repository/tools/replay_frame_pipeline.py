"""Replay a recorded scene through the frozen HLS C model; no board programming."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'pynq_km1_vision'))
from src.frame_pipeline import process_frame, result_words, PLANES
from src.station_vision import REGION_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tool-root', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT/'local-only') or any(c.isspace() or ord(c)>126 for c in str(output)):
        raise ValueError('fresh_ascii_local_only_output_required')
    image = cv2.imread(str(args.image))
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    if image is None or list(image.shape[1::-1]) != config['image_size']:
        raise ValueError('recorded_scene_size_mismatch')
    output.mkdir(parents=True, exist_ok=False)
    h, w = image.shape[:2]
    # Zero is a diagnostic reference, never a task baseline or empty-bin proof.
    base = np.zeros_like(image)
    frames = [image, np.roll(image, 1, axis=1), np.roll(image, 3, axis=0)]
    def packed(frame):
        data = np.zeros((h, w, 4), np.uint8)
        data[:, :, :3] = frame
        return data.view('<u4').reshape(-1)
    header = np.asarray([w, h, config['threshold'], len(frames)] +
        [v for name in REGION_NAMES for v in config['regions'][name]], dtype='<u4')
    input_path, result_path = output/'input.bin', output/'actual.bin'
    with input_path.open('wb') as stream:
        header.tofile(stream)
        packed(base).tofile(stream)
        for frame in frames:
            packed(frame).tofile(stream)
    source = (args.build/'sources/station_frame_pipeline.cpp').resolve()
    testbench = ROOT/'pynq_km1_vision/src/hls/station_frame_pipeline_replay.cpp'
    (output/'station_frame_pipeline.cpp').write_bytes(source.read_bytes())
    (output/'station_frame_pipeline_replay.cpp').write_bytes(testbench.read_bytes())
    cfg = output/'replay.cfg'
    cfg.write_text('\n'.join(['part=xc7z020clg400-1', '[hls]', 'flow_target=vivado',
        'syn.top=station_frame_pipeline', 'syn.file='+str(output/'station_frame_pipeline.cpp').replace('\\','/'),
        'tb.file='+str(output/'station_frame_pipeline_replay.cpp').replace('\\','/'),
        'csim.argv='+str(input_path).replace('\\','/')+' '+str(result_path).replace('\\','/')])+'\n',
        encoding='ascii')
    command = [str(args.tool_root/'Vitis/bin/vitis-run.bat'), '--mode', 'hls', '--config',
               str(cfg), '--work_dir', str(output/'work'), '--csim']
    with (output/'csim.log').open('w', encoding='utf-8') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    actual = np.fromfile(result_path, dtype='<u4')
    history, rows, offset = None, [], 0
    n, history_words = w*h, (w*h+31)//32
    if actual.size != len(frames)*(88+history_words+n):
        raise RuntimeError('recorded_scene_output_size_mismatch')
    for i, frame in enumerate(frames):
        expected = process_frame(base, frame, config['regions'], config['threshold'], history)
        history = expected['history']
        words = result_words(expected)
        packed_history = np.packbits(history.ravel(), bitorder='little')
        padded = np.zeros(history_words*4, np.uint8)
        padded[:packed_history.size] = packed_history
        reference = np.concatenate((words, padded.view('<u4'), expected['preview'].view('<u4').ravel()))
        count = len(reference)
        if not np.array_equal(actual[offset:offset+count], reference):
            raise RuntimeError('recorded_scene_hls_cpu_mismatch_frame_'+str(i))
        preview = actual[offset+88+history_words:offset+count].view(np.uint8).reshape(h,w,4)
        if i == 0:
            cv2.imwrite(str(output/'original.png'), frame)
            for j, name in enumerate(PLANES):
                cv2.imwrite(str(output/(name+'.png')), preview[:, :, j])
        rows.append(dict(input_bgr_sha256=hashlib.sha256(frame.tobytes()).hexdigest(),
                         exact_words=count, all_equal=True))
        offset += count
    report = dict(source='recorded_scene_HLS_C_simulation_not_board_execution', all_equal=True,
        measured_at=datetime.now(timezone.utc).isoformat(),
        image_size=[w,h], frames=rows, planes=list(PLANES),
        hls_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        testbench_sha256=hashlib.sha256(testbench.read_bytes()).hexdigest(),
        cpu_reference_sha256=hashlib.sha256((ROOT/'pynq_km1_vision/src/frame_pipeline.py').read_bytes()).hexdigest(),
        image_file_sha256=hashlib.sha256(args.image.read_bytes()).hexdigest(),
        baseline='diagnostic_zero_not_task_evidence', board_programmed=False,
        mechanical_commands_sent=False, speedup_measured=False)
    (output/'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(status='recorded_scene_hls_cpu_equal', frames=len(frames),
                         compared_u32_words=int(actual.size), output=str(output))))


if __name__ == '__main__':
    main()
