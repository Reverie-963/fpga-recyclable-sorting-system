"""Full-workload CPU/FPGA comparison before camera capture. No actuators."""
from datetime import datetime, timezone
import hashlib
import statistics
import time
import numpy as np

from .frame_pipeline import process_frame, PLANES


def startup_selftest(backend, config):
    if backend.backend != 'pl_board_local' or backend.pipeline_contract != 'station-frame-pipeline-v1':
        raise ValueError('real_frame_pipeline_backend_required')
    w, h = config['image_size']
    rng = np.random.default_rng(271828)
    base = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    frames = [base.copy(), np.roll(base, 1, axis=1), np.zeros_like(base), np.full_like(base, 255)]
    started = time.perf_counter_ns()
    backend.reset(base, config['regions'], config['threshold'])
    setup_ms = (time.perf_counter_ns()-started)/1e6
    rows, history = [], None
    for i in range(15):
        if i % 3 == 0:
            history = None
            backend.begin_window()
        frame = frames[i % len(frames)]
        started = time.perf_counter_ns()
        expected = process_frame(base, frame, config['regions'], config['threshold'], history,
                                 preview_enabled=backend.preview_enabled)
        cpu_ms = (time.perf_counter_ns()-started)/1e6
        history = expected['history']
        started = time.perf_counter_ns()
        actual = backend.process(frame)
        pl_ms = (time.perf_counter_ns()-started)/1e6
        if actual != expected['regions']:
            raise RuntimeError('frame_pipeline_original_rgb_mismatch')
        processing = backend.last_processing
        if (processing['regions'] != expected['processing'] or
                processing['gray_histogram_16'] != expected['gray_histogram_16']):
            raise RuntimeError('frame_pipeline_summary_mismatch')
        if backend.preview_enabled and not np.array_equal(backend.preview_image, expected['preview']):
            raise RuntimeError('frame_pipeline_preview_pixel_mismatch')
        bits = np.pad(history.ravel(), (0, (-w*h) % 32))
        packed_history = np.packbits(bits, bitorder='little').view(np.uint32)
        if not np.array_equal(backend.read_history_words(), packed_history):
            raise RuntimeError('frame_pipeline_packed_history_mismatch')
        if i >= 3:
            rows.append(dict(cpu_ms=cpu_ms, pl_ms=pl_ms, phases=dict(backend.last_timing),
                frame_sha256=hashlib.sha256(frame.tobytes()).hexdigest(),
                preview_sha256=hashlib.sha256(backend.preview_image.tobytes()).hexdigest()
                    if backend.preview_enabled else None))
    cpu = statistics.median(r['cpu_ms'] for r in rows)
    pl = statistics.median(r['pl_ms'] for r in rows)
    return dict(schema_version='frame-pipeline-startup-comparison-v1',
        measured_at=datetime.now(timezone.utc).isoformat(), source='synthetic_frame_pipeline_startup',
        backend='pl_board_local', assets=backend.assets, image_size=[w, h], threshold=config['threshold'],
        cpu='board_ARM_OpenCV_NumPy_full_pipeline', cpu_median_ms=cpu, pl_median_ms=pl,
        median_speedup=cpu/pl, samples=len(rows), all_equal=True,
        preview_enabled=backend.preview_enabled, planes=list(PLANES),
        packed_history_equal=True, acceptance_readback_excluded_from_timing=True,
        baseline_sha256=hashlib.sha256(base.tobytes()).hexdigest(), baseline_setup_ms=setup_ms,
        measurements=rows, kernel_only_measured=False, whole_system_speedup_measured=False,
        mechanical_commands_sent=False,
        scope='Full RGB evidence, gray, 3x3 mean, Sobel, binary opening, six-region summaries and histogram; '
              'FPGA includes packing, cache sync, DDR wait and readback; excludes capture/JPEG/network.')
