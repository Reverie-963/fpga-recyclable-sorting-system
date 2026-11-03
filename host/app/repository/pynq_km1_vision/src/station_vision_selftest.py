"""Synthetic board startup comparison, before opening the camera. No motion."""
from datetime import datetime,timezone
import hashlib
import statistics
import time
import numpy as np
from .station_vision import CPUVision


def startup_selftest(backend,config):
    if backend.backend!='pl_board_local':raise ValueError('real_pl_backend_required')
    w,h=config['image_size'];rng=np.random.default_rng(271828)
    base=rng.integers(0,256,(h,w,3),dtype=np.uint8)
    frames=[base.copy(),np.roll(base,1,axis=1),np.zeros_like(base),np.full_like(base,255)]
    cpu=CPUVision();cpu.reset(base,config['regions'],config['threshold'])
    started=time.perf_counter_ns();backend.reset(base,config['regions'],config['threshold'])
    setup_ms=(time.perf_counter_ns()-started)/1e6
    rows=[]
    for i in range(15):
        if i%3==0:cpu.begin_window();backend.begin_window()
        frame=frames[i%len(frames)]
        begin=time.perf_counter_ns();expected=cpu.process(frame);cpu_ms=(time.perf_counter_ns()-begin)/1e6
        begin=time.perf_counter_ns();actual=backend.process(frame);pl_ms=(time.perf_counter_ns()-begin)/1e6
        if actual!=expected:raise RuntimeError('station_vision_startup_integer_mismatch')
        if i>=3:rows.append(dict(cpu_ms=cpu_ms,pl_ms=pl_ms,frame_sha256=hashlib.sha256(frame.tobytes()).hexdigest()))
    cpu_median=statistics.median(r['cpu_ms'] for r in rows);pl_median=statistics.median(r['pl_ms'] for r in rows)
    return dict(schema_version='station-vision-startup-comparison-v1',
        measured_at=datetime.now(timezone.utc).isoformat(),source='synthetic_board_startup',
        backend=backend.backend,assets=backend.assets,image_size=config['image_size'],threshold=config['threshold'],
        baseline_sha256=hashlib.sha256(base.tobytes()).hexdigest(),samples=len(rows),all_equal=True,
        cpu='board_ARM_numpy_reference',cpu_median_ms=cpu_median,pl_median_ms=pl_median,
        median_speedup=cpu_median/pl_median,baseline_setup_ms=setup_ms,measurements=rows,
        scope='Board CPU reference versus PL including input packing, cache sync and result readback; excludes capture/JPEG/network.',
        cpu_utilization_measured=False,whole_system_speedup_measured=False,physical_acceptance=False,
        mechanical_commands_sent=False)
