"""Candidate FPGA frame pipeline driver; explicit paired assets and exclusive lease."""
import hashlib
import os
from pathlib import Path
import time
import numpy as np

_quarantined_buffers = []


class StationFramePipeline:
    backend = 'pl_board_local'
    pipeline_contract = 'station-frame-pipeline-v1'

    def __init__(self, bitfile, assets, size=(640, 480), load=False, preview_enabled=True):
        import fcntl
        if os.geteuid() != 0:
            raise RuntimeError('Use the privileged PYNQ runtime with xrt_setup.sh sourced')
        from pynq import Overlay, allocate
        w, h = size
        if (type(w) is not int or type(h) is not int or not 1 <= w <= 640 or not 1 <= h <= 480
                or type(preview_enabled) is not bool):
            raise ValueError('invalid_frame_pipeline_dimensions_or_preview')
        bitfile = Path(bitfile).resolve()
        # Logical keys keep the station-vision-v1 asset identity contract.
        actual = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in
            [('station_vision.bit', bitfile), ('station_vision.hwh', bitfile.with_suffix('.hwh'))]}
        if actual != assets:
            raise ValueError('frame_pipeline_assets_mismatch')
        self.size, self.assets = tuple(size), actual
        self.job, self.poisoned, self.buffers = 0, False, []
        self.preview_enabled = preview_enabled
        self.preview_image = None
        self.last_timing = self.last_processing = None
        self.lease = open('/run/lock/km1-station-vision.lock', 'a')
        try:
            fcntl.flock(self.lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.overlay = Overlay(str(bitfile), download=load)
            if not self.overlay.is_loaded():
                raise RuntimeError('reviewed_frame_pipeline_not_loaded')
            self.ip = self.overlay.frame_pipeline_0
            n = w*h
            for count in (n, n, (n+31)//32, (n+31)//32, n, 24, 88):
                self.buffers.append(allocate((count,), dtype=np.uint32))
            self.base, self.frame, self.prior, self.next, self.preview, self.rois, self.results = self.buffers
            self.prior[:] = 0
            self.next[:] = 0
            self.prior.flush()
            self.next.flush()
            for offset, buf in ((0x10, self.base), (0x20, self.frame), (0x50, self.preview),
                                (0x60, self.rois), (0x70, self.results)):
                self.address(offset, buf)
            self.ip.write(0x80, w)
            self.ip.write(0x88, h)
            self.reset_history = True
        except BaseException:
            self.close()
            raise

    def address(self, offset, buf):
        value = int(buf.physical_address)
        self.ip.write(offset, value & 0xffffffff)
        self.ip.write(offset+4, value >> 32)

    def copy_image(self, target, image):
        w, h = self.size
        if image.shape != (h, w, 3) or image.dtype != np.uint8:
            raise ValueError('expected_bgr_u8')
        packed = target.view(np.uint8).reshape(h, w, 4)
        packed[:, :, :3] = image
        packed[:, :, 3] = 0
        target.flush()

    def reset(self, baseline, regions, threshold):
        from src.station_vision import validate_regions, REGION_NAMES
        validate_regions(regions, list(self.size))
        if type(threshold) is not int or not 0 <= threshold <= 765:
            raise ValueError('invalid_threshold')
        self.copy_image(self.base, baseline)
        self.rois[:] = [v for name in REGION_NAMES for v in regions[name]]
        self.rois.flush()
        self.ip.write(0x90, threshold)
        self.begin_window()

    def begin_window(self):
        self.reset_history = True

    def read_history_words(self):
        """Acceptance-only readback, excluded from normal processing timing."""
        if self.poisoned or self.last_processing is None:
            raise RuntimeError('completed_frame_pipeline_required')
        self.prior.invalidate()
        return np.asarray(self.prior).copy()

    def process(self, current):
        from src.frame_pipeline import unpack_results
        if self.poisoned or not self.overlay.is_loaded():
            raise RuntimeError('frame_pipeline_unavailable')
        started = time.perf_counter_ns()
        self.copy_image(self.frame, current)
        packed = time.perf_counter_ns()
        self.job += 1
        if self.job >= 2**32:
            raise RuntimeError('job_id_exhausted')
        self.address(0x30, self.prior)
        self.address(0x40, self.next)
        self.ip.write(0x98, int(self.reset_history))
        self.ip.write(0xa0, int(self.preview_enabled))
        self.ip.write(0xa8, self.job)
        self.ip.read(0)
        dispatched = time.perf_counter_ns()
        self.poisoned = True
        self.ip.write(0, 1)
        deadline = time.monotonic()+2
        while not self.ip.read(0) & 2:
            if time.monotonic() > deadline:
                raise TimeoutError('frame_pipeline_timeout_no_automatic_reload')
            time.sleep(.0001)
        if self.ip.read(0xb8) != 1 or self.ip.read(0xb0) != self.job:
            raise RuntimeError('frame_pipeline_job_or_status_error')
        completed = time.perf_counter_ns()
        self.results.invalidate()
        if self.preview_enabled:
            self.preview.invalidate()
            w, h = self.size
            self.preview_image = self.preview.view(np.uint8).reshape(h, w, 4).copy()
        else:
            self.preview_image = None
        copied = time.perf_counter_ns()
        decoded = unpack_results(np.asarray(self.results))
        self.last_processing = dict(contract=self.pipeline_contract,
            regions=decoded['processing'], gray_histogram_16=decoded['gray_histogram_16'],
            preview_enabled=self.preview_enabled, automatic_motion_allowed=False)
        finished = time.perf_counter_ns()
        self.last_timing = dict(pack_and_flush_ms=(packed-started)/1e6,
            dispatch_ms=(dispatched-packed)/1e6, fpga_ddr_and_wait_ms=(completed-dispatched)/1e6,
            readback_and_preview_ms=(copied-completed)/1e6, decode_ms=(finished-copied)/1e6,
            total_ms=(finished-started)/1e6, kernel_only_measured=False)
        # No CPU touches packed history between calls. The next output buffer
        # is distinct from the next input buffer, preventing AXI read/write aliasing.
        self.prior, self.next = self.next, self.prior
        self.reset_history = False
        self.poisoned = False
        return decoded['regions']

    def close(self):
        if self.poisoned:
            _quarantined_buffers.extend(self.buffers)
            self.buffers = []
            # Keep the lease with quarantined DMA-visible storage after timeout.
            if getattr(self, 'lease', None):
                _quarantined_buffers.append(self.lease)
                self.lease = None
            return
        for buf in self.buffers:
            buf.freebuffer()
        self.buffers = []
        if getattr(self, 'lease', None):
            self.lease.close()
            self.lease = None
