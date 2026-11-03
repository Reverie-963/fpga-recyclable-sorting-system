"""Exclusive dual-core overlay owner, bounded completion and DMA buffer lifetime.

Share ONE DualCoreSession between MaskAccel and VisualAccel. Recovery resets the
whole overlay. No camera, arm, serial, flash or startup configuration is touched.
"""
import math
import operator
import threading
import time

import numpy as np
from pynq import Overlay, allocate

# Keep failed-recovery buffers alive even if a caller discards the session.
# They may only be reclaimed after an independently confirmed device reset.
_quarantined_buffers = []


def uint_arg(value, name, maximum=0xffffffff):
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError('%s must be an integer' % name) from exc
    if not 0 <= result <= maximum:
        raise ValueError('%s must be in 0..%d' % (name, maximum))
    return result


def image_u8(value):
    image = np.asarray(value)
    if image.ndim != 2 or not (1 <= image.shape[0] <= 480 and 1 <= image.shape[1] <= 640):
        raise ValueError('image shape must be (height 1..480, width 1..640)')
    if image.dtype.kind not in 'bui' or np.any(image < 0) or np.any(image > 255):
        raise ValueError('pixels must be integers in 0..255')
    return image.astype(np.uint8, copy=False)


class DualCoreSession:
    def __init__(self, bit_path='vision.bit'):
        self.overlay = Overlay(bit_path)
        self.lock = threading.Lock()
        self.poisoned = False
        self.quarantine = []

    def run(self, core_name, inputs, output_dma, registers, read_stats, timeout_sec):
        timeout_sec = float(timeout_sec)
        if not math.isfinite(timeout_sec) or timeout_sec <= 0:
            raise ValueError('timeout_sec must be finite and positive')
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('overlay is busy; serialize mask/visual jobs')
        buffers = []
        active = False
        try:
            if self.poisoned:
                raise RuntimeError('overlay recovery failed; session is quarantined')
            ol = self.overlay
            if not ol.is_loaded():
                raise RuntimeError('overlay was replaced; create a new shared session')
            reg = getattr(ol, core_name)
            senders = [getattr(ol, name).sendchannel for name, _ in inputs]
            receiver = getattr(ol, output_dma).recvchannel
            channels = senders + [receiver]
            n = inputs[0][1].size
            # Read the actual hardware width. Never hard-code the old 14-bit cap.
            for name in set([name for name, _ in inputs] + [output_dma]):
                bits = int(ol.ip_dict[name]['parameters']['c_sg_length_width'])
                if n * 4 > (1 << bits) - 1:
                    raise ValueError('frame requires %d bytes; %s has %d-bit DMA length' % (n*4, name, bits))
            for _, image in inputs:
                buf = allocate((n,), dtype=np.uint32)
                buffers.append(buf)
                buf[:] = image.reshape(-1)
            out = allocate((n,), dtype=np.uint32)
            buffers.append(out)
            deadline = time.monotonic() + timeout_sec
            for offset, value in registers.items():
                reg.write(offset, value)
            reg.read(0)  # Clear any previous AP_DONE before issuing this job.
            active = True  # Even a partially failed transfer requires recovery.
            receiver.transfer(out)
            reg.write(0, 1)
            for channel, buf in zip(senders, buffers):
                channel.transfer(buf)
            done = False
            while True:
                done = done or bool(reg.read(0) & 2)  # AP_DONE is clear-on-read.
                for channel in channels:
                    if channel.error:
                        raise RuntimeError('DMA reported a transfer error')
                if done and all(channel.idle for channel in channels):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError('DMA/core did not complete within %.3fs' % timeout_sec)
                time.sleep(0.0001)
            # All channels are confirmed idle and owned exclusively; wait only
            # performs PYNQ bookkeeping/cache invalidation, never a pending wait.
            for channel in channels:
                channel.wait()
            active = False
            if any(channel.transferred != n * 4 for channel in channels):
                raise RuntimeError('DMA completed with an unexpected byte count')
            stats = read_stats(reg)
            if stats['status'] != 1:
                raise RuntimeError('invalid accelerator status: %d' % stats['status'])
            if stats['job_id'] != registers[0x30]:
                raise RuntimeError('accelerator job ID mismatch')
            return stats, np.array(out, copy=True)
        except BaseException:
            if active:
                try:
                    # PYNQ 3.1.1 download does NOT invalidate _IPMap's cached
                    # DMA objects. Rebuild metadata without a second download.
                    self.overlay.download()
                    self.overlay = Overlay(self.overlay.bitfile_name, download=False)
                except BaseException as recovery_error:
                    self.poisoned = True
                    self.quarantine.extend(buffers)
                    _quarantined_buffers.extend(buffers)
                    buffers = []  # DMA may still own these physical addresses.
                    raise RuntimeError('recovery failed; DMA buffers retained until device reset') from recovery_error
            raise
        finally:
            try:
                for buf in buffers:
                    buf.freebuffer()
            finally:
                self.lock.release()
