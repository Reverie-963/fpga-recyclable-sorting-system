"""PYNQ driver for motion-path-v1. Constructor never loads a bitstream."""
import math
import secrets
import threading
import time
import xml.etree.ElementTree as ET

from .motion_path import MAX_SEGMENTS, checked_segments, decode_results

_quarantined_buffers = []


def validate_hwh(path):
    modules = [m for m in ET.parse(path).getroot().iter('MODULE')
               if m.get('INSTANCE') == 'motion_path_0']
    if len(modules) != 1 or modules[0].get('VLNV') != 'xilinx.com:hls:motion_path_accel:1.0':
        raise ValueError('motion_hwh_wrong_core')
    offsets = {}
    for register in modules[0].iter('REGISTER'):
        values = {p.get('NAME'): p.get('VALUE') for p in register.findall('PROPERTY')}
        name = register.get('NAME', '')
        if name.endswith('_1'):
            name = name[:-2]
        if 'ADDRESS_OFFSET' in values:
            offsets[name] = int(values['ADDRESS_OFFSET'], 0)
    for name, expected in {'input_words': 0x10, 'input_words_2': 0x14,
                           'output_words': 0x1c, 'output_words_2': 0x20,
                           'segment_count': 0x28, 'request_tag': 0x30}.items():
        if offsets.get(name) != expected:
            raise ValueError('motion_hwh_register_layout_changed:'+name)
    return offsets


class PLMotionPath:
    backend = 'pl_motion_path_v1'

    def __init__(self, ip, allocator=None, timeout_s=2, identity_check=None):
        if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or not 0 < timeout_s <= 10:
            raise ValueError('motion_timeout_outside_0_10_s')
        self.lease = None
        if allocator is None:
            import fcntl
            from pynq import allocate
            allocator = allocate
            self.lease = open('/run/lock/km1-motion-path.lock', 'a')
            try:
                fcntl.flock(self.lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException:
                self.lease.close()
                raise
        self.ip = ip
        self.timeout_s = timeout_s
        self.identity_check = identity_check or (lambda: None)
        self.lock = threading.Lock()
        self.faulted = False
        self.closed = False
        self.active = False
        self.quarantined = False
        self.tag = secrets.randbelow(0xfffffffe)+1
        try:
            self.input = allocator(shape=(MAX_SEGMENTS*12,), dtype='uint32')
        except BaseException:
            if self.lease:
                self.lease.close()
            raise
        try:
            self.output = allocator(shape=(MAX_SEGMENTS*8,), dtype='uint32')
        except BaseException:
            self.input.freebuffer()
            if self.lease:
                self.lease.close()
            raise

    def evaluate(self, segments):
        started = time.perf_counter_ns()
        segments = checked_segments(segments)
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('motion_accelerator_busy')
        try:
            if self.closed or self.faulted:
                raise RuntimeError('motion_accelerator_closed_or_faulted')
            self.identity_check()
            control = self.ip.read(0)
            if not control & 4 or control & 0x80:
                raise RuntimeError('motion_accelerator_not_idle_or_auto_restart')
            self.tag = self.tag % 0xffffffff+1
            flat = [v for segment in segments for endpoint in segment for v in endpoint]
            self.input[:len(flat)] = flat
            self.output[:len(segments)*8] = [0]* (len(segments)*8)
            packed = time.perf_counter_ns()
            self.input.flush()
            self.output.flush()
            synced = time.perf_counter_ns()
            for offset, address in ((0x10, self.input.physical_address), (0x1c, self.output.physical_address)):
                self.ip.write(offset, address & 0xffffffff)
                self.ip.write(offset+4, address >> 32)
            self.ip.write(0x28, len(segments))
            self.ip.write(0x30, self.tag)
            # From this point an error may leave DMA active: retain allocations.
            self.active = True
            self.ip.write(0, 1)
            deadline = time.monotonic()+self.timeout_s
            while True:
                if self.ip.read(0) & 2:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError('motion_accelerator_timeout_buffers_retained')
            self.active = False
            waited = time.perf_counter_ns()
            self.output.invalidate()
            words = self.output[:len(segments)*8].tolist()
            rows = decode_results(words, segments, self.tag)
            self.identity_check()
            finished = time.perf_counter_ns()
            return dict(backend=self.backend, segments=rows, timing=dict(
                total_ms=(finished-started)/1e6, validation_pack_ms=(packed-started)/1e6,
                cache_sync_ms=(synced-packed)/1e6, submit_wait_ms=(waited-synced)/1e6,
                readback_decode_ms=(finished-waited)/1e6,
                excludes='one-time allocation, overlay loading, network, route generation and mechanical motion'))
        except BaseException:
            self.faulted = True
            if self.active and not self.quarantined:
                _quarantined_buffers.append((self.input, self.output, self.lease))
                self.quarantined = True
            raise
        finally:
            self.lock.release()

    def close(self):
        with self.lock:
            if self.closed:
                return
            if self.active:
                if not self.ip.read(0) & 4:
                    raise RuntimeError('motion_dma_active_buffers_must_remain_allocated')
                self.active = False
            self.output.freebuffer()
            self.input.freebuffer()
            if self.quarantined:
                _quarantined_buffers[:] = [pair for pair in _quarantined_buffers
                    if pair[0] is not self.input]
                self.quarantined = False
            if self.lease:
                self.lease.close()
            self.closed = True
