"""Exercise completion ordering and buffer ownership without hardware."""
import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


class Buffer(np.ndarray):
    def __new__(cls, shape, dtype):
        obj = np.zeros(shape, dtype=dtype).view(cls)
        obj.freed = False
        return obj

    def freebuffer(self):
        self.freed = True


class Channel:
    idle = True
    error = False

    def transfer(self, buf):
        self.buf = buf

    def wait(self):
        if not self.idle:
            raise AssertionError('wait called before idle')
        self.transferred = self.buf.nbytes


class Reg:
    def __init__(self):
        self.values={}

    def read(self, offset):
        return 2 if offset == 0 else self.values.get(offset,1)

    def write(self, offset, value):
        self.values[offset]=value


class Overlay:
    def __init__(self, path, download=True):
        self.bitfile_name=path
        self.ip_dict = {}
        self.downloads = 0
        self.loaded = True
        self.fail_reset = False
        for name in ('dma_mask', 'dma_vis_a', 'dma_vis_b'):
            self.ip_dict[name] = {'parameters': {'c_sg_length_width':'26'}}
            setattr(self, name, types.SimpleNamespace(sendchannel=Channel(), recvchannel=Channel()))
        self.mask_0 = self.visual_0 = Reg()

    def is_loaded(self):
        return self.loaded

    def download(self):
        self.downloads += 1
        if self.fail_reset:
            raise OSError('reset failed')
        # Like PYNQ 3.1.1, keep stale cached channel objects after download.


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.buffers = []
        def allocate(shape, dtype):
            b = Buffer(shape, dtype)
            self.buffers.append(b)
            return b
        fake = types.SimpleNamespace(Overlay=Overlay, allocate=allocate)
        with patch.dict(sys.modules, {'pynq':fake}):
            self.rt = importlib.import_module('km1_dma_runtime')
            self.maskmod = importlib.import_module('km1_mask_overlay')
            self.vismod = importlib.import_module('km1_visual_overlay')
        self.patches = [patch.object(self.rt,'Overlay',Overlay), patch.object(self.rt,'allocate',allocate)]
        for p in self.patches: p.start()
        self.session = self.rt.DualCoreSession('test.bit')
        self.original_overlay = self.session.overlay
        self.mask = self.maskmod.MaskAccel(session=self.session)
        self.vis = self.vismod.VisualAccel(session=self.session)

    def tearDown(self):
        for p in self.patches: p.stop()

    def test_max_frame_and_release(self):
        _, out = self.mask.run(np.zeros((480,640),np.uint8))
        self.assertEqual(out.size,307200)
        self.assertTrue(all(b.freed for b in self.buffers))
        self.assertIs(type(out),np.ndarray)

    def test_reject_old_dma_before_allocate(self):
        self.session.overlay.ip_dict['dma_mask']['parameters']['c_sg_length_width']='14'
        with self.assertRaises(ValueError): self.mask.run(np.zeros((480,640),np.uint8))
        self.assertFalse(self.buffers)

    def test_done_before_dma_timeout_and_reuse(self):
        self.session.overlay.dma_mask.sendchannel.idle=False
        with self.assertRaises(TimeoutError): self.mask.run(np.zeros((3,3),np.uint8),timeout_sec=.001)
        self.assertEqual(self.original_overlay.downloads,1)
        self.assertIsNot(self.session.overlay,self.original_overlay)
        self.assertTrue(all(b.freed for b in self.buffers))
        self.mask.run(np.zeros((3,3),np.uint8))

    def test_reset_failure_retains_buffers_and_rejects_retry(self):
        self.session.overlay.dma_mask.sendchannel.idle=False
        self.session.overlay.fail_reset=True
        with self.assertRaisesRegex(RuntimeError,'buffers retained'):
            self.mask.run(np.zeros((3,3),np.uint8),timeout_sec=.001)
        self.assertTrue(self.session.poisoned)
        self.assertTrue(all(not b.freed for b in self.buffers))
        with self.assertRaisesRegex(RuntimeError,'quarantined'): self.mask.run(np.ones((2,2),np.uint8))

    def test_error_recovers(self):
        self.session.overlay.dma_vis_b.sendchannel.error=True
        with self.assertRaisesRegex(RuntimeError,'transfer error'):
            self.vis.run(np.zeros((2,2),np.uint8),np.ones((2,2),np.uint8))
        self.assertEqual(self.original_overlay.downloads,1)
        self.assertTrue(all(b.freed for b in self.buffers))

    def test_invalid_inputs_before_allocate(self):
        for img in (np.zeros((0,1),np.uint8),np.zeros((481,1),np.uint8),np.zeros((1,641),np.uint8),
                    np.array([[256]]),np.array([[-1]]),np.zeros((1,1),float)):
            with self.assertRaises(ValueError): self.mask.run(img)
        for timeout in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError): self.mask.run(np.ones((1,1),np.uint8),timeout_sec=timeout)
        with self.assertRaises(ValueError): self.mask.run(np.ones((1,1),np.uint8),erode=2)
        with self.assertRaises(ValueError): self.vis.run(np.ones((1,1),np.uint8),np.ones((1,2),np.uint8))
        self.assertFalse(self.buffers)

    def test_allocation_failure_frees_earlier_buffer(self):
        first = Buffer((9,),np.uint32)
        with patch.object(self.rt,'allocate',side_effect=[first,MemoryError()]):
            with self.assertRaises(MemoryError): self.mask.run(np.zeros((3,3),np.uint8))
        self.assertTrue(first.freed)
        self.assertEqual(self.session.overlay.downloads,0)

    def test_exclusive_and_replaced_overlay(self):
        self.session.lock.acquire()
        with self.assertRaisesRegex(RuntimeError,'busy'): self.mask.run(np.ones((1,1),np.uint8))
        self.session.lock.release()
        self.session.overlay.loaded=False
        with self.assertRaisesRegex(RuntimeError,'replaced'): self.mask.run(np.ones((1,1),np.uint8))

    def test_short_output_is_rejected_after_completion(self):
        ch=self.session.overlay.dma_mask.recvchannel
        def short_wait(): ch.transferred=4
        ch.wait=short_wait
        with self.assertRaisesRegex(RuntimeError,'byte count'): self.mask.run(np.ones((3,3),np.uint8))
        self.assertTrue(all(b.freed for b in self.buffers))


if __name__ == '__main__':
    unittest.main()
