import importlib.util
from pathlib import Path
import unittest
import numpy as np
spec=importlib.util.spec_from_file_location('pair_driver',Path(__file__).resolve().parents[1]/'board/km1_station_vision.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
class Buffer(np.ndarray):
 def flush(self):pass
 def invalidate(self):pass
 def freebuffer(self):self.freed=True
class IP:
 def __init__(self):self.r={o:o+7 for o in range(0,0x94,4)}
 def read(self,o):return self.r.get(o,0)
 def write(self,o,v):self.r[o]=v
class Overlay:
 def is_loaded(self):return True
class Fake(driver.StationVisionAccel):
 def __init__(self):
  self.size=(8,6);self.poisoned=False;self.job=0;self.ip=IP();self.overlay=Overlay();self.allocations=[];self.reset_history=False
  self.base,self.frame,self.history,self.rois,self.results=[self.allocate((n,),d) for n,d in [(48,np.uint32),(48,np.uint32),(48,np.uint8),(24,np.uint32),(48,np.uint32)]]
 def allocate(self,shape,dtype):
  a=np.zeros(shape,dtype).view(Buffer);a.physical_address=4096*(len(self.allocations)+1);self.allocations.append(a);return a
 def process(self,current):
  self.copy_image(self.frame,current);a=self.base.view(np.uint8).reshape(6,8,4)[:,:,:3]
  self.history[:]=(np.abs(a.astype(np.int16)-current.astype(np.int16)).sum(2)>self.ip.read(0x70)).astype(np.uint8).ravel()
  if getattr(self,'fail',False):self.poisoned=True;raise TimeoutError('unknown DMA')
class PairTests(unittest.TestCase):
 def test_state_restored_and_scratch_reused(self):
  a=Fake();original=(a.base,a.frame,a.history,a.rois,a.results);a.history[:]=1;registers=dict(a.ip.r)
  before=np.zeros((6,8,3),np.uint8);after=before.copy();after[2,3]=16
  mask,timing=a.compare_pair(before,after,45)
  self.assertEqual(int(mask.sum()),1);self.assertTrue(timing['includes_pack_sync_readback'])
  self.assertTrue(all(x is y for x,y in zip(original,(a.base,a.frame,a.history,a.rois,a.results))))
  self.assertTrue(a.history.all());self.assertEqual(registers,a.ip.r);self.assertFalse(a.reset_history)
  count=len(a.allocations);a.compare_pair(before,after,45);self.assertEqual(len(a.allocations),count)
 def test_unknown_dma_poisoned_no_register_restore(self):
  a=Fake();registers=dict(a.ip.r);a.fail=True;img=np.zeros((6,8,3),np.uint8)
  with self.assertRaises(TimeoutError):a.compare_pair(img,img,45)
  self.assertTrue(a.poisoned);self.assertNotEqual(registers,a.ip.r)
  with self.assertRaises(RuntimeError):a.compare_pair(img,img,45)
 def test_invalid_input_no_allocation(self):
  a=Fake();count=len(a.allocations);registers=dict(a.ip.r)
  with self.assertRaises(ValueError):a.compare_pair(np.zeros((1,1,3),np.uint8),np.zeros((6,8,3),np.uint8),45)
  self.assertEqual(count,len(a.allocations));self.assertEqual(registers,a.ip.r)
