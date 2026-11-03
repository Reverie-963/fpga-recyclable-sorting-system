import copy,hashlib,unittest
import numpy as np
from src.phase_dataflow import PreparedFrameCamera
class Source:
 def __init__(self):
  self.c=dict(camera_id='cam',model_version='weight',runtime_id='cfg',image_size=[8,6],prompt_mode='text',max_age_ms=1000);self.calls=0;self.barrier=None
 def next(self,timeout):self.calls+=1;return {'fallback':True}
 def after(self,stamp):self.barrier=stamp
 def raw(self,timeout):return {'raw':True}
def row(source):
 image=np.zeros((6,8,3),np.uint8)
 return dict(image=image,bgr_sha256=hashlib.sha256(image.tobytes()).hexdigest(),observation=dict(source.c,frame_id=1,source_frame_id='42',source_stream_id='stream',capture_started_ms=9500,instances=[]))
class Tests(unittest.TestCase):
 def test_one_verified_result_reused_without_timestamp_refresh(self):
  s=Source();checks=[];c=PreparedFrameCamera(s,row(s),lambda:checks.append(True),lambda:10)
  r=c.next(2);self.assertTrue(c.used);self.assertEqual(r['observation']['capture_started_ms'],9500);self.assertEqual(s.calls,0);self.assertEqual(checks,[True]);self.assertEqual(s.previous,42);self.assertEqual(s.sequence,1)
  self.assertTrue(c.next(2)['fallback']);self.assertEqual(s.calls,1)
 def test_stale_future_wrong_model_fall_back(self):
  for change in (lambda r:r['observation'].update(capture_started_ms=8000),lambda r:r['observation'].update(capture_started_ms=11000),lambda r:r['observation'].update(model_version='other')):
   s=Source();r=row(s);change(r);c=PreparedFrameCamera(s,r,lambda:None,lambda:10);self.assertTrue(c.next(2)['fallback']);self.assertFalse(c.used)
 def test_camera_policy_updates_reach_the_underlying_raw_reader(self):
  s=Source();c=PreparedFrameCamera(s,row(s),lambda:None,lambda:10)
  c.c=dict(s.c,max_age_ms=800);c.post_retreat_count=None
  self.assertEqual(s.c['max_age_ms'],800);self.assertIsNone(s.post_retreat_count)
 def test_barrier_discards_prepared_frame(self):
  s=Source();c=PreparedFrameCamera(s,row(s),lambda:None,lambda:10);c.after(9999);self.assertTrue(c.next(2)['fallback']);self.assertEqual(s.barrier,9999)
 def test_model_generation_error_is_not_swallowed(self):
  def changed():raise ValueError('generation changed')
  s=Source();c=PreparedFrameCamera(s,row(s),changed,lambda:10)
  with self.assertRaisesRegex(ValueError,'generation changed'):c.next(2)
  self.assertEqual(s.calls,0)
 def test_pixels_changed_rejected_and_source_row_is_copied(self):
  s=Source();r=row(s);c=PreparedFrameCamera(s,r,lambda:None,lambda:10);r['image'][:]=1;self.assertFalse(c.next(2)['image'].any())
  c=PreparedFrameCamera(s,row(s),lambda:None,lambda:10);c.seed['image'][:]=1
  with self.assertRaisesRegex(ValueError,'pixels_changed'):c.next(2)
