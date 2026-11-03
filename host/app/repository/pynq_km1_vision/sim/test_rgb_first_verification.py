import copy,hashlib,unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
from sim.test_bin_occupancy import OccupancyIntegrationTests
class Tests(unittest.TestCase):
 fixture=OccupancyIntegrationTests.fixture
 def setup_controller(self,change=True):
  f=self.fixture();c=f.factory(0);before=c.acquire();after=f.source.next(2)
  after['observation'].pop('instances',None)
  for k in ('model_version','runtime_id','image_size','prompt_mode'):after['observation'].pop(k,None)
  if change:
   x,y,_,_=c.c['profiles']['metal']['bin_roi'];after['image'][y+5:y+15,x+5:x+15]=255
  after['bgr_sha256']=hashlib.sha256(after['image'].tobytes()).hexdigest();after['source_encoding']='png'
  detector=SimpleNamespace(verify_identity=Mock(),local_marker_detector=Mock())
  raw=Mock(return_value=after);c.source=SimpleNamespace(source=SimpleNamespace(detector=detector,raw=raw,c=c.c),after=Mock())
  c.geometry.check=Mock(return_value={});c.policy['rgb_first_verification']=True
  return f,c,before,after,detector
 def test_visible_change_skips_semantic_acquisition(self):
  f,c,before,after,det=self.setup_controller();c.verification_frames=Mock(side_effect=AssertionError('semantic call forbidden'))
  rows=[];result=c.verify_within_window(before,({'label':'metal'},None),{'bin_id':'bin_metal'},before['observation']['capture_started_ms'],rows)
  self.assertTrue(result['dataflow']['semantic_inference_skipped']);self.assertEqual(result['status'],'bin_occupied_pass');self.assertFalse(result['class_verified']);self.assertEqual(det.verify_identity.call_count,2);self.assertEqual(rows[0]['perception_kind'],'raw_rgb_only_no_model_inference')
 def test_stale_or_identity_failure_never_becomes_success(self):
  f,c,before,after,det=self.setup_controller();after['observation']['capture_started_ms']-=30000
  with self.assertRaisesRegex(ValueError,'stale_or_future_frame'):c.verify_rgb_first(before,({'label':'metal'},None),{'bin_id':'bin_metal'},0,[],c.clock()+10)
  self.assertEqual(det.verify_identity.call_count,1)
 def test_no_change_returns_to_existing_semantic_window(self):
  f,c,before,after,det=self.setup_controller(False)
  c.verification_frames=Mock(side_effect=RuntimeError('original semantic window reached'))
  with self.assertRaisesRegex(RuntimeError,'original semantic window reached'):c.verify_within_window(before,({'label':'metal'},None),{'bin_id':'bin_metal'},0,[])
  self.assertEqual(c.verification_attempts[0]['reason'],'unified_target_bin_empty')

 def test_rgb_empty_uses_same_validated_frame_for_semantic_or_branch(self):
  f,c,before,after,det=self.setup_controller(False)
  import numpy as np
  mask=np.zeros(after['image'].shape[:2],bool);x,y,_,_=c.c['profiles']['metal']['bin_roi'];mask[y+5:y+13,x+5:x+13]=True
  flat=mask.ravel();runs=[];last=False;n=0
  for value in flat:
   if bool(value)==last:n+=1
   else:runs.append(n);n=1;last=bool(value)
  runs.append(n)
  obs={k:c.c[k] for k in ('model_version','runtime_id','image_size','prompt_mode')};obs.update(image_sha256=after['observation']['image_sha256'],instances=[dict(label='plastic',confidence=.21,bbox_xyxy=[x+5,y+5,x+13,y+13],mask_rle=runs)])
  det.predict_frame=Mock(return_value=obs);c.verification_frames=Mock(side_effect=AssertionError('new capture forbidden'))
  rows=[];r=c.verify_within_window(before,({'label':'metal'},None),{'bin_id':'bin_metal'},0,rows)
  self.assertEqual(r['basis'],'any_bin_detection');self.assertTrue(r['dataflow']['same_frame_fallback']);self.assertFalse(r['dataflow']['semantic_inference_skipped']);self.assertEqual(det.predict_frame.call_count,1);self.assertEqual(c.source.source.raw.call_count,1)
