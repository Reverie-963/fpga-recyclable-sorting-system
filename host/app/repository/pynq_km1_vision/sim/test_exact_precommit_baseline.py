import hashlib,unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np
from src.sorting_station_io import StationStatistics,StationCamera
from src.sorting_known import KnownController
class Tests(unittest.TestCase):
 def values(self):
  image=np.zeros((6,8,3),np.uint8);sha=hashlib.sha256(image.tobytes()).hexdigest()
  row=dict(image=image,bgr_sha256=sha,source_encoding='png',observation=dict(source_stream_id='stream',source_frame_id='42',capture_started_ms=9800))
  stats=object.__new__(StationStatistics);stats.stream='stream';stats.signature='cfg';stats.c={'max_age_ms':2000};stats.checked=Mock(side_effect=lambda x:x)
  result=dict(stream_id='stream',config_sha256='cfg',baseline_frame_id=42,baseline_bgr_sha256=sha)
  stats.request=Mock(return_value=result)
  return stats,row,result
 def test_exact_frame_uses_one_bind_and_no_second_image(self):
  stats,row,result=self.values()
  with patch('src.sorting_station_io.time.monotonic',return_value=10):
   self.assertEqual(stats.bind_exact_capture(row),result);self.assertIs(stats.baseline_row,row);self.assertEqual(stats.bind(row),result)
  self.assertEqual([c.args[0] for c in stats.request.call_args_list],['/baseline','/health'])
 def test_old_capture_rejected_before_post(self):
  stats,row,_=self.values()
  with patch('src.sorting_station_io.time.monotonic',return_value=11),self.assertRaises(ValueError):stats.bind_exact_capture(row)
  stats.request.assert_not_called()
 def test_unknown_bind_response_never_retried(self):
  stats,row,_=self.values();stats.request.side_effect=OSError('unknown response');stats.bind_next_capture=True;camera=object.__new__(StationCamera);camera.statistics=stats
  with patch('src.sorting_station_io.time.monotonic',return_value=10),self.assertRaises(OSError):camera.before_semantic(row)
  self.assertFalse(stats.bind_next_capture);self.assertEqual(stats.request.call_count,1)
 def test_changed_pixels_cannot_reuse_baseline(self):
  stats,row,_=self.values()
  with patch('src.sorting_station_io.time.monotonic',return_value=10):
   stats.bind_exact_capture(row);row['image'][:]=255
   with self.assertRaises(ValueError):stats.bind(row)
 def test_only_bound_same_frame_geometry_proof_skips_duplicate(self):
  row=dict(bgr_sha256='pixels',layout_guard_bgr_sha256='pixels',layout_guard_signature='calib',layout_evidence={})
  row['image']=np.zeros((6,8,3),np.uint8)
  stats=SimpleNamespace(backend='pl_board_local',baseline_row=row,bind=Mock(return_value={}))
  controller=SimpleNamespace(stats=stats,geometry=SimpleNamespace(signature='calib'),check_layout=Mock(),journal=SimpleNamespace(capture=Mock(return_value={})))
  KnownController.prepare_motion(controller,row,'task');controller.check_layout.assert_not_called()
  row['layout_guard_signature']='changed';KnownController.prepare_motion(controller,row,'task');controller.check_layout.assert_called_once()
