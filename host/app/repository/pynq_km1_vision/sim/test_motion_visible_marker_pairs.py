import copy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import cv2
import numpy as np
from src.sorting_relocated_geometry import RelocatedGeometry
from src.sorting_relocated import RelocatedMotion
from src.sorting_scene import check_motion_layout

class VisiblePairTests(unittest.TestCase):
    def fixture(self):
        g=RelocatedGeometry.__new__(RelocatedGeometry)
        image=np.full((300,300,3),255,np.uint8);rows=[]
        for sid,(x,y) in zip((10,11,12,13),((40,40),(180,40),(40,180),(180,180))):
            marker=cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),sid,40)
            image[y:y+40,x:x+40]=marker[:,:,None]
            rows.append(dict(id=sid,corners_px=[[x,y],[x+39,y],[x+39,y+39],[x,y+39]]))
        layout=dict(static_rois=[[34,34,86,86],[34,174,86,226]],search_margin_px=5,max_layout_shift_px=1,
            min_patch_std=8,min_patch_corners=4,min_patch_peak_margin=.01,min_patch_correlation=.98,
            max_patch_mean_difference=12,min_motion_patch_dynamic_range=80)
        g.reference=image;g.runtime=dict(layout=layout,image_size=[300,300]);g.camera={}
        g.signature='frozen';g.min_fixed_markers=2;g.motion_shift_px=1
        g.expected={r['id']:np.array(r['corners_px'],float) for r in rows}
        g.pose=dict(marker_ids=[10,11,12,13],evidence=[dict(detections=copy.deepcopy(rows))])
        return g,rows

    def test_arm_occluding_left_patch_uses_two_visible_registered_marks(self):
        g,rows=self.fixture();current=g.reference.copy();current[170:235,25:95]=128
        with self.assertRaises(ValueError):
            check_motion_layout(g.reference,current,dict(g.runtime['layout'],image_size=[300,300]))
        with patch('src.station_geometry.detect_markers',return_value=[r for r in rows if r['id']!=12]):
            result=g.check(current)
        self.assertEqual(len(result['photometric_marker_ids']),2)
        self.assertNotIn(12,result['photometric_marker_ids'])
        self.assertFalse(result['camera_pose_recomputed'])
        self.assertTrue(all(m['correlation']>=.98 for m in result['static']['markers']))

    def test_one_mark_global_motion_and_all_bad_patches_still_reject(self):
        g,rows=self.fixture()
        with patch('src.station_geometry.detect_markers',return_value=rows[:1]):
            with self.assertRaisesRegex(ValueError,'two_fixed'):g.check(g.reference)
        moved=copy.deepcopy(rows)
        for r in moved:r['corners_px']=(np.array(r['corners_px'])+[3,0]).tolist()
        with patch('src.station_geometry.detect_markers',return_value=moved):
            with self.assertRaisesRegex(ValueError,'fixed_board_moved'):g.check(g.reference)
        with patch('src.station_geometry.detect_markers',return_value=rows):
            with self.assertRaisesRegex(ValueError,'moved_occluded_or_ambiguous'):g.check(255-g.reference)
        self.assertEqual(len(g.last_patch_attempts),6)

    def test_completed_wait_advances_camera_barrier_but_failed_wait_does_not(self):
        m=RelocatedMotion.__new__(RelocatedMotion);m.camera=Mock()
        with patch('src.sorting_io.Motion.wait',return_value=None) as waited,patch('src.sorting_relocated.time.monotonic',return_value=12.5):
            m.wait(1)
        waited.assert_called_once_with(1);m.camera.after.assert_called_once_with(12500)
        m.camera.after.reset_mock()
        with patch('src.sorting_io.Motion.wait',side_effect=TimeoutError('uncertain')):
            with self.assertRaises(TimeoutError):m.wait(1)
        m.camera.after.assert_not_called()

    def test_frame_from_previous_motion_cannot_reach_uart_write(self):
        m=RelocatedMotion.__new__(RelocatedMotion);m.last_motion_completed_ms=2000
        m.check_stop=lambda:None;m.expected_stream=lambda:'fixed';m.guard_index=0
        m.c=dict(frame_timeout_s=2,max_age_ms=2000);m.geometry=Mock();m.journal=Mock()
        m.journal.pending.return_value={'task_id':'recovery-proof'}
        m.camera=Mock();m.camera.raw.return_value=dict(image=np.zeros((10,10,3),np.uint8),
            observation=dict(source_stream_id='fixed',source_frame_id='1',capture_started_ms=1999))
        written=[]
        def transport(_motion,data,duration,guard):guard();written.append(data)
        with patch('src.sorting_io.Motion.send',side_effect=transport):
            with self.assertRaisesRegex(ValueError,'precedes_previous_motion_end'):m.send(b'not-sent',500)
        self.assertEqual(written,[]);m.geometry.check.assert_not_called()
