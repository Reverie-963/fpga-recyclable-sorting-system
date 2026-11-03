import copy
import unittest
from unittest.mock import Mock,patch
from src.prompted_observation import PromptedWindowedCamera,PromptedStationStatistics,single_frame_pose,frame_count


class SingleFrameTests(unittest.TestCase):
    def test_camera_requests_exactly_one_inference_without_three_frame_batch(self):
        source=Mock();source.next.return_value={'one_frame':True}
        observer=PromptedWindowedCamera(source)
        self.assertEqual(frame_count(),1)
        self.assertEqual(observer.next(6),{'one_frame':True})
        source.next.assert_called_once_with(6);source.raw.assert_not_called()

    def test_rgb_accepts_one_frame_without_claiming_persistence(self):
        stats=object.__new__(PromptedStationStatistics)
        stats.c=dict(min_bin_gain_px=10,max_other_bin_changed_fraction=.05,max_scene_changed_fraction=.05)
        stats.baseline={'id':'fixture'};stats.provenance=lambda:{'backend':'fixture'}
        evidence=dict(window_complete=False,consecutive_frames=1,regions={
            'source':dict(changed_px=15,area_px=100,persistent_px=0),
            'bin_metal':dict(changed_px=15,area_px=100,persistent_px=0),
            'static_a':dict(changed_px=0,area_px=100,persistent_px=0)})
        stats.evidence=lambda row:evidence
        source=Mock();extra=Mock()
        result=stats.verify_cycle([{}],'bin_metal',source,extra)
        self.assertEqual(result['decision_frame_count'],1);self.assertFalse(result['persistence_required'])
        self.assertFalse(result['frames'][0]['window_complete'])
        source.raw.assert_not_called();extra.assert_not_called()
        evidence['regions']['source']['changed_px']=0
        with self.assertRaisesRegex(ValueError,'change_missing'):stats.verify_cycle([{}],'bin_metal',source,extra)

    def test_single_frame_pose_preserves_real_marker_geometry_without_duplicate_evidence(self):
        from sim.test_sorting_relocated import RelocatedGeometryTests
        fixture=RelocatedGeometryTests();fixture.setUp()
        try:
            record=fixture.pose['evidence'][0]
            pose=single_frame_pose(record,fixture.camera,fixture.station)
            self.assertEqual(len(pose['evidence']),1)
            self.assertFalse(pose['stability_quality']['temporal_consistency_checked'])
            self.assertEqual(set(pose['marker_ids']),set(fixture.pose['marker_ids']))
        finally:fixture.doCleanups()
