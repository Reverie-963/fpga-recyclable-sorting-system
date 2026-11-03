import unittest
import numpy as np
from src.prompted_region_geometry import visible_rectangle
from src.sorting_relocated_geometry import bounded_rectangle


class RegionGeometryTests(unittest.TestCase):
    def test_external_markers_use_original_guard_and_restore_detector_after_error(self):
        import copy
        from unittest.mock import patch
        from sim.test_sorting_relocated import RelocatedGeometryTests
        from src.prompted_region_geometry import geometry_for_policy
        from src import station_geometry
        fixture=RelocatedGeometryTests();fixture.setUp()
        try:
            detections=station_geometry.detect_markers(fixture.image,fixture.camera)
            args=(fixture.c,fixture.camera,fixture.station,fixture.reg,fixture.scene,fixture.pose,fixture.image)
            def visual_inputs(image,camera):return copy.deepcopy(detections)
            with patch.object(station_geometry,'detect_markers',side_effect=AssertionError('board detector unavailable')) as unavailable:
                geometry=geometry_for_policy(*args,marker_detector=visual_inputs)
                self.assertFalse(geometry.check(fixture.image)['camera_pose_recomputed'])
                self.assertIs(station_geometry.detect_markers,unavailable)
                for detection in detections:
                    detection['corners_px']=(np.asarray(detection['corners_px'])+[100,0]).tolist()
                with self.assertRaisesRegex(ValueError,'motion_camera_or_fixed_board_moved'):
                    geometry.check(fixture.image)
                self.assertIs(station_geometry.detect_markers,unavailable)
                unavailable.assert_not_called()
        finally:fixture.doCleanups()

    def test_partial_outside_region_is_clipped_on_every_edge(self):
        for points,expected in [([[-5,10],[25,10],[25,40],[-5,40]],[0,10,26,41]),
            ([[10,-5],[40,-5],[40,25],[10,25]],[10,0,41,26]),
            ([[70,10],[105,10],[105,40],[70,40]],[70,10,100,41]),
            ([[10,70],[40,70],[40,105],[10,105]],[10,70,41,100])]:
            with self.assertRaises(ValueError):bounded_rectangle(points,[100,100])
            self.assertEqual(visible_rectangle(points,[100,100]),expected)

    def test_inside_coordinates_and_inset_match_legacy_geometry(self):
        points=[[20,20],[70,20],[70,70],[20,70]]
        for inset in (False,True):
            self.assertEqual(visible_rectangle(points,[100,100],inset),bounded_rectangle(points,[100,100],inset))

    def test_inset_clipping_keeps_corners_inside_original_bin_polygon(self):
        import cv2
        points=np.asarray([[-5,10],[50,20],[45,80],[-15,70]],np.float32)
        rect=visible_rectangle(points,[100,100],True)
        self.assertGreaterEqual(rect[0],0)
        for x,y in [(rect[0],rect[1]),(rect[2]-1,rect[1]),(rect[2]-1,rect[3]-1),(rect[0],rect[3]-1)]:
            self.assertGreaterEqual(cv2.pointPolygonTest(points,(float(x),float(y)),False),0)

    def test_nonfinite_or_fully_invisible_regions_do_not_create_fake_image_areas(self):
        for points in ([[float('nan'),0],[1,1]],[[200,200],[250,200],[250,250],[200,250]]):
            with self.assertRaises(ValueError):visible_rectangle(points,[100,100])

    def test_in_frame_geometry_preserves_original_mapping_domains_and_regions(self):
        from sim.test_sorting_relocated import RelocatedGeometryTests
        from src.sorting_relocated_geometry import RelocatedGeometry
        from src.prompted_region_geometry import PromptedGeometry
        fixture=RelocatedGeometryTests();fixture.setUp()
        try:
            args=(fixture.c,fixture.camera,fixture.station,fixture.reg,fixture.scene,fixture.pose,fixture.image)
            old=RelocatedGeometry(*args);new=PromptedGeometry(*args)
            self.assertEqual(old.runtime,new.runtime)
            self.assertEqual(old.robot_domains,new.robot_domains)
            np.testing.assert_allclose(old.map('metal',[[320,180]]),new.map('metal',[[320,180]]))
        finally:fixture.doCleanups()


if __name__=='__main__':unittest.main()
