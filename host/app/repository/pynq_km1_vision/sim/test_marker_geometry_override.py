import unittest
from unittest.mock import Mock,patch
from src import prompted_region_geometry as geometry,station_geometry


class MarkerGeometryOverrideTests(unittest.TestCase):
    def build(self,default):
        class Base:
            def __init__(self):self.signature='base';self.check('initial')
            def check(self,image):return station_geometry.detect_markers(image,{'camera':'fixed'})
        with patch.object(geometry,'RelocatedGeometry',Base),patch.object(geometry,'read_motion_options',return_value={}):
            return geometry.geometry_for_policy(marker_detector=default)

    def test_marker_only_override_reaches_existing_external_wrapper(self):
        original=station_geometry.detect_markers;default=Mock(return_value=['semantic']);marker_only=Mock(return_value=['markers'])
        obj=self.build(default)
        self.assertEqual(obj.check('after',marker_detector=marker_only),['markers'])
        marker_only.assert_called_once_with('after',{'camera':'fixed'})
        self.assertEqual(obj.check('next'),['semantic']);self.assertEqual(default.call_count,2)
        self.assertIs(station_geometry.detect_markers,original)

    def test_failed_override_restores_original_detector(self):
        original=station_geometry.detect_markers;obj=self.build(Mock(return_value=[]))
        with self.assertRaisesRegex(ValueError,'identity_changed'):
            obj.check('after',marker_detector=Mock(side_effect=ValueError('identity_changed')))
        self.assertIs(station_geometry.detect_markers,original)


if __name__=='__main__':unittest.main()
