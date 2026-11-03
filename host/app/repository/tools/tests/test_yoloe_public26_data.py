import sys
from pathlib import Path
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from prepare_yoloe_recyclables26 import polygons


class PublicPolygonTests(unittest.TestCase):
    def annotation(self,segmentation,**override):
        return dict(dict(id=9,category_id=1,segmentation=segmentation,iscrowd=0),**override)

    def test_target_class_order_and_normalized_geometry_are_preserved(self):
        result=polygons([self.annotation([[10,20,40,20,40,60,10,60]])],{1:'Drink carton'},
                        ['Plastic straw','Drink carton'],100,100)
        self.assertEqual(result[0][0],1);self.assertEqual(result[0][2],9)
        np.testing.assert_allclose(result[0][1],[[.1,.2],[.4,.2],[.4,.6],[.1,.6]])

    def test_crowd_missing_or_invalid_target_mask_blocks_the_image(self):
        for annotation in [self.annotation([],iscrowd=1),self.annotation({'counts':'RLE'}),
                           self.annotation([[0,0,float('nan'),1,2,2]]),
                           self.annotation([[0,0,200,0,200,20]])]:
            with self.assertRaises(ValueError):polygons([annotation],{1:'Drink carton'},['Drink carton'],100,100)

    def test_disconnected_instance_is_not_silently_split_or_discarded(self):
        ann=self.annotation([[5,5,20,5,20,20,5,20],[70,70,90,70,90,90,70,90]])
        with self.assertRaises(ValueError):polygons([ann],{1:'Drink carton'},['Drink carton'],100,100)

    def test_excluded_taxonomy_is_not_relabelled_as_another_material(self):
        ann=self.annotation([[10,10,40,10,40,40,10,40]])
        self.assertEqual(polygons([ann],{1:'Glass bottle'},['Clear plastic bottle'],100,100),[])


if __name__=='__main__':unittest.main()
