import copy
import hashlib
import unittest
import numpy as np
from src.station_instance_filter import annotate_station
from src.yoloe_inference import encode_mask


class StationFilterTests(unittest.TestCase):
    def setUp(self):
        self.ref=np.full((40,60,3),100,np.uint8)
        self.options=dict(regions={'source':[0,0,30,30],'bin_paper':[35,0,60,30]},
            workcell_id='fixture-A',reference_workcell_id='fixture-A',alignment_verified=True,
            reference_sha256=hashlib.sha256(self.ref.tobytes()).hexdigest())

    def instance(self,x0,y0,x1,y1):
        mask=np.zeros((40,60),bool);mask[y0:y1,x0:x1]=True
        return dict(label='Drink carton',confidence=.7,mask_rle=encode_mask(mask))

    def test_static_source_false_alarm_is_annotated_without_removing_obstacle(self):
        objects=[self.instance(5,5,20,20)];before=copy.deepcopy(objects)
        result=annotate_station(self.ref,self.ref,objects,**self.options)
        self.assertFalse(result['instances'][0]['station_grasp_candidate'])
        self.assertIn('matches_registered_empty_background',result['instances'][0]['station_filter_reasons'])
        self.assertEqual(objects,before)
        self.assertEqual(result['instances'][0]['mask_rle'],objects[0]['mask_rle'])

    def test_new_source_object_is_retained_but_motion_is_never_granted(self):
        image=self.ref.copy();image[5:20,5:20]=200
        row=annotate_station(image,self.ref,[self.instance(5,5,20,20)],**self.options)['instances'][0]
        self.assertTrue(row['station_grasp_candidate']);self.assertFalse(row['automatic_motion_allowed'])

    def test_bin_target_is_preserved_for_verification_and_not_a_source_candidate(self):
        image=self.ref.copy();image[5:20,40:55]=200
        row=annotate_station(image,self.ref,[self.instance(40,5,55,20)],**self.options)['instances'][0]
        self.assertEqual(row['station_zone'],'bin_paper');self.assertFalse(row['station_grasp_candidate'])
        self.assertTrue(row['obstacle_mask_preserved'])

    def test_reference_geometry_alignment_and_invalid_masks_are_rejected(self):
        for override in [dict(workcell_id='other'),dict(alignment_verified=False),dict(reference_sha256='bad')]:
            with self.assertRaises(ValueError):annotate_station(self.ref,self.ref,[],**dict(self.options,**override))
        with self.assertRaises(ValueError):
            annotate_station(self.ref,self.ref,[dict(mask_rle=[1])],**self.options)

    def test_polygon_bin_does_not_claim_empty_space_in_its_bounding_rectangle(self):
        options=dict(self.options,regions={'source':[0,0,60,40],'bin_paper':[[35,0],[59,0],[59,24]]})
        image=self.ref.copy();image[15:20,36:40]=200
        row=annotate_station(image,self.ref,[self.instance(36,15,40,20)],**options)['instances'][0]
        self.assertEqual(row['station_zone'],'source');self.assertTrue(row['station_grasp_candidate'])


if __name__=='__main__':unittest.main()
