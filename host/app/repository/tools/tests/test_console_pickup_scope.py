import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).parents[1]))
from console_pickup_scope import PickupScope


def runs(rect):
    mask=np.zeros((40,60),bool);x,y,x1,y1=rect;mask[y:y1,x:x1]=True
    result=[0];previous=False
    for bit in mask.ravel():
        if bit==previous:result[-1]+=1
        else:result.append(1);previous=bit
    return result


class ScopeTests(unittest.TestCase):
    def setUp(self):
        self.layout=dict(image_size=[60,40],source_roi=[10,10,30,30],bins=[[40,10,55,30]],bin_labels=['metal'],static_rois=[])
        self.state=dict(board_root='/runtime',min_fixed_markers=2,execution_model=dict(model_sha256='a'*64,configuration_sha256='b'*64,profile_sha256='c'*64,revision=7))
        self.capture_calls=[]
        def capture():
            self.capture_calls.append('frame');return b'png',np.zeros((40,60,3),np.uint8),dict(stream='camera',frame='new-region-frame',age=5)
        self.scope=PickupScope(self.layout,'/runtime',lambda:copy.deepcopy(self.state),capture=capture)
        self.vision=dict(model_version='a'*64,configuration_sha256='b'*64,source_stream_id='camera',frame_id='inference-frame',image_size=[60,40],
            instances=[dict(instance_id=4,label='plastic',official_label='Straw',mask_rle=runs([15,15,20,25])),
                dict(instance_id=8,label='plastic',official_label='Clear bottle',mask_rle=runs([32,12,38,25])),
                dict(instance_id=9,label='metal',official_label='Drink can',mask_rle=runs([42,15,50,25]))],
            display_grasp_points=[dict(instance_id=4,pixel=[17,20]),dict(instance_id=8,pixel=[35,20]),dict(instance_id=9,pixel=[45,20])])

    def annotate(self):
        with patch('console_pickup_scope.projected_layout',return_value=self.layout):return self.scope.annotate(self.vision)

    def test_only_source_requires_dimensions_but_all_raw_classes_retained(self):
        before=copy.deepcopy(self.vision);value=self.annotate()
        self.assertEqual([i['station_grasp_candidate'] for i in value['instances']],[True,False,False])
        self.assertEqual([i['station_zone'] for i in value['instances']],['source','outside','bin_metal'])
        self.assertEqual([i['official_label'] for i in value['instances']],['Straw','Clear bottle','Drink can'])
        self.assertEqual(value['display_grasp_points'],[self.vision['display_grasp_points'][0]])
        self.assertEqual(value['frame_id'],'inference-frame');self.assertEqual(value['pickup_scope']['region_frame_id'],'new-region-frame')
        self.assertTrue(value['pickup_scope']['advisory_only']);self.assertFalse(value['pickup_scope']['control_authorized'])
        self.assertEqual(self.vision,before)

    def test_camera_model_or_native_identity_change_fails_closed_without_losing_detections(self):
        for key,value in (('source_stream_id','other'),('model_version','wrong'),('image_size',[40,40])):
            old=self.vision[key];self.vision[key]=value;self.scope.cached=None
            result=self.annotate();self.assertEqual(result['pickup_scope']['status'],'unavailable')
            self.assertEqual(len(result['instances']),3);self.assertTrue(all(i['station_grasp_candidate'] is False for i in result['instances']))
            self.vision[key]=old

    def test_new_camera_stream_does_not_use_cached_region(self):
        self.annotate();self.vision['source_stream_id']='new-camera'
        self.assertEqual(self.annotate()['pickup_scope']['status'],'unavailable');self.assertEqual(len(self.capture_calls),2)

    def test_scope_unavailable_preserves_recognition_and_hides_pick_points(self):
        self.scope.capture=lambda:(_ for _ in ()).throw(OSError('camera unavailable'))
        result=self.annotate();self.assertEqual(result['display_grasp_points'],[])
        self.assertEqual(len(result['instances']),3);self.assertEqual(result['pickup_scope']['status'],'unavailable')

if __name__=='__main__':unittest.main()
