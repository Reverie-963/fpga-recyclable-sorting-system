import base64
import copy
import json
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).parents[1]))
from item_profile_store import ItemProfileStore
from voice_dimensions import Dimensions,features,match,make_server,pickup_candidate,projected_layout,fresh_observation


class DimensionsTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.store=ItemProfileStore(Path(self.temp.name)/'profiles.json')
        self.size=[40,30];mask=np.zeros((30,40),np.uint8);mask[12:18,12:18]=1
        self.runs=[];last=0;count=0
        for v in mask.ravel():
            if v==last:count+=1
            else:self.runs.append(count);last=int(v);count=1
        self.runs.append(count)
        self.rgba=np.zeros((30,40,4),np.uint8);self.rgba[:]=[230,40,70,255]
        self.query=features(self.rgba,self.runs,self.size)
        self.profile=dict(name='Measured straw',pose='flat',material='plastic',height_mm=10,width_mm=10,
            official_label='Plastic straw',reference_model='a'*64,features=self.query)
        self.store.command(dict(action='save',expected_revision=0,profile=self.profile))
        self.layout=dict(image_size=self.size,source_roi=[8,8,30,25],bins=[[0,0,8,8]],static_rois=[[35,25,40,30]],sha256='d'*64)
        self.binding=dict(profile_sha256='c'*64,revision=3,model_sha256='a'*64,configuration_sha256='b'*64)
        self.state=dict(board_root='/runtime',compute_location='pynq_ps',busy=False,blockers=[],execution_model=self.binding,per_instance_widths_supported=True)
        ok,png=cv2.imencode('.png',cv2.cvtColor(self.rgba,cv2.COLOR_RGBA2BGR));self.assertTrue(ok)
        self.vision=dict(model_version='a'*64,configuration_sha256='b'*64,source_stream_id='camera',frame_id='42',image_size=self.size,
            result_age_ms=20,overlay_base64=base64.b64encode(png).decode(),instances=[dict(label='plastic',official_label='Plastic straw',mask_rle=self.runs)])
        self.calls=[]
        def request(path):
            self.calls.append(path);return copy.deepcopy(self.state if path=='/operation' else self.vision)
        self.service=Dimensions(request,self.store,self.layout,'/runtime')
        self.data=dict(request_id='12345678123446789234123456789012',material='AUTO',profile_sha256='c'*64,revision=3)

    def test_exact_ui_features_parity(self):
        source=Path(__file__).parents[2]/'pynq_km1_vision/cloud_console/ui/item_profiles.js'
        script='require(process.argv[1]);const p=globalThis.SortingItemProfiles;let s="";process.stdin.on("data",x=>s+=x);process.stdin.on("end",()=>{const a=JSON.parse(s);console.log(JSON.stringify(p.features(...a)))});'
        result=subprocess.run(['node','-e',script,str(source)],input=json.dumps([self.rgba.ravel().tolist(),self.runs,self.size]),text=True,capture_output=True,check=True)
        expected=json.loads(result.stdout)
        for key in ('rgb','hist','aspect','area'):np.testing.assert_allclose(expected[key],self.query[key],rtol=1e-12,atol=1e-12)

    def test_current_library_dimensions_frozen_no_actions(self):
        answer=self.service.prepare(self.data);item=answer['height_plan']['items'][0]
        self.assertEqual((item['object_height_mm'],item['grasp_width_mm']),(10,10))
        self.assertEqual(item['measurement_profile']['id'],self.store.read()['profiles'][0]['id'])
        self.assertEqual(set(self.calls),{'/operation','/live-result'})
        self.assertFalse(answer['physical_request_sent'])

    def test_fresh_inference_is_bound_to_lossless_pixels_and_camera_metadata(self):
        import hashlib
        bgr=cv2.cvtColor(self.rgba,cv2.COLOR_RGBA2BGR);ok,png=cv2.imencode('.png',bgr);self.assertTrue(ok)
        digest=hashlib.sha256(bgr.tobytes()).hexdigest();headers={'X-Decoded-Bgr-Sha256':digest,'X-Stream-Id':'camera','X-Frame-Id':'77','X-Frame-Age-Ms':'5'}
        class Reply(io.BytesIO):pass
        def open_reply(*args,**kwargs):
            reply=Reply(png.tobytes());reply.headers=headers;return reply
        class Opener:open=staticmethod(open_reply)
        def predict(payload):
            self.assertEqual(base64.b64decode(payload['image_base64']),png.tobytes())
            return dict(source_bgr_sha256=digest)
        with patch('voice_dimensions.urllib.request.build_opener',return_value=Opener()):
            vision,image=fresh_observation(predict)
            self.assertEqual(vision['source_frame_id'],'77');self.assertEqual(vision['source_stream_id'],'camera');np.testing.assert_array_equal(image,bgr)
            with self.assertRaisesRegex(ValueError,'inference_pixels_changed'):fresh_observation(lambda payload:dict(source_bgr_sha256='other'))
            headers['X-Decoded-Bgr-Sha256']='wrong'
            with self.assertRaisesRegex(ValueError,'source_pixels_changed'):fresh_observation(predict)

    def test_missing_ambiguous_manual_only_or_wrong_model_never_falls_back_by_class(self):
        self.assertIsNone(match([dict(self.profile,features=None,reference_model=None)],self.query,'plastic','a'*64))
        self.assertIsNone(match([self.profile,self.profile],self.query,'plastic','a'*64))
        self.assertIsNone(match([self.profile],self.query,'plastic','e'*64))
        self.store.command(dict(action='save',expected_revision=1,profile=self.profile))
        with self.assertRaisesRegex(ValueError,'missing_or_ambiguous'):self.service.prepare(self.data)

    def test_unregistered_external_object_is_not_added_to_voice_height_plan(self):
        outside=np.zeros((30,40),bool);outside[10:16,31:36]=True;encoded=[0];previous=False
        for bit in outside.ravel():
            if bit==previous:encoded[-1]+=1
            else:encoded.append(1);previous=bit
        self.vision['instances'].append(dict(label='metal',official_label='Unregistered can',mask_rle=encoded))
        answer=self.service.prepare(self.data)
        self.assertEqual(len(answer['height_plan']['items']),1)
        self.assertEqual(answer['height_plan']['items'][0]['label'],'plastic')
        self.assertEqual(len(self.vision['instances']),2)

    def test_stale_frame_model_root_busy_stop_and_width_capability_rejected(self):
        for target,key,value in ((self.vision,'result_age_ms',3000),(self.vision,'configuration_sha256','e'*64),
                (self.state,'board_root','/other'),(self.state,'busy',True),(self.state,'blockers',['operator_stop_requested']),
                (self.state,'per_instance_widths_supported',False)):
            original=target[key];target[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.service.prepare(self.data)
            target[key]=original

    def test_no_target_and_known_bin_are_excluded(self):
        with self.assertRaisesRegex(ValueError,'no_registered_source'):self.service.prepare(dict(self.data,material='METAL'))
        mask=np.zeros((30,40),bool);mask[1:7,1:7]=True
        self.assertFalse(pickup_candidate(mask,self.layout))
        mask[:]=False;mask[10:16,31:36]=True
        self.assertFalse(pickup_candidate(mask,self.layout))
        mask[:]=False;mask[26:29,36:39]=True
        self.assertFalse(pickup_candidate(mask,self.layout))

    def test_bin_projection_uses_same_frame_and_existing_bound_review_policy(self):
        sys.path.insert(0,str(Path(__file__).parents[2]/'pynq_km1_vision'))
        from src.station_geometry import fingerprint
        layout=dict(self.layout,camera=dict(camera_id='camera'),station=dict(world_frame_id='world'),review_factor=3,highest_mm=65,
            scene=dict(scene_regions=dict(source=dict(world_xyz_mm=[[0,0,0]]*4),bin_metal=dict(world_xyz_mm=[[1,1,0]]*4))))
        pose=dict(rotation_world_to_camera=np.eye(3).tolist(),translation_world_to_camera_mm=[0,0,500])
        def project(points,value,camera,station):
            self.assertEqual(value['kind'],'station-pose-preview-v1');self.assertEqual(value['camera_sha256'],fingerprint(camera))
            return [[10,10],[26,10],[26,26],[10,26]]
        with patch('src.station_geometry.detect_markers',return_value=['same-frame']) as markers,patch('src.station_geometry.estimate_pose',return_value=pose) as estimate,patch('src.station_geometry.world_to_pixels',side_effect=project):
            projected=projected_layout(layout,self.rgba[:,:,:3],2)
        self.assertEqual(estimate.call_args.args[-2:],(2,3));self.assertEqual(projected['bins'],[[11,11,26,26]])
        self.assertIs(markers.call_args.args[0].base,self.rgba)

    def test_model_and_library_change_during_preparation_rejected(self):
        original=self.service.request
        def changed(path):
            value=original(path)
            if path=='/live-result':self.state['execution_model']=dict(self.binding,revision=4)
            return value
        self.service.request=changed
        with self.assertRaisesRegex(ValueError,'model_changed'):self.service.prepare(self.data)
        self.state['execution_model']=self.binding
        def library_changed(path):
            value=original(path)
            if path=='/live-result':self.store.command(dict(action='save',expected_revision=1,profile=dict(self.profile,name='another'),id=self.store.read()['profiles'][0]['id']))
            return value
        self.service.request=library_changed
        with self.assertRaisesRegex(ValueError,'library_changed'):self.service.prepare(self.data)

    def test_service_authentication_and_no_replay(self):
        server=make_server(0,'test-token',self.service);thread=threading.Thread(target=server.serve_forever);thread.start()
        self.addCleanup(server.server_close);self.addCleanup(thread.join);self.addCleanup(server.shutdown)
        url='http://127.0.0.1:'+str(server.server_port)+'/prepare'
        def post(token):
            request=urllib.request.Request(url,data=json.dumps(self.data).encode(),headers={'Authorization':'Bearer '+token})
            return json.load(urllib.request.urlopen(request,timeout=3))
        with self.assertRaises(urllib.error.HTTPError) as caught:post('wrong')
        self.assertEqual(caught.exception.code,403)
        self.assertIn('height_plan',post('test-token'))
        with self.assertRaises(urllib.error.HTTPError) as caught:post('test-token')
        self.assertEqual(caught.exception.code,409)

    def test_unbound_measurement_round_trip_and_half_reference_rejected(self):
        data=self.store.command(dict(action='save',expected_revision=1,profile=dict(self.profile,reference_model=None,features=None,
            measurement_source='operator measurement, dated record')))
        self.assertIsNone(data['profiles'][-1]['features'])
        with self.assertRaises(ValueError):self.store.command(dict(action='save',expected_revision=2,profile=dict(self.profile,features=None)))


if __name__=='__main__':unittest.main()
