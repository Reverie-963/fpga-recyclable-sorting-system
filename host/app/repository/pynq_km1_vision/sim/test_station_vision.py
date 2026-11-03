import json
import threading
import unittest
import urllib.request
import urllib.error
import numpy as np
import base64
import hashlib
import cv2
from unittest.mock import patch

from src.station_vision import oracle,CPUVision,StationVision,REGION_NAMES
from src.station_vision_server import FrameStore,make_server
from src.station_vision_selftest import startup_selftest


def regions(w=10,h=8):return {name:[0,0,w,h] for name in REGION_NAMES}


class BaselineSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.clock=[2.]
        self.store=FrameStore(StationVision(CPUVision(),regions(),[10,8],45,.6,3),clock=lambda:self.clock[0])
        self.image=np.arange(240,dtype=np.uint8).reshape(8,10,3)
        self.frame=self.store.publish(self.image,1.99)
        self.body=dict(stream_id=self.store.stream_id,config_sha256=self.store.engine.config_sha256,frame_id=1)

    def test_lossless_cache_is_immutable_and_encodes_once_per_frame(self):
        original=cv2.imencode
        with patch('src.station_vision_server.cv2.imencode',wraps=original) as encoded:
            a,ha=self.store.lossless(self.frame);b,hb=self.store.lossless(self.frame)
            self.assertIs(a,b);self.assertEqual(ha,hb);self.assertEqual(encoded.call_count,1)
        original_pixels=self.frame['image'].copy();self.image[:]=0
        np.testing.assert_array_equal(self.frame['image'],original_pixels)
        with self.assertRaises(ValueError):self.frame['image'].setflags(write=True)

    def test_cached_payload_never_refreshes_frame_freshness(self):
        self.store.lossless(self.frame);self.clock[0]+=3
        with self.assertRaisesRegex(ValueError,'fresh_frame_unavailable'):self.store.latest(1)
        with self.assertRaises(ValueError):self.store.bind_snapshot(self.body)
        self.assertIsNone(self.store.engine.baseline_id)

    def test_png_encoding_does_not_hold_the_capture_store_lock(self):
        entered,release,published=threading.Event(),threading.Event(),threading.Event()
        original=cv2.imencode
        def slow(extension,*args,**kwargs):
            if extension=='.png':entered.set();release.wait(2)
            return original(extension,*args,**kwargs)
        with patch('src.station_vision_server.cv2.imencode',side_effect=slow):
            encoder=threading.Thread(target=lambda:self.store.lossless(self.frame),daemon=True)
            encoder.start();self.assertTrue(entered.wait(1))
            def publish():self.store.publish(self.image,2.);published.set()
            publisher=threading.Thread(target=publish,daemon=True);publisher.start()
            try:self.assertTrue(published.wait(1),'PNG encoding blocked camera publication')
            finally:release.set();encoder.join(3);publisher.join(3)

    def test_combined_reply_binds_identical_pixels_and_does_not_authorize_motion(self):
        payload,headers=self.store.bind_snapshot(self.body)
        result=json.loads(base64.b64decode(headers['X-Baseline-Record']))
        decoded=cv2.imdecode(np.frombuffer(payload,np.uint8),cv2.IMREAD_COLOR)
        self.assertEqual(result['baseline_frame_id'],1)
        self.assertEqual(result['baseline_bgr_sha256'],hashlib.sha256(decoded.tobytes()).hexdigest())
        self.assertEqual(headers['X-Decoded-Bgr-Sha256'],result['baseline_bgr_sha256'])
        self.assertFalse(result['automatic_motion_allowed'])

    def test_encoding_time_does_not_relax_original_half_second_gate(self):
        original=cv2.imencode
        def slow(*args,**kwargs):
            result=original(*args,**kwargs);self.clock[0]+=.6;return result
        with patch('src.station_vision_server.cv2.imencode',side_effect=slow):
            with self.assertRaisesRegex(ValueError,'baseline_frame_too_old'):self.store.bind_snapshot(self.body)
        self.assertIsNone(self.store.engine.baseline_id)

    def test_identity_and_encoding_failures_leave_baseline_unbound(self):
        for key,value in (('stream_id','other'),('config_sha256','wrong'),('frame_id',True)):
            with self.assertRaises(ValueError):self.store.bind_snapshot(dict(self.body,**{key:value}))
            self.assertIsNone(self.store.engine.baseline_id)
        with patch('src.station_vision_server.cv2.imencode',return_value=(False,None)):
            with self.assertRaisesRegex(ValueError,'png_encode_failed'):self.store.bind_snapshot(self.body)
        self.assertIsNone(self.store.engine.baseline_id)

    def test_new_http_endpoint_still_requires_auth_and_explicit_frame(self):
        token='x'*32;server=make_server('127.0.0.1',0,token,self.store)
        worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
        try:
            url='http://127.0.0.1:'+str(server.server_port)+'/baseline-snapshot.png'
            def request(auth):
                return urllib.request.urlopen(urllib.request.Request(url,data=json.dumps(self.body).encode(),
                    headers={'Authorization':'Bearer '+token} if auth else {}),timeout=2)
            with self.assertRaises(urllib.error.HTTPError) as error:request(False)
            self.assertEqual(error.exception.code,403);error.exception.close()
            with request(True) as reply:
                self.assertEqual(reply.headers['Content-Type'],'image/png')
                self.assertTrue(reply.read().startswith(b'\x89PNG'))
        finally:server.shutdown();worker.join();server.server_close()


class StationVisionTests(unittest.TestCase):
    def setUp(self):self.base=np.zeros((8,10,3),np.uint8)

    def test_rgb_sum_does_not_overflow_and_boundary_is_strict(self):
        image=self.base.copy();image[2,3]=[100,100,100];image[3,4]=[85,85,85]
        result,history=oracle(self.base,image,regions(),255)
        self.assertEqual(result['source']['changed_px'],1)
        self.assertEqual(result['source']['delta_sum'],555)
        self.assertEqual(result['source']['bbox_inclusive_px'],[3,2,3,2])
        self.assertTrue(history[2,3]);self.assertFalse(history[3,4])

    def test_region_binding_compare_and_swap_invalidates_cached_baseline_evidence(self):
        now=[1.];engine=StationVision(CPUVision(),regions(),[10,8],45)
        store=FrameStore(engine,clock=lambda:now[0]);frame=store.publish(self.base,1.)
        store.bind(dict(stream_id=store.stream_id,config_sha256=engine.config_sha256,frame_id=frame['id']))
        now[0]=1.1;store.publish(self.base,1.1)
        self.assertIsNotNone(store.latest_evidence)
        health=store.health();rs=regions();rs['source']=[1,1,9,7]
        request=dict(stream_id=store.stream_id,expected_config_sha256=health['config_sha256'],
            expected_baseline_frame_id=health['baseline_frame_id'],frame_id=health['frame_id'],
            regions=rs,calibration_sha256='a'*64)
        changed=store.reconfigure(request)
        self.assertTrue(changed['baseline_invalidated']);self.assertIsNone(store.engine.baseline_id)
        self.assertIsNone(store.latest_evidence)
        self.assertTrue(all(row['evidence'] is None for row in store.frames))
        self.assertNotEqual(store.engine.config_sha256,engine.config_sha256)
        with self.assertRaisesRegex(ValueError,'identity_changed'):store.reconfigure(request)
        now[0]=1.2;frame=store.publish(self.base,1.2)
        store.bind(dict(stream_id=store.stream_id,config_sha256=store.engine.config_sha256,frame_id=frame['id']))
        now[0]=1.3;store.publish(self.base,1.3)
        self.assertEqual(store.latest_evidence['regions']['source']['area_px'],48)

    def test_region_binding_rejects_invalid_or_stale_request_without_mutation(self):
        now=[1.];engine=StationVision(CPUVision(),regions(),[10,8],45)
        store=FrameStore(engine,clock=lambda:now[0]);frame=store.publish(self.base,1.)
        request=dict(stream_id=store.stream_id,expected_config_sha256=engine.config_sha256,
            expected_baseline_frame_id=None,frame_id=frame['id'],regions=regions(),calibration_sha256='a'*64)
        for field,value in [('stream_id','wrong'),('expected_config_sha256','b'*64),
                            ('expected_baseline_frame_id',4),('frame_id',None),('calibration_sha256',None)]:
            with self.assertRaises(ValueError):store.reconfigure(dict(request,**{field:value}))
            self.assertIs(store.engine,engine)
        now[0]=1.6
        with self.assertRaisesRegex(ValueError,'too_old'):store.reconfigure(request)
        self.assertIs(store.engine,engine)

    def test_half_open_rois_and_empty_change_bbox(self):
        rs=regions();rs['source']=[1,1,3,3];image=self.base.copy();image[3,3]=255
        result,_=oracle(self.base,image,rs,0)
        self.assertEqual(result['source']['changed_px'],0)
        self.assertEqual(result['source']['area_px'],4)
        self.assertEqual(result['source']['bbox_inclusive_px'],[0]*4)
        self.assertEqual(result['bin_paper']['changed_px'],1)

    def test_reject_invalid_geometry_type_and_threshold(self):
        for threshold in (-1,766,True):
            with self.assertRaises(ValueError):oracle(self.base,self.base,regions(),threshold)
        bad=regions();bad['source']=[0,0,11,8]
        with self.assertRaises(ValueError):oracle(self.base,self.base,bad,0)
        with self.assertRaises(ValueError):oracle(self.base.astype(float),self.base,regions(),0)

    def test_persistence_is_pixel_intersection_not_count_agreement(self):
        image=self.base.copy();image[2,3]=255;first,h=oracle(self.base,image,regions(),0)
        shifted=self.base.copy();shifted[2,4]=255;second,_=oracle(self.base,shifted,regions(),0,h)
        self.assertEqual(first['source']['changed_px'],second['source']['changed_px'])
        self.assertEqual(second['source']['persistent_px'],0)

    def test_three_frame_windows_restart_without_replacing_baseline(self):
        engine=StationVision(CPUVision(),regions(),[10,8],0)
        engine.bind_baseline(self.base,1,1.)
        image=self.base.copy();image[2,3]=255
        for i in range(2,8):
            out=engine.process(self.base if i==2 else image,i,1+(i-1)*.1)
            if i==4:self.assertEqual(out['regions']['source']['persistent_px'],0)
        self.assertEqual(out['regions']['source']['persistent_px'],1)
        self.assertTrue(out['window_complete']);self.assertEqual(out['baseline_frame_id'],1)
        self.assertFalse(out['automatic_motion_allowed']);self.assertFalse(out['physical_success_verified'])

    def test_stale_frames_do_not_advance_history_and_gap_invalidates_baseline(self):
        engine=StationVision(CPUVision(),regions(),[10,8],0)
        engine.bind_baseline(self.base,1,1.)
        with self.assertRaises(ValueError):engine.process(self.base,1,1.1)
        with self.assertRaises(ValueError):engine.process(self.base,2,float('nan'))
        self.assertEqual(engine.samples,0)
        with self.assertRaisesRegex(ValueError,'capture_gap'):engine.process(self.base,2,2.)
        self.assertIsNone(engine.baseline_id)

    def test_frame_bound_baseline_and_raw_snapshot_cache(self):
        engine=StationVision(CPUVision(),regions(),[10,8],0);store=FrameStore(engine,clock=lambda:1.1)
        frame=store.publish(self.base,1.)
        body=dict(stream_id=store.stream_id,config_sha256=engine.config_sha256,frame_id=frame['id'])
        with self.assertRaises(ValueError):store.bind(dict(body,stream_id='old'))
        with self.assertRaises(ValueError):store.bind(dict(body,frame_id=999))
        store.bind(body);store.publish(self.base,.0+1.1)
        self.assertTrue(store.vision()['available']);self.assertEqual(store.latest(1)['id'],1)
        for _ in range(4):store.publish(self.base,1.1)
        with self.assertRaises(ValueError):store.latest(1)

    def test_hardware_fault_never_becomes_a_cpu_fallback(self):
        class Broken(CPUVision):
            backend='pl_board_local'
            def process(self,image):raise TimeoutError('broken')
        engine=StationVision(Broken(),regions(),[10,8],0);store=FrameStore(engine,clock=lambda:1.1)
        store.publish(self.base,1.);store.bind(dict(stream_id=store.stream_id,config_sha256=engine.config_sha256,frame_id=1))
        store.publish(self.base,1.1);out=store.vision()
        self.assertTrue(out['faulted']);self.assertFalse(out['available']);self.assertEqual(out['backend'],'pl_board_local')

    def test_snapshot_metadata_stays_with_selected_cached_pixels(self):
        engine=StationVision(CPUVision(),regions(),[10,8],0)
        store=FrameStore(engine,clock=lambda:1.3)
        store.publish(self.base,1.)
        body=dict(stream_id=store.stream_id,config_sha256=engine.config_sha256,frame_id=1)
        store.bind(body)
        old=store.publish(self.base,1.1)
        changed=self.base.copy();changed[2,3]=255
        store.publish(changed,1.2)
        data=store.snapshot_metadata(old)
        self.assertEqual(data['frame_id'],old['id'])
        self.assertEqual(data['evidence']['frame_id'],old['id'])
        self.assertEqual(data['evidence']['regions']['source']['changed_px'],0)
        self.assertEqual(store.vision()['evidence']['regions']['source']['changed_px'],1)
        store.bind(dict(body,frame_id=3))
        self.assertFalse(store.snapshot_metadata(old)['available'])

    def test_complete_window_selection_respects_minimum_frame(self):
        engine=StationVision(CPUVision(),regions(),[10,8],0)
        store=FrameStore(engine,clock=lambda:1.5)
        store.publish(self.base,1.)
        store.bind(dict(stream_id=store.stream_id,config_sha256=engine.config_sha256,frame_id=1))
        for index in range(2,6):store.publish(self.base,1+index*.05)
        self.assertEqual(store.latest(complete_window=True,min_frame_id=3)['id'],4)
        with self.assertRaises(ValueError):store.latest(complete_window=True,min_frame_id=5)

    def test_http_auth_baseline_binding_and_absent_motion_routes(self):
        engine=StationVision(CPUVision(),regions(),[10,8],0);store=FrameStore(engine,clock=lambda:1.1)
        store.publish(self.base,1.);token='a'*32;server=make_server('127.0.0.1',0,token,store)
        worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));base='http://127.0.0.1:'+str(server.server_port)
        def request(path,body=None,auth=True):return opener.open(urllib.request.Request(base+path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={'Authorization':'Bearer '+token} if auth else {}),timeout=2)
        try:
            with self.assertRaises(urllib.error.HTTPError) as e:request('/health',auth=False)
            self.assertEqual(e.exception.code,403);e.exception.close()
            with request('/health') as reply:self.assertTrue(json.load(reply)['camera_connected'])
            with request('/baseline',dict(stream_id=store.stream_id,config_sha256=engine.config_sha256,frame_id=1)) as reply:
                self.assertFalse(json.load(reply)['automatic_motion_allowed'])
            for path in ('/execute','/serial','/grasp'):
                with self.assertRaises(urllib.error.HTTPError) as e:request(path,{})
                self.assertEqual(e.exception.code,404);e.exception.close()
            with request('/snapshot.png?frame_id=1') as reply:self.assertTrue(reply.read().startswith(b'\x89PNG'))
        finally:server.shutdown();worker.join();server.server_close()


class StartupComparisonTests(unittest.TestCase):
    def test_cpu_fallback_cannot_be_labelled_as_fpga(self):
        config=dict(image_size=[10,8],regions=regions(),threshold=45)
        with self.assertRaises(ValueError):startup_selftest(CPUVision(),config)

    def test_startup_requires_exact_statistics(self):
        class SimulatedBoard(CPUVision):
            backend='pl_board_local';assets={'simulation':True}
        config=dict(image_size=[10,8],regions=regions(),threshold=45)
        result=startup_selftest(SimulatedBoard(),config)
        self.assertEqual(result['samples'],12);self.assertTrue(result['all_equal'])
        self.assertFalse(result['whole_system_speedup_measured']);self.assertFalse(result['physical_acceptance'])
        class Wrong(SimulatedBoard):
            def process(self,current):
                output=super().process(current);output['source']['changed_px']+=1;return output
        with self.assertRaisesRegex(RuntimeError,'integer_mismatch'):startup_selftest(Wrong(),config)


if __name__=='__main__':unittest.main()
