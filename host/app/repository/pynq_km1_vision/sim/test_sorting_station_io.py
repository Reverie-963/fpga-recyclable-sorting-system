"""New RGB interface identity, exact pixels, baseline and persistence gates."""
import copy
import base64
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import urllib.error
from unittest.mock import patch
from urllib.parse import urlsplit, parse_qs

import cv2
import numpy as np

from sim.test_sorting_known import config
from src.sorting_demo import DemoClock, DemoSource, DemoMotion
from src.sorting_io import digest, Statistics
from src.sorting_known import KnownController
from src.sorting_loop import Journal
from src.sorting_station_io import StationCamera, StationStatistics, station_policy
from src.station_vision import CPUVision, StationVision
from src.station_vision_server import FrameStore


def station_config():
    c = config()
    rs = dict(source=c['source_roi'], static_a=c['layout']['static_rois'][0],
              static_b=c['layout']['static_rois'][1])
    rs.update({p['bin_id']:p['bin_roi'] for p in c['profiles'].values()})
    c['statistics'] = dict(backend='pl_board_local', url='http://127.0.0.1:8780',
        assets={'station_vision.bit':'c'*64,'station_vision.hwh':'d'*64}, regions=copy.deepcopy(rs),
        image_size=c['image_size'], threshold=45, window_frames=3,max_gap_s=.6)
    c['camera_url'] = 'http://127.0.0.1:8780/frame.jpg'
    return c


class StationAdapterTests(unittest.TestCase):
    def setUp(self):
        self.c, self.clock = station_config(), DemoClock()
        self.image = DemoSource(self.c,self.clock).reference.copy()
        class SyntheticPL(CPUVision):
            backend='pl_board_local'
        backend = SyntheticPL();backend.assets=self.c['statistics']['assets']
        cfg=self.c['statistics']
        self.engine=StationVision(backend,cfg['regions'],cfg['image_size'],45,.6,3)
        self.store=FrameStore(self.engine,clock=self.clock.monotonic)
        self.store.publish(self.image,self.clock.monotonic())
        self.calls=[]
        self.runtime=patch('src.sorting_station_io.time',SimpleNamespace(
            monotonic=self.clock.monotonic,sleep=self.clock.sleep))
        self.runtime.start();self.addCleanup(self.runtime.stop)
        self.request_patch=patch.object(StationStatistics,'request',self.request)
        self.request_patch.start();self.addCleanup(self.request_patch.stop)
        self.stats=StationStatistics(self.c)
        self.camera=StationCamera(self.c,None,self.stats)
        self.camera.opener=SimpleNamespace(open=self.snapshot)
        self.stats.opener=self.camera.opener
        self.tamper=False

    def request(self,route,body=None):
        self.calls.append(route)
        if route=='/baseline':return self.store.bind(body)
        if route=='/vision':
            self.clock.sleep(.2)
            self.store.publish(self.image,self.clock.monotonic()-.01)
            return self.store.vision()
        return self.store.health()

    def snapshot(self,url,timeout):
        query=parse_qs(urlsplit(url).query)
        if 'frame_id' in query:
            index=int(query['frame_id'][0])
        else:
            self.clock.sleep(.2)
            self.store.publish(self.image,self.clock.monotonic()-.01)
            if query.get('window_complete')==['1']:
                try:
                    index=self.store.latest(complete_window=True,min_frame_id=int(query['min_frame_id'][0]))['id']
                except ValueError:
                    raise urllib.error.HTTPError(url,503,'not_ready',{},None)
            else:index=self.store.sequence
        frame=self.store.latest(index)
        ok,png=cv2.imencode('.jpg' if urlsplit(url).path=='/frame.jpg' else '.png',frame['image'])
        reply=io.BytesIO(png.tobytes())
        reply.headers={'X-Stream-Id':self.store.stream_id,'X-Frame-Id':str(index),
            'X-Frame-Age-Ms':str((self.clock.monotonic()-frame['captured'])*1000),
            'X-Decoded-Bgr-Sha256':('e'*64 if self.tamper else digest(frame['image'].tobytes())),
            'X-Station-Metadata':base64.b64encode(json.dumps(self.store.snapshot_metadata(frame)).encode()).decode()}
        return reply

    def test_cpu_geometry_is_explicit_and_no_old_run_requests(self):
        mask=np.zeros(self.image.shape[:2],bool);mask[3:8,4:10]=True
        self.assertEqual(self.stats.geometry(mask)['m00'],30)
        self.assertEqual(self.stats.visual(mask,~mask)['inter'],0)
        self.assertEqual(self.calls,['/health'])
        self.assertEqual(self.stats.provenance()['geometry_backend'],'pc_numpy')
        self.assertFalse(self.stats.provenance()['legacy_dual_core_used'])

    def test_opt_in_atomic_baseline_retains_photometry_after_cache_eviction(self):
        row=self.camera.raw(1);self.c['statistics']['baseline_snapshot_enabled']=True
        posts=[]
        def combined(request,timeout):
            self.assertTrue(request.full_url.endswith('/baseline-snapshot.png'))
            posts.append(request.full_url)
            payload,headers=self.store.bind_snapshot(json.loads(request.data))
            # A slow client can receive an intact bound frame after camera
            # publication evicts it from the normal four-frame selection cache.
            for _ in range(5):self.store.publish(self.image,self.clock.monotonic())
            reply=io.BytesIO(payload);reply.headers=headers;return reply
        self.stats.opener=SimpleNamespace(open=combined)
        bound=self.stats.bind(row)
        self.assertEqual(len(posts),1);self.assertNotIn('/baseline',self.calls)
        self.assertEqual(self.stats.baseline_row['observation']['source_frame_id'],str(bound['baseline_frame_id']))
        self.assertIn('photometry',self.stats.baseline_row['observation'])

    def test_unknown_atomic_post_response_never_retries_or_falls_back(self):
        row=self.camera.raw(1);self.c['statistics']['baseline_snapshot_enabled']=True;posts=[]
        def unknown(request,timeout):
            posts.append(request.full_url);raise urllib.error.URLError('unknown POST result')
        self.stats.opener=SimpleNamespace(open=unknown)
        with self.assertRaises(urllib.error.URLError):self.stats.bind(row)
        self.assertEqual(len(posts),1);self.assertNotIn('/baseline',self.calls)

    def test_unsupported_service_uses_old_route_before_any_atomic_post(self):
        row=self.camera.raw(1);self.c['statistics']['baseline_snapshot_enabled']=True
        original=self.stats.request
        def older(route,body=None):
            reply=original(route,body)
            if route=='/health':reply.pop('capabilities',None)
            return reply
        self.stats.request=older
        self.stats.bind(row)
        self.assertIn('/baseline',self.calls)

    def test_exact_lossless_frame_is_bound_to_pl_evidence(self):
        baseline=self.camera.raw(2)
        self.stats.bind(baseline)
        self.image[4:9,5:12]=255
        row=self.camera.raw(2)
        self.assertIsNone(row['jpeg_sha256'])
        self.assertEqual(row['source_encoding'],'png')
        evidence=self.stats.evidence(row)
        self.assertEqual(row['bgr_sha256'],evidence['frame_bgr_sha256'])
        self.assertEqual(int(row['observation']['source_frame_id']),evidence['frame_id'])

    def test_changed_snapshot_hash_stops_without_cpu_fallback(self):
        self.tamper=True
        with self.assertRaisesRegex(ValueError,'snapshot_pixels_changed'):self.camera.raw(2)

    def test_marker_jpeg_does_not_claim_exact_pl_pixels(self):
        camera=StationCamera(self.c,None,self.stats,lossless=False)
        camera.opener=self.camera.opener
        row=camera.raw(2)
        self.assertEqual(row['source_encoding'],'jpeg')
        self.assertIsNone(row['station_vision'])
        self.assertTrue(row['observation']['marker_guard_only'])

    def test_third_post_retreat_image_has_complete_new_pl_window(self):
        self.stats.bind(self.camera.raw(2))
        self.camera.after(self.clock.monotonic()*1000)
        rows=[self.camera.raw(2) for _ in range(3)]
        last=rows[-1]['station_vision']
        self.assertTrue(last['window_complete'])
        self.assertGreaterEqual(last['frame_id']-2,int(rows[0]['observation']['source_frame_id']))

    def test_assets_and_configuration_mismatch_are_rejected(self):
        for key,value in [('assets',{}),('config_sha256','x'),('faulted',True)]:
            data=self.store.health();data[key]=value
            with self.assertRaises(ValueError):self.stats.checked(data)

    def test_external_baseline_replacement_is_rejected(self):
        baseline=self.camera.raw(2);self.stats.bind(baseline)
        self.camera.raw(2)
        self.store.bind(dict(stream_id=self.store.stream_id,config_sha256=self.stats.signature,
                             frame_id=self.store.sequence))
        with self.assertRaisesRegex(ValueError,'baseline_replaced'):self.camera.raw(2)

    def test_new_stream_and_wrong_roi_are_rejected(self):
        self.store.stream_id='changed'
        with self.assertRaisesRegex(ValueError,'stream_changed'):self.camera.raw(2)
        c=copy.deepcopy(self.c);c['statistics']['regions']['source'][0]+=1
        with self.assertRaisesRegex(ValueError,'roi_mismatch'):station_policy(c)

    def changed_rows(self,wrong_bin=False):
        self.stats.bind(self.camera.raw(2))
        names=['source','bin_paper']+(['bin_metal'] if wrong_bin else [])
        for name in names:
            x,y,x1,y1=self.c['statistics']['regions'][name]
            self.image[y:y1,x:x1]=255
        self.c['min_bin_gain_px']=5
        return [self.camera.raw(2) for _ in range(3)]

    def test_new_pl_gate_requires_persistent_source_and_correct_bin_changes(self):
        rows=self.changed_rows()
        result=self.stats.verify_cycle(rows,'bin_paper',self.camera,lambda i,r:None)
        self.assertEqual(result['status'],'board_rgb_evidence_pass')
        self.assertFalse(result['semantic_verified'])

    def test_wrong_region_change_stops_even_when_target_changed(self):
        rows=self.changed_rows(wrong_bin=True)
        with self.assertRaisesRegex(ValueError,'unexpected_region_change:bin_metal'):
            self.stats.verify_cycle(rows,'bin_paper',self.camera,lambda i,r:None)

    def test_matching_frame_number_with_wrong_pixels_is_rejected(self):
        row=self.changed_rows()[0]
        row['station_vision']['frame_bgr_sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'cycle_evidence_changed'):self.stats.evidence(row)

    def test_impossible_region_counts_are_rejected(self):
        row=self.changed_rows()[0]
        row['station_vision']['regions']['source']['persistent_px']=10000000
        with self.assertRaisesRegex(ValueError,'statistics_invalid'):self.stats.evidence(row)

    def test_incomplete_post_retreat_window_waits_for_new_window(self):
        self.changed_rows()
        rows=[self.camera.raw(2) for _ in range(3)]
        saved=[]
        result=self.stats.verify_cycle(rows,'bin_paper',self.camera,lambda i,r:saved.append(r))
        self.assertEqual(result['status'],'board_rgb_evidence_pass')
        self.assertTrue(any(e['window_complete'] for e in result['frames']))


class StationControllerHookTests(unittest.TestCase):
    def cycle(self, label, bind_failure=False, gate_failure=False):
        c=station_config();clock=DemoClock();source=DemoSource(c,clock)
        source.items=[i for i in source.items if i['label']==label]
        motion=DemoMotion(source,clock)
        class SyntheticGate(Statistics):
            backend='pl_board_local'
            def bind(self,row):
                if bind_failure:raise ValueError('synthetic_baseline_failure')
                self.baseline_row=copy.deepcopy(row)
                return dict(simulated=True)
            def verify_cycle(self,*args):
                if gate_failure:raise ValueError('synthetic_PL_gate_failure')
                return dict(status='synthetic_board_gate_pass')
        stats=SyntheticGate()
        with tempfile.TemporaryDirectory() as folder:
            journal=Journal(Path(folder))
            controller=KnownController(c,source,stats,motion,journal,source.reference,
                source.reference,folder,mode='simulation',motion_profile='compact',
                clock=clock.monotonic,sleep=clock.sleep)
            with journal:
                report=controller.cycle();pending=journal.pending()
        return report,motion,pending

    def test_all_three_cycles_include_new_board_gate(self):
        for label in ('paper','metal','plastic'):
            with self.subTest(label=label):
                report,motion,pending=self.cycle(label)
                self.assertEqual(report['status'],'simulated_visual_pass',report)
                self.assertEqual(report['verification']['board_rgb']['status'],'synthetic_board_gate_pass')
                self.assertEqual(report['verification']['semantic_mask_backend'],'pc_numpy')
                self.assertIsNone(pending)

    def test_baseline_failure_prevents_usb_open_and_any_motion(self):
        report,motion,pending=self.cycle('metal',bind_failure=True)
        self.assertEqual(report['status'],'rejected',report)
        self.assertEqual(motion.sent,[])
        self.assertFalse(motion.opened)
        self.assertIsNone(pending)

    def test_new_board_gate_failure_latches_even_after_correct_semantic_placement(self):
        report,motion,pending=self.cycle('plastic',gate_failure=True)
        self.assertEqual(report['status'],'needs_review',report)
        self.assertEqual(motion.sent[-1],'return')
        self.assertIsNotNone(pending)


if __name__=='__main__':unittest.main()
