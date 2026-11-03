import base64
import io
import json
import os
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

import cv2
import numpy as np

from src.sorting_demo import DemoClock, demo_config
from src.sorting_io import (CameraYoloe, HTTPYoloe, Motion, PLStatistics, Statistics, digest, image_stats,
                            pack_mask, unpack_mask)
from src.sorting_statistics_server import make_server
from src.static_mask_contract import geometry, visual


class StatisticsTests(unittest.TestCase):
    def test_cpu_matches_existing_integer_contract(self):
        rng = np.random.default_rng(19)
        stats = Statistics()
        for w, h in [(1,1),(9,3),(640,480)]:
            a, b = (rng.random((h,w)) > .6 for _ in range(2))
            result = stats.geometry(a)
            self.assertEqual(result, geometry(a.astype(int).ravel().tolist(), [w,h], result['job_id']))
            result = stats.visual(a,b)
            self.assertEqual(result, visual(a.astype(int).ravel().tolist(), b.astype(int).ravel().tolist(),
                                            [w,h], result['job_id']))

    def test_packed_mask_round_trip_including_non_byte_width(self):
        for shape in [(1,1),(3,9),(480,640)]:
            a = np.random.default_rng(8).random(shape) > .4
            np.testing.assert_array_equal(unpack_mask(pack_mask(a), [shape[1], shape[0]]), a)

    def test_invalid_packed_size_or_padding_rejected(self):
        with self.assertRaisesRegex(ValueError, 'invalid_packed_mask_length'):
            unpack_mask('', [8,1])
        with self.assertRaisesRegex(ValueError, 'nonzero_mask_padding'):
            unpack_mask(base64.b64encode(b'\xff').decode(), [1,1])
        with self.assertRaisesRegex(ValueError, 'invalid_mask_size'):
            unpack_mask('', [641,1])


class FakeBackend:
    assets = {'vision.bit':'a'*64, 'vision.hwh':'b'*64}
    def __init__(self):
        self.fail = self.corrupt = False
        self.calls = 0

    def run(self, role, masks, job):
        self.calls += 1
        if self.fail:
            raise TimeoutError('synthetic DMA timeout')
        a = masks[0]
        if role == 'geometry':
            result, output = image_stats(a, job), a
        else:
            stats = Statistics()
            stats.job = job-1
            result, output = stats.visual(*masks), masks[0]^masks[1]
        if self.corrupt:
            result['job_id'] += 1
        return result, output.astype(np.uint32).ravel()


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend()
        self.token = 't'*40
        self.server = make_server('127.0.0.1', 0, self.token, self.backend)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)
        self.cfg = dict(url='http://127.0.0.1:'+str(self.server.server_port),
                        token_env='TEST_SORTING_TOKEN', assets=self.backend.assets, timeout_s=2)
        self.env = patch.dict(os.environ, {'TEST_SORTING_TOKEN':self.token})
        self.env.start()
        self.addCleanup(self.env.stop)

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_client_and_dual_core_contract_over_http(self):
        client = PLStatistics(self.cfg)
        a = np.zeros((17,23), bool)
        a[3:10, 4:15] = True
        b = np.roll(a, 2, axis=0)
        self.assertEqual(client.geometry(a)['m00'], 77)
        self.assertEqual(client.visual(a,b)['erased'], 22)
        self.assertEqual(self.backend.calls, 2)
        self.assertEqual(client.provenance()['assets'], self.backend.assets)

    def test_wrong_token_rejected_without_driver_call(self):
        with patch.dict(os.environ, {'TEST_SORTING_TOKEN':'wrong'*8}):
            with self.assertRaises(urllib.error.HTTPError) as result:
                PLStatistics(self.cfg)
        self.assertEqual(result.exception.code, 403)
        self.assertEqual(self.backend.calls, 0)

    def test_asset_pair_mismatch_rejected(self):
        self.cfg['assets'] = dict(self.cfg['assets'], **{'vision.hwh':'c'*64})
        with self.assertRaisesRegex(ValueError, 'pl_assets_or_contract_mismatch'):
            PLStatistics(self.cfg)

    def test_dma_timeout_latches_service_and_does_not_retry(self):
        client = PLStatistics(self.cfg)
        self.backend.fail = True
        with self.assertRaises(urllib.error.HTTPError):
            client.geometry(np.zeros((10,10), bool))
        self.assertEqual(self.backend.calls, 1)
        self.backend.fail = False
        with self.assertRaises(urllib.error.HTTPError):
            client.geometry(np.zeros((10,10), bool))
        self.assertEqual(self.backend.calls, 1)

    def test_register_job_mismatch_rejected(self):
        client = PLStatistics(self.cfg)
        self.backend.corrupt = True
        with self.assertRaises(urllib.error.HTTPError):
            client.geometry(np.ones((10,10), bool))

    def test_oversize_and_bad_mask_do_not_reach_driver(self):
        client = PLStatistics(self.cfg)
        for body in [dict(role='geometry', image_size=[641,1], masks=['AA==']),
                     dict(role='geometry', image_size=[10,10], masks=['AA=='])]:
            body.update(session_id=client.session, job_id=1, request_id='x'*32)
            with self.assertRaises(urllib.error.HTTPError):
                client.request('run', body)
        self.assertEqual(self.backend.calls, 0)

    def test_stale_server_session_rejected(self):
        client = PLStatistics(self.cfg)
        client.session = 'old'
        with self.assertRaises(urllib.error.HTTPError):
            client.geometry(np.ones((10,10), bool))
        self.assertEqual(self.backend.calls, 0)


class CameraTests(unittest.TestCase):
    def setUp(self):
        self.c = demo_config()
        self.c['camera_url'] = 'http://127.0.0.1:8780/frame.jpg'
        self.clock = DemoClock()
        self.clock_patch = patch('src.sorting_io.time', SimpleNamespace(
            monotonic=self.clock.monotonic, sleep=self.clock.sleep))
        self.clock_patch.start()
        self.addCleanup(self.clock_patch.stop)
        detector = SimpleNamespace(predict=lambda image,prompt_mode: dict(instances=[]))
        self.source = CameraYoloe(self.c, detector)
        self.image = np.full((120,160,3), 100, np.uint8)
        ok, encoded = cv2.imencode('.jpg', self.image)
        self.raw = encoded.tobytes()
        self.source_id = '1'
        self.stream = 'one'
        self.age = '0'
        self.source.opener = SimpleNamespace(open=self.reply)

    def reply(self, *args, **kwargs):
        self.clock.sleep(.2)
        response = io.BytesIO(self.raw)
        response.headers = {'X-Frame-Id':self.source_id, 'X-Stream-Id':self.stream, 'X-Frame-Age-Ms':self.age}
        return response

    def test_exact_camera_bytes_bound_to_inference(self):
        row = self.source.next(2)
        self.assertEqual(row['jpeg_sha256'], digest(self.raw))
        self.assertEqual(row['observation']['image_sha256'], digest(self.raw))
        self.assertEqual(row['bgr_sha256'], digest(row['image'].tobytes()))

    def test_model_input_hash_is_preserved_separately_from_camera_encoding(self):
        self.source.detector.predict = lambda image, prompt_mode: dict(instances=[], image_sha256='png-input-hash')
        row = self.source.next(2)
        self.assertEqual(row['observation']['model_input_image_sha256'], 'png-input-hash')
        self.assertEqual(row['observation']['image_sha256'], digest(self.raw))

    def test_raw_motion_frame_never_calls_model(self):
        self.source.detector = None
        row = self.source.raw(2)
        self.assertEqual(row['observation']['source_stream_id'], self.stream)
        self.assertEqual(row['bgr_sha256'], digest(row['image'].tobytes()))
        self.assertNotIn('instances', row['observation'])

    def test_repeated_upstream_id_times_out_without_rejuvenating(self):
        first = self.source.next(2)
        with self.assertRaises(TimeoutError):
            self.source.next(1)
        self.assertEqual(self.source.sequence, 1)

    def test_stream_restart_and_backward_ids_stop(self):
        self.source_id = '9'
        self.source.next(2)
        self.source_id = '8'
        with self.assertRaisesRegex(ValueError, 'upstream_frame_reordered'):
            self.source.next(2)
        self.source_id, self.stream = '10', 'restart'
        with self.assertRaisesRegex(ValueError, 'camera_stream_restarted'):
            self.source.next(2)

    def test_upstream_age_applies_to_capture_estimate(self):
        self.age = '1500'
        row = self.source.next(2)
        self.assertAlmostEqual(self.clock.monotonic()*1000-row['observation']['capture_started_ms'],1500)

    def test_pre_retreat_cache_is_discarded(self):
        self.source.after(self.clock.monotonic()*1000+5000)
        with self.assertRaises(TimeoutError):
            self.source.next(1)

    def test_unbounded_or_remote_camera_source_rejected(self):
        self.c['camera_url'] = 'http://example.com/frame.jpg'
        with self.assertRaisesRegex(ValueError, 'use_local_camera_bridge'):
            CameraYoloe(self.c, None)


class MotionWaitTests(unittest.TestCase):
    def setUp(self):
        self.clock = DemoClock()
        self.runtime = patch('src.sorting_io.time', SimpleNamespace(
            monotonic=self.clock.monotonic, sleep=self.clock.sleep))
        self.runtime.start()
        self.addCleanup(self.runtime.stop)
        self.motion = Motion({}, 'SYNTHETIC_STOP_FILE_NOT_PRESENT')
        self.payload = b'{#005P1000T1500!}'
        self.replies = [self.payload]
        def read(size):
            self.clock.sleep(.05)
            return self.replies.pop(0) if self.replies else b''
        self.motion.connection = SimpleNamespace(port=SimpleNamespace(read=read))
        self.motion.echo = self.payload

    def test_complete_echo_still_waits_entire_move_duration(self):
        start = self.clock.monotonic()
        self.motion.wait(1.5)
        self.assertGreaterEqual(self.clock.monotonic()-start, 1.5)
        self.assertTrue(self.motion.last_receipt['echo_matched'])
        self.assertTrue(self.motion.last_receipt['completed_movement_wait'])
        self.assertFalse(self.motion.last_receipt['servo_ack_claimed'])

    def test_partial_echo_is_unknown_and_never_retried(self):
        self.replies = [self.payload[:5]]
        with self.assertRaisesRegex(ValueError, 'missing_serial_echo_execution_unknown'):
            self.motion.wait(.2)
        self.assertFalse(self.motion.last_receipt['echo_matched'])

    def test_echo_may_arrive_in_chunks(self):
        self.replies = [self.payload[:8], self.payload[8:]]
        self.motion.wait(.3)
        self.assertIsNone(self.motion.echo)

    def test_operator_stop_interrupts_wait(self):
        def stop():
            if self.clock.monotonic() > 10.1:
                raise ValueError('operator_stop_requested')
        self.motion.check_stop = stop
        with self.assertRaisesRegex(ValueError, 'operator_stop_requested'):
            self.motion.wait(5)
        self.assertLess(self.clock.monotonic(), 11)


class SharedYoloeTests(unittest.TestCase):
    def setUp(self):
        self.adapter = object.__new__(HTTPYoloe)
        self.adapter.c = dict(model_version='model',runtime_id='runtime')
        self.image = np.random.default_rng(3).integers(0,256,(60,80,3),dtype=np.uint8)
        self.change = {}
        def request(route, data):
            self.assertEqual(route, 'detect')
            raw = base64.b64decode(data['image_base64'])
            np.testing.assert_array_equal(cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR),self.image)
            return dict(image_sha256=digest(raw),model_version='model',runtime_id='runtime',
                        prompt_mode='visual',image_size=[80,60],instances=[],**self.change)
        self.adapter.request = request

    def test_lossless_pixels_sent_to_shared_gpu_model(self):
        self.assertEqual(self.adapter.predict(self.image,'visual')['instances'],[])

    def test_different_input_reply_cannot_drive_control(self):
        original=self.adapter.request
        def request(*args):
            row=original(*args);row['image_sha256']='wrong';return row
        self.adapter.request=request
        with self.assertRaisesRegex(ValueError,'yoloe_input_image_hash_mismatch'):
            self.adapter.predict(self.image,'visual')

    def test_model_runtime_mode_and_size_are_bound(self):
        original=self.adapter.request
        for key,value in [('model_version','other'),('runtime_id','other'),('prompt_mode','text'),('image_size',[640,640])]:
            with self.subTest(key=key):
                def request(*args,key=key,value=value):
                    row=original(*args);row[key]=value;return row
                self.adapter.request=request
                with self.assertRaisesRegex(ValueError,'yoloe_result_identity_mismatch'):
                    self.adapter.predict(self.image,'visual')

    def test_remote_model_is_not_allowed(self):
        c=dict(detector_url='http://example.com',frame_timeout_s=2)
        with self.assertRaisesRegex(ValueError,'use_local_yoloe_service'):
            HTTPYoloe(c,None)

    def test_trained_model_requires_explicit_family_and_supervised_mode(self):
        for family in ('yolo11s', 'yolov8s'):
            with self.subTest(family=family):
                health=dict(model_loaded=True,model_family=family,model_versions={'supervised':'new-model'},
                            runtime_id='new-runtime',execution=dict(device='cuda:0',precision='fp16'))
                with patch.object(HTTPYoloe,'request',return_value=health):
                    with self.assertRaisesRegex(ValueError,'detector_service_is_not_yoloe'):
                        HTTPYoloe(dict(detector_url='http://127.0.0.1:8778'),None)
                    c=dict(detector_url='http://127.0.0.1:8778',model_family=family,prompt_mode='supervised',
                           device='cuda:0',half=True)
                    detector=HTTPYoloe(c,None)
                    self.assertEqual(detector.execution_id('supervised'),'new-runtime')
                    with self.assertRaisesRegex(ValueError,'trained_model_has_no_prompt_modes'):
                        detector.execution_id('visual')
                    c['model_family']='yolov8s' if family=='yolo11s' else 'yolo11s'
                    with self.assertRaisesRegex(ValueError,'deployed_model_family_mismatch'):
                        HTTPYoloe(c,None)


if __name__ == '__main__':
    unittest.main()
