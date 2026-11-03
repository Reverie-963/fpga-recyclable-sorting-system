import copy
import hashlib
import unittest
from unittest.mock import patch

import numpy as np

from src.prompted_recyclables import read_policy, settings_for
from src.board_prompted_runtime import PromptedRelayDetector, BoardStationStatistics,BoardPromptedRuntime


class PromptedTransportTests(unittest.TestCase):
    def test_fixed_model_name_and_station_filter_are_pinned(self):
        fixed='yoloe11m-public26-best.pt'
        health=copy.deepcopy(self.health)
        health['model_hashes']={fixed:self.policy['model']['sha256']}
        health['station_filter_config_sha256']='f'*64
        result=dict(self.observation,station_filter=dict(config_sha256='f'*64,raw_instances_preserved=True))
        def request(route,payload=None):return copy.deepcopy(health if route=='health' else result)
        with patch('src.board_prompted_runtime.RelayTransport') as cls:
            cls.return_value.request.side_effect=request
            detector=PromptedRelayDetector('http://localhost:1234','t'*40,2,health,model_name=fixed)
        detector.predict(self.image)
        result['station_filter']['config_sha256']='bad'
        with self.assertRaisesRegex(ValueError,'board_station_filter_result_changed'):detector.predict(self.image)
        result['station_filter']['config_sha256']='f'*64
        result['model_sha256']='bad'
        with self.assertRaisesRegex(ValueError,'prompted_board_result_identity_changed'):detector.predict(self.image)

    def setUp(self):
        self.policy = read_policy()
        self.settings = settings_for(self.policy,{})
        self.health = dict(ready=True,settings=self.settings,generation=3,
            model_hashes={'yoloe-11m-seg.pt':self.policy['model']['sha256']},execution=dict(device='cuda:0'))
        self.image = np.arange(60,dtype=np.uint8).reshape(4,5,3)
        self.observation = dict(settings_sha256=self.settings['settings_sha256'],
            policy_sha256=self.settings['policy_sha256'],generation=3,
            model_sha256=self.policy['model']['sha256'],image_size=[5,4],
            source_bgr_sha256=hashlib.sha256(self.image.tobytes()).hexdigest(),instances=[])

    def detector(self, result=None):
        calls = []
        def request(route,payload=None):
            calls.append(route)
            return copy.deepcopy(self.health if route=='health' else result or self.observation)
        with patch('src.board_prompted_runtime.RelayTransport') as cls:
            cls.return_value.request.side_effect=request
            detector=PromptedRelayDetector('http://localhost:1234','t'*40,2,copy.deepcopy(self.health))
        return detector,calls

    def test_current_24_class_settings_preserved(self):
        detector,calls=self.detector()
        result=detector.predict(self.image)
        self.assertEqual(detector.expected['settings'],self.settings)
        self.assertEqual(len(self.settings['classes']),24)
        self.assertEqual(result['source_bgr_sha256'],self.observation['source_bgr_sha256'])
        self.assertEqual(calls,['health','detect'])

    def test_settings_generation_model_or_pixels_change_rejected_without_retry(self):
        for key,value in [('settings_sha256','bad'),('policy_sha256','bad'),('generation',4),
                          ('model_sha256','bad'),('image_size',[1,1]),('source_bgr_sha256','bad')]:
            with self.subTest(key=key):
                detector,calls=self.detector(dict(self.observation,**{key:value}))
                with self.assertRaises(ValueError):detector.predict(self.image)
                self.assertEqual(calls,['health','detect'])

    def test_changed_prompt_settings_after_window_are_rejected(self):
        detector,_=self.detector()
        changed=copy.deepcopy(self.health);changed['settings']['heights_mm']['paper']=120
        with self.assertRaises(ValueError):detector.validate_health(changed)

    def test_owner_check_interrupts_before_inference(self):
        detector,calls=self.detector()
        detector.check=lambda:(_ for _ in ()).throw(ValueError('stopped'))
        with self.assertRaisesRegex(ValueError,'stopped'):detector.predict(self.image)
        self.assertEqual(calls,['health'])

    def test_pc_marker_detections_must_match_same_pixels_and_camera(self):
        marker=dict(location='pc_vision',camera_sha256='c'*64,
                    source_bgr_sha256=self.observation['source_bgr_sha256'],detections=[])
        detector,_=self.detector(dict(self.observation,fixed_marker_observation=marker))
        detector.camera_sha256='c'*64
        detector.predict(self.image)
        for key,value in [('location','unverified'),('camera_sha256','bad'),('source_bgr_sha256','bad')]:
            detector,_=self.detector(dict(self.observation,fixed_marker_observation=dict(marker,**{key:value})))
            detector.camera_sha256='c'*64
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'board_fixed_marker_image_binding_changed'):
                detector.predict(self.image)

    def test_readonly_statistics_cannot_write_baseline_or_regions(self):
        instance=object.__new__(BoardStationStatistics);instance.token='unused'
        with patch('src.board_prompted_runtime.station_get') as get:
            for route in ('/baseline','/regions'):
                with self.assertRaises(ValueError):instance.request(route,dict(frame_id=1))
            get.assert_not_called()

    def test_current_motion_timing_snapshot_is_explicit_and_bounded(self):
        from pathlib import Path
        policy=Path(__file__).parents[1]/'config/yoloe_taco_recyclables.json'
        timing=dict(max_age_ms=6000,frame_timeout_s=6,source_sha256='a'*64,source_module='sorting_prompted_motion.py')
        runtime=BoardPromptedRuntime('unused-binding',policy,'http://localhost:1234','t'*40,'unused','unused-artifacts',timing)
        self.assertEqual(runtime.timing,timing)
        for key,value in [('max_age_ms',6001),('frame_timeout_s',7),('source_module','untrusted.py')]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                BoardPromptedRuntime('unused',policy,'http://localhost:1234','t'*40,'unused','unused',dict(timing,**{key:value}))

    def test_stale_motion_options_rejected_before_loading_station_or_capturing(self):
        from pathlib import Path
        policy=Path(__file__).parents[1]/'config/yoloe_taco_recyclables.json'
        runtime=BoardPromptedRuntime('unused',policy,'http://localhost:1234','t'*40,'unused','unused')
        expected=dict(self.health,motion_options=dict(runtime.motion_options,stable_frames=3))
        with patch('src.board_prompted_runtime.load_unified') as load:
            with self.assertRaisesRegex(ValueError,'motion_options_snapshot_changed'):runtime.preview(expected)
            load.assert_not_called()


if __name__ == '__main__':unittest.main()
