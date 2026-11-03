import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from src.monitor_station_vision import station_vision_status,bind_station_baseline,pc_reference_summary,saved_acceleration_summary
from src.station_vision import StationVision,CPUVision
from src.station_vision_server import FrameStore
from sim.test_station_vision import regions


class MonitorStationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'config.json'
        self.assets={'station_vision.bit':'a'*64,'station_vision.hwh':'b'*64}
        config=dict(synthetic_fixture=False,backend='pl_board_local',regions=regions(),image_size=[10,8],
            threshold=0,max_gap_s=.6,window_frames=3,assets=self.assets)
        self.path.write_text(json.dumps(config))
        backend=CPUVision();backend.backend='pl_board_local';backend.assets=self.assets
        engine=StationVision(backend,regions(),[10,8],0)
        store=FrameStore(engine,clock=lambda:1.1);image=np.zeros((8,10,3),np.uint8)
        store.publish(image,1.);store.bind(dict(stream_id=store.stream_id,config_sha256=engine.config_sha256,frame_id=1))
        store.publish(image,1.1);self.data=store.vision()

    def test_exact_board_config_and_frame_identity_are_required(self):
        with patch('src.monitor_station_vision.board_request',return_value=self.data):
            self.assertTrue(station_vision_status(self.path)['available'])
        for change in (dict(assets={}),dict(backend='cpu_reference'),dict(frame_age_ms=2500),dict(frame_id=99)):
            with self.subTest(change=change),patch('src.monitor_station_vision.board_request',return_value=dict(self.data,**change)):
                self.assertFalse(station_vision_status(self.path)['available'])

    def test_stale_evidence_cannot_be_promoted_by_current_health(self):
        data=copy.deepcopy(self.data);data['evidence']['baseline_frame_id']=99
        with patch('src.monitor_station_vision.board_request',return_value=data):
            self.assertFalse(station_vision_status(self.path)['available'])

    def test_runtime_region_display_recomputes_identity_without_baseline_permission(self):
        data=copy.deepcopy(self.data)
        changed=copy.deepcopy(regions());changed['source']=[0,0,2,2]
        engine=StationVision(CPUVision(),changed,[10,8],0)
        data.update(regions=changed,config_sha256=engine.config_sha256)
        data['evidence']['config_sha256']=engine.config_sha256
        with patch('src.monitor_station_vision.board_request',return_value=data):
            self.assertFalse(station_vision_status(self.path)['available'])
            result=station_vision_status(self.path,observe_runtime_regions=True)
            self.assertTrue(result['available'],result);self.assertTrue(result['runtime_regions_observed'])
            self.assertFalse(result['automatic_motion_allowed']);self.assertFalse(result['region_geometry_qualified'])
            with self.assertRaises(ValueError):
                bind_station_baseline(dict(stream_id=data['stream_id'],config_sha256=engine.config_sha256,frame_id=2),self.path)
        for update in (dict(assets={}),dict(config_sha256='forged'),dict(threshold=2)):
            with patch('src.monitor_station_vision.board_request',return_value=dict(data,**update)):
                self.assertFalse(station_vision_status(self.path,observe_runtime_regions=True)['available'])

    def test_baseline_forwarding_has_no_extra_fields_or_motion(self):
        body=dict(stream_id=self.data['stream_id'],config_sha256=self.data['config_sha256'],frame_id=2,execute=True)
        with patch('src.monitor_station_vision.board_request',side_effect=[self.data,dict(automatic_motion_allowed=False)]) as request:
            self.assertFalse(bind_station_baseline(body,self.path)['automatic_motion_allowed'])
            self.assertEqual(set(request.call_args.args[1]),{'stream_id','frame_id','config_sha256'})
        with patch('src.monitor_station_vision.board_request',return_value=self.data):
            with self.assertRaises(ValueError):bind_station_baseline(dict(body,stream_id='old'),self.path)

    def test_missing_configuration_does_not_start_hardware(self):
        with patch('src.monitor_station_vision.board_request') as request:
            self.assertFalse(station_vision_status(self.path.parent/'missing')['available']);request.assert_not_called()

    def test_pc_reference_only_matches_the_exact_board_startup_report(self):
        startup=dict(samples=12,assets=self.assets,baseline_sha256='input')
        report=dict(board_startup_summary=startup,same_input_verified=True,samples=12,assets=self.assets,
            cpu='HX370 reference',cpu_median_ms=15.,host_measured_at='2026-10-29T11:53:19Z')
        self.path.write_text(json.dumps(report));self.assertEqual(pc_reference_summary(startup,self.path)['cpu_median_ms'],15.)
        self.assertIsNone(pc_reference_summary(dict(startup,baseline_sha256='other'),self.path))
        report['cpu_median_ms']=float('nan');self.path.write_text(json.dumps(report))
        self.assertIsNone(pc_reference_summary(startup,self.path))

    def test_saved_acceleration_recomputes_times_and_rejects_unbound_or_invalid_evidence(self):
        folder=self.path.parent;active=folder/'active.json';board=folder/'board.json';binding=folder/'binding.json'
        manifest=dict(assets=self.assets,config_sha256=self.data['config_sha256'])
        active.write_text(json.dumps(manifest))
        report=dict(assets=self.assets,backend='pl_board_local',source='synthetic_board_startup',all_equal=True,cpu='board_ARM_numpy_reference',
            image_size=[10,8],threshold=0,samples=3,measurements=[dict(cpu_ms=100,pl_ms=40),
            dict(cpu_ms=90,pl_ms=30),dict(cpu_ms=80,pl_ms=20)],median_speedup=999)
        def save_report(value):
            board.write_text(json.dumps(value))
            binding.write_text(json.dumps(dict(manifest,report_sha256=hashlib.sha256(board.read_bytes()).hexdigest())))
        def load():return saved_acceleration_summary(self.path,active,board,folder/'missing-pc.json',binding)
        save_report(report)
        result=load()
        self.assertEqual(result['median_speedup'],3);self.assertAlmostEqual(result['latency_reduction_percent'],200/3)
        self.assertFalse(result['live_speedup']);self.assertFalse(result['whole_system_speedup_measured'])
        self.assertIsNone(result['pc_reference'])
        for change in (dict(assets={}),dict(all_equal=False),dict(samples=2),dict(threshold=1),dict(cpu='HX370'),
            dict(measurements=[dict(cpu_ms=float('nan'),pl_ms=20)]*3)):
            with self.subTest(change=change):
                save_report(dict(report,**change))
                self.assertFalse(load()['available'])
        save_report(report);board.write_text(json.dumps(dict(report,samples=4)))
        self.assertFalse(load()['available'])
        save_report(report);active.write_text(json.dumps(dict(manifest,config_sha256='old')))
        self.assertFalse(load()['available'])


if __name__=='__main__':unittest.main()
