import json
from pathlib import Path
import sys
import threading
import unittest
import urllib.request

sys.path.insert(0,str(Path(__file__).parents[1]))
sys.path.insert(0,str(Path(__file__).parents[2]/'pynq_km1_vision'))
from cloud_console_bridge import Bridge
from local_console import validate_command
from src.sorting_monitor_server import make_server


def measured_plan():
    # More than 2 KB, representing a real full-frame RLE rather than a scalar.
    return dict(kind='instance-heights-v1',model_sha256='a'*64,configuration_sha256='b'*64,
        source_stream_id='stream',image_size=[640,480],items=[dict(item_id=1,label='metal',
        official_label='Drink can',object_height_mm=56,mask_rle=[0]+[100]*3072)])


class HeightTransportTests(unittest.TestCase):
    def test_registration_requires_measurements_and_scene_confirmation(self):
        payload=dict(action='register_heights',owner='cloud-00000000-0000-0000-0000-000000000000',
            request_id='registration-001',height_plan=measured_plan(),ready=True,height_registration_id=None)
        validate_command(dict(kind='operation',payload=payload))
        for values in ({'ready':False},{'height_plan':None},{'request_id':None}):
            with self.assertRaises(ValueError):validate_command(dict(kind='operation',payload=dict(payload,**values)))

    def test_register_and_revoke_cross_bridge_without_start(self):
        bridge=Bridge(dict(site_url='http://127.0.0.1'));calls=[]
        def request(path,payload=None,**kw):
            if payload is None:return dict(token='page',operation_id='old-task')
            calls.append(payload);return dict(voice_height_registration={'id':'registration'})
        bridge.local_request=request
        try:
            for action in ('register_heights','clear_heights'):
                payload=dict(action=action,expected_operation_id='old-task',height_plan=measured_plan(),
                    height_registration_id='registration' if action=='clear_heights' else None)
                result=bridge.execute(dict(id=action,kind='operation',payload=payload))
                self.assertTrue(result['ok'],result)
            self.assertEqual([v['action'] for v in calls],['register_heights','clear_heights'])
            self.assertEqual(calls[0]['height_plan'],measured_plan())
        finally:bridge.pool.shutdown(wait=False,cancel_futures=True)

    def test_local_validator_accepts_plan_and_rejects_conflicting_scalar(self):
        payload=dict(action='preview',material='AUTO',owner='cloud-00000000-0000-0000-0000-000000000000',
                     request_id='height-preview',height_plan=measured_plan())
        validate_command(dict(kind='operation',payload=payload))
        with self.assertRaises(ValueError):validate_command(dict(kind='operation',payload=dict(payload,object_height_mm=56)))

    def test_bridge_to_station_preserves_large_plan_with_one_submission(self):
        calls=[]
        class Operations:
            backend=None
            def status(self):return dict(token='page',operation_id=None)
            def command(self,data):calls.append(data);return dict(status='preview_only')
            def close(self):pass
        server=make_server(0,operations=Operations())
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        bridge=Bridge(dict(site_url='http://127.0.0.1',local_url='http://127.0.0.1:'+str(server.server_port)))
        try:
            command=dict(id='once',kind='operation',payload=dict(action='preview',material='AUTO',height_plan=measured_plan(),expected_operation_id=None))
            command['payload']['height_plan']['items'][0]['grasp_width_mm']=10
            command['payload']['height_plan']['items'][0]['measurement_profile']=dict(id='a'*32,revision=1,name='measured can',pose='flat')
            self.assertGreater(len(json.dumps(command)),2048)
            result=bridge.execute(command)
            self.assertTrue(result['ok'],result)
            self.assertEqual(calls[0]['height_plan'],command['payload']['height_plan'])
            self.assertEqual(calls[0]['token'],'page')
            self.assertFalse(bridge.execute(command)['ok'])
            self.assertEqual(len(calls),1)
        finally:
            server.shutdown();server.server_close();thread.join();bridge.pool.shutdown(wait=False,cancel_futures=True)


if __name__=='__main__':unittest.main()
