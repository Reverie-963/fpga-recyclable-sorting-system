import base64
import importlib.util
import json
from pathlib import Path
import re
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
import uuid
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1]))
spec = importlib.util.spec_from_file_location('local_console', Path(__file__).parents[1] / 'local_console.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class LocalConsoleTests(unittest.TestCase):
    def setUp(self):
        self.bridge = module.Bridge({'site_url': 'http://127.0.0.1'})
        self.bridge.snapshot = {'models': {'revision': 7}, 'operation': {'operation_id': 'old'}, 'camera': {'base64': base64.b64encode(b'frame').decode(), 'age_ms': 0}}
        self.bridge.snapshot_at = time.monotonic()
        self.calls = []

        def request(path, payload=None, timeout=3):
            self.calls.append((path, payload))
            if path == '/operation' and payload is None:
                return {'operation_id': 'old', 'owner': 'original', 'token': 'private'}
            if path == '/model-switch/status':
                return {'revision': 7, 'token': 'private'}
            return {'available': True, 'token': 'private', 'task_id': 'record'}

        self.bridge.local_request = request
        self.temp=tempfile.TemporaryDirectory()
        self.server = module.create_server(0, bridge=self.bridge,profile_path=Path(self.temp.name)/'profiles.json')
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.origin = 'http://127.0.0.1:' + str(self.server.server_port)
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with self.opener.open(self.origin) as response:
            html = response.read().decode()
        self.token = json.loads(re.search(r'window.__LOCAL_CONSOLE__=(.*?);', html).group(1))['token']

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.bridge.pool.shutdown(wait=False, cancel_futures=True)
        self.temp.cleanup()

    def call(self, path='/api/local-command', body=None, headers=None):
        h = {'X-Console-Session': self.token, 'Origin': self.origin, 'Content-Type': 'application/json'}
        h.update(headers or {})
        request = urllib.request.Request(self.origin + path, headers=h, data=None if body is None else json.dumps(body).encode())
        try:
            with self.opener.open(request) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()

    def task(self):
        return {'id': str(uuid.uuid4()), 'kind': 'task', 'payload': {'task_id': 'record'}}

    def test_foreign_host_origin_and_missing_session_cannot_submit(self):
        for headers in ({'Host': 'foreign.example'}, {'Origin': 'https://foreign.example'}, {'X-Console-Session': ''}):
            self.assertEqual(self.call(body=self.task(), headers=headers)[0], 403)
        self.assertEqual(self.calls, [])

    def test_local_task_is_read_only_and_tokens_never_return(self):
        status, raw = self.call(body=self.task())
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(raw)['ok'])
        self.assertNotIn(b'private', raw)
        self.assertEqual(self.calls, [('/control-status?task_id=record', None)])

    def test_fresh_recovery_status_is_read_only_and_session_protected(self):
        status, raw = self.call('/api/operation')
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)['operation_id'], 'old')
        self.assertNotIn(b'private', raw)
        self.assertEqual(self.call('/api/operation', headers={'X-Console-Session': ''})[0], 403)
        self.assertEqual(self.calls, [('/operation', None)])

    def test_duplicate_http_request_does_not_replay(self):
        command = self.task()
        self.assertTrue(json.loads(self.call(body=command)[1])['ok'])
        self.assertTrue(json.loads(self.call(body=command)[1])['unknown'])
        self.assertEqual(len(self.calls), 1)

    def test_stale_station_never_reaches_local_api(self):
        self.bridge.snapshot_at -= 8
        self.assertEqual(self.call(body=self.task())[0], 503)
        self.assertEqual(self.calls, [])

    def test_injected_fields_and_unconfirmed_start_are_rejected(self):
        owner = 'cloud-' + str(uuid.uuid4())
        bad = [dict(action='start', owner=owner, ready=False, request_id='one'), dict(action='preview', owner=owner, request_id='one', token='inject')]
        for payload in bad:
            command = {'id': str(uuid.uuid4()), 'kind': 'operation', 'payload': payload}
            self.assertEqual(self.call(body=command)[0], 400)
        self.assertEqual(self.calls, [])

    def measured_command(self,action):
        plan=dict(kind='instance-heights-v1',model_sha256='a'*64,configuration_sha256='b'*64,
            source_stream_id='stream',image_size=[2,2],items=[dict(item_id=1,label='plastic',official_label='Plastic straw',
            mask_rle=[0,1,3],object_height_mm=10,grasp_width_mm=10)])
        return dict(id=str(uuid.uuid4()),kind='operation',payload=dict(action=action,owner='cloud-'+str(uuid.uuid4()),
            request_id=uuid.uuid4().hex,ready=action=='start',expected_operation_id='old',height_plan=plan))

    def test_missing_measurement_dependency_returns_json_and_never_submits(self):
        with patch.dict(sys.modules,{'src.sorting_item_heights':None}):
            for action in ('preview','start'):
                status,raw=self.call(body=self.measured_command(action))
                self.assertEqual(status,503);reply=json.loads(raw)
                self.assertIs(reply['submitted'],False);self.assertIn('本次操作未提交',reply['error'])
        self.assertEqual(self.calls,[])

    def test_measured_preview_and_start_cross_real_http_once_into_stub_station(self):
        for action in ('preview','start'):
            command=self.measured_command(action);status,raw=self.call(body=command)
            self.assertEqual(status,200);self.assertTrue(json.loads(raw)['ok'])
            writes=[body for path,body in self.calls if body]
            self.assertEqual(len(writes),1);self.assertEqual(writes[0]['height_plan'],command['payload']['height_plan'])
            self.assertEqual(writes[0]['action'],action);self.calls.clear()

    def test_review_preserves_confirmation_and_identity_and_rejects_unconfirmed(self):
        payload=dict(action='review',owner='cloud-'+str(uuid.uuid4()),request_id=uuid.uuid4().hex,expected_operation_id='old',
            review_epoch='epoch-current',review_request_id='failed-task',ready=True,review_confirmed=True)
        command=dict(id=str(uuid.uuid4()),kind='operation',payload=payload)
        status,raw=self.call(body=command);self.assertEqual(status,200);self.assertTrue(json.loads(raw)['ok'])
        writes=[p for path,p in self.calls if p];self.assertEqual(len(writes),1)
        self.assertEqual(writes[0]['review_request_id'],'failed-task');self.assertTrue(writes[0]['review_confirmed'])
        self.calls.clear();payload['review_confirmed']=False;command['id']=str(uuid.uuid4())
        self.assertEqual(self.call(body=command)[0],400);self.assertEqual(self.calls,[])

    def test_stop_can_pass_while_nonurgent_request_is_running(self):
        entered, release = threading.Event(), threading.Event()
        original = self.bridge.local_request

        def slow(path, payload=None, timeout=3):
            if path.startswith('/control-status'):
                entered.set()
                release.wait(3)
            return original(path, payload, timeout)

        self.bridge.local_request = slow
        job = threading.Thread(target=lambda: self.call(body=self.task()))
        job.start()
        try:
            self.assertTrue(entered.wait(2))
            self.assertEqual(self.call(body=self.task())[0], 409)
            stop = {'id': str(uuid.uuid4()), 'kind': 'operation', 'payload': {'action': 'stop', 'owner': 'cloud-' + str(uuid.uuid4())}}
            self.assertTrue(json.loads(self.call(body=stop)[1])['ok'])
            writes = [payload for path, payload in self.calls if payload]
            self.assertEqual([p['action'] for p in writes], ['stop'])
        finally:
            release.set()
            job.join()

    def test_frame_freshness_and_cross_site_reads(self):
        self.assertEqual(self.call('/api/frame?kind=camera')[1], b'frame')
        self.assertEqual(self.call('/api/frame?kind=camera', headers={'Sec-Fetch-Site': 'cross-site'})[0], 403)
        self.bridge.snapshot_at -= 8
        self.assertEqual(self.call('/api/frame?kind=camera')[0], 503)

    def test_profile_api_is_persistent_session_protected_and_never_sends_robot_command(self):
        from test_item_profile_store import profile
        self.assertEqual(self.call('/api/item-profiles',headers={'X-Console-Session':''})[0],403)
        command=dict(action='save',expected_revision=0,profile=profile())
        self.assertEqual(self.call('/api/item-profiles',body=command,headers={'Origin':'https://foreign.example'})[0],403)
        status,raw=self.call('/api/item-profiles',body=command)
        self.assertEqual(status,200);saved=json.loads(raw)
        self.assertEqual(saved,json.loads(self.call('/api/item-profiles')[1]))
        self.assertEqual(self.call('/api/item-profiles',body=command)[0],400)
        self.assertEqual(self.calls,[])


if __name__ == '__main__':
    unittest.main()
