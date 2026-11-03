import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.error
from src.km1_voice_dashboard import DashboardVoiceClient, VoiceRecords, ROUTES
from src.km1_voice_run import VoiceSession
from src.km1_voice import request_frame
from src.monitor_operation import Operations
from src.sorting_monitor_server import make_server


class Backend:
    unified=True
    def __init__(self):self.calls=[];self.stops=0;self.block=False;self.release=threading.Event()
    def check(self):return []
    def cycle(self, material, execute):
        self.calls.append((material,execute))
        if self.block:self.release.wait(3)
        return dict(status='preview_only',task_id='preview')
    def stop(self):self.stops+=1


class DashboardVoiceTests(unittest.TestCase):
    def setUp(self):
        self.backend=Backend();self.operations=Operations(self.backend)
        self.server=make_server(0,'http://127.0.0.1:1',operations=self.operations)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.temp=tempfile.TemporaryDirectory();self.records=VoiceRecords(Path(self.temp.name))
        self.fallback=[]
        self.client=DashboardVoiceClient('http://127.0.0.1:'+str(self.server.server_port),self.records,
                                         lambda:self.fallback.append('stop'))

    def tearDown(self):
        self.backend.release.set();self.server.shutdown();self.thread.join();self.server.server_close();self.temp.cleanup()

    def test_four_materials_use_shared_interfaces_without_motion_in_preview(self):
        for material in ROUTES:
            result=self.client.run(material,False,False)
            self.assertEqual(result['voice_transport_reply'],'PREVIEW')
        self.assertEqual(self.backend.calls,[(m,False) for m in ROUTES])

    def test_foreign_web_operation_is_busy_not_stopped_or_queued(self):
        self.backend.block=True
        self.operations.command(dict(token=self.operations.token,action='preview',owner='web-page',
            request_id='web-request'))
        self.assertEqual(self.client.run('PAPER',True,True),dict(voice_transport_reply='BUSY'))
        self.assertEqual(self.backend.stops,0);self.assertEqual(self.fallback,[])

    def test_ready_gate_and_invalid_material_precede_motion(self):
        for material,ready in (('UNKNOWN',True),('PAPER',False)):
            with self.assertRaises(ValueError):self.client.run(material,True,ready)
        self.assertEqual(self.backend.calls,[])

    def test_server_restart_is_not_silently_reauthorized(self):
        self.operations.token='a-new-process-token'
        with self.assertRaisesRegex(ValueError,'restarted'):self.client.latest()
        self.assertEqual(self.backend.calls,[])

    def test_http_409_race_does_not_stop_another_owner(self):
        import io
        reply=json.dumps(dict(operation=dict(busy=True,owner='web-page'))).encode()
        exc=urllib.error.HTTPError('local',409,'busy',{},io.BytesIO(reply))
        with patch.object(self.client,'_command',side_effect=exc):
            self.assertEqual(self.client.run('AUTO',True,True)['voice_transport_reply'],'BUSY')
        self.assertEqual(self.fallback,[])

    def test_ambiguous_start_failure_is_never_retried_and_requests_stop(self):
        calls=[]
        def command(action,*args,**kwargs):
            calls.append(action)
            if action=='start':raise TimeoutError('lost response')
            return {}
        with patch.object(self.client,'_command',side_effect=command):
            with self.assertRaises(TimeoutError):self.client.run('AUTO',True,True)
        self.assertEqual(calls,['start','stop']);self.assertEqual(self.fallback,['stop'])

    def test_stop_uses_shared_stop_and_persistent_fallback(self):
        self.client.stop()
        self.assertEqual(self.fallback,['stop']);self.assertEqual(self.backend.stops,1)
        self.assertTrue(self.client.cancelled.is_set())

    def test_nonlocal_or_old_endpoint_is_rejected(self):
        for endpoint in ('https://127.0.0.1:8789','http://example.com:8789','http://127.0.0.1:8789/x'):
            with self.assertRaises(ValueError):DashboardVoiceClient(endpoint,self.records)
        self.operations.backend.unified=False
        with self.assertRaisesRegex(ValueError,'version_mismatch'):
            DashboardVoiceClient(self.client.endpoint,self.records)

    def test_transport_can_never_inject_a_success_clip(self):
        session=VoiceSession(lambda *_:dict(voice_transport_reply='PAPER_PASS'),lambda:None,lambda:None)
        session.handle(request_frame('PAPER'));session.worker.join(1)
        self.assertEqual(session.replies.get_nowait(),'NO_RESULT')


class VoiceRecordTests(unittest.TestCase):
    def test_batch_success_reads_only_hash_bound_physical_record(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);reader=VoiceRecords(folder)
            task=dict(task_id='physical',batch_id='batch',mode='known_physical',
                status='completed_visual_evidence',choice=dict(label='metal'))
            raw=json.dumps(task).encode();(folder/'physical.json').write_bytes(raw)
            batch=dict(batch_id='batch',status='completed',items=[dict(task_id='physical',
                physical_motion_attempted=True,report_sha256=hashlib.sha256(raw).hexdigest())])
            (folder/'batch.batch.json').write_text(json.dumps(batch))
            self.assertEqual(reader.selected(dict(batch_id='batch',status='completed')),task)
            (folder/'physical.json').write_bytes(raw+b' ')
            with self.assertRaisesRegex(ValueError,'hash_mismatch'):
                reader.selected(dict(batch_id='batch',status='completed'))

    def test_invalid_identifier_and_unrelated_batch_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            reader=VoiceRecords(Path(directory))
            with self.assertRaises(ValueError):reader.read('../other')
            (Path(directory)/'batch.batch.json').write_text(json.dumps(dict(batch_id='other',status='completed')))
            with self.assertRaisesRegex(ValueError,'mismatch'):reader.selected(dict(batch_id='batch',status='completed'))


if __name__=='__main__':unittest.main()
