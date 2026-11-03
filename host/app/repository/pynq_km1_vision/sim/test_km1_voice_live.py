import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from src.km1_voice import request_frame
from src.km1_voice_run import VoiceSession
from src.km1_voice_live import LiveControl


class Client:
    def __init__(self,endpoint,*args):
        self.endpoint=endpoint;self.calls=[];self.stops=0
    def _status(self):return dict(configured=True,blockers=[])
    def run(self,*args):self.calls.append(args);return dict(voice_transport_reply='EMPTY')
    def latest(self):return dict(mode='known_physical',status='needs_review')
    def stop(self):self.stops+=1


class LiveVoiceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.created=[];self.endpoints=['http://127.0.0.1:8809']
        def factory(endpoint,*args):
            client=Client(endpoint);self.created.append(client);return client
        self.control=LiveControl(self.root,candidates=lambda:list(self.endpoints),client_factory=factory)
        self.config=patch.object(self.control,'configuration',return_value=(self.root/'config/c.json',dict(journal_dir='../local-only/ledger',known=dict(stop_files=['../local-only/STOP_SORTING']))))
        self.config.start()
    def tearDown(self):self.config.stop();self.temp.cleanup()
    def test_each_fresh_word_uses_current_controller_without_page(self):
        session=VoiceSession(self.control.run,self.control.latest,self.control.stop,True,True,repeat_requests=True)
        for material,endpoint in (('PAPER','http://127.0.0.1:8809'),('METAL','http://127.0.0.1:8810')):
            self.endpoints=[endpoint];session.handle(request_frame(material));session.worker.join(2)
            self.assertEqual(session.replies.get_nowait(),'EMPTY')
            self.assertEqual(self.created[-1].endpoint,endpoint)
            self.assertEqual(self.created[-1].calls,[(material,True,True)])
    def test_stale_source_controller_is_never_used(self):
        self.endpoints=['http://127.0.0.1:8808','http://127.0.0.1:8809']
        def factory(endpoint,*args):
            c=Client(endpoint)
            if endpoint.endswith('8808'):c._status=lambda:dict(configured=True,blockers=['station_evidence_changed'])
            return c
        self.control.client_factory=factory
        self.assertTrue(self.control.select().endpoint.endswith('8809'))
    def test_pending_review_stays_with_controller_instead_of_being_cleared(self):
        def factory(endpoint,*args):
            c=Client(endpoint);c._status=lambda:dict(configured=True,blockers=['existing_physical_trial_unresolved'])
            c.run=lambda *args:dict(voice_transport_reply='REJECTED')
            return c
        self.control.client_factory=factory
        self.assertEqual(self.control.run('AUTO',True,True),dict(voice_transport_reply='REJECTED'))
        self.assertFalse((self.root/'local-only/STOP_SORTING').exists())
    def test_stop_during_discovery_prevents_start(self):
        original=self.control.select
        def select():self.control.stop();return original()
        with patch.object(self.control,'select',side_effect=select):
            self.assertEqual(self.control.run('AUTO',True,True),dict(voice_transport_reply='STOPPED'))
        self.assertEqual(self.created[-1].calls,[])
        self.assertTrue((self.root/'local-only/STOP_SORTING').exists())
    def test_stop_is_persistent_and_new_word_cannot_clear_it(self):
        self.control.stop()
        before=(self.root/'local-only/STOP_SORTING').read_bytes()
        self.control.run('AUTO',True,True)
        self.assertEqual((self.root/'local-only/STOP_SORTING').read_bytes(),before)
    def test_continuous_session_does_not_retry_failed_request(self):
        calls=[]
        def run(*args):calls.append(args);raise OSError('offline')
        s=VoiceSession(run,lambda:None,lambda:None,True,True,repeat_requests=True)
        s.handle(request_frame('AUTO'));s.worker.join(1)
        self.assertEqual(calls,[('AUTO',True,True)])
        self.assertEqual(s.replies.get_nowait(),'REJECTED')
        s.handle(request_frame('AUTO'));s.worker.join(1)
        self.assertEqual(len(calls),2)
    def test_discovery_only_uses_local_manual_controller_processes(self):
        data=[
            dict(pid=1,create_time=1,cwd=str(self.root/'pynq_km1_vision'),cmdline=['python','-m','src.sorting_monitor_server','--port','8809']),
            dict(pid=2,create_time=2,cwd=str(self.root/'pynq_km1_vision'),cmdline=['python','-m','src.sorting_monitor_server','--port','8793','--voice-resident']),
            dict(pid=3,create_time=3,cwd=str(self.root/'another'),cmdline=['python','-m','src.sorting_monitor_server','--port','8810'])]
        from types import SimpleNamespace
        self.control.candidates=None
        with patch('psutil.process_iter',return_value=[SimpleNamespace(info=d) for d in data]):
            self.assertEqual(self.control.controller_candidates(),['http://127.0.0.1:8809'])


if __name__=='__main__':unittest.main()
