"""A permanent listener must never become permanent motion authorization."""
import queue
from pathlib import Path
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.km1_voice import request_frame
from src.km1_voice_resident import ResidentVoice
from src.monitor_operation import Operations


class Backend:
    def __init__(self):
        self.calls=[];self.stops=0;self.blockers=[]
        self.hold=False;self.release=threading.Event();self.entered=threading.Event()
    def check(self):return self.blockers
    def cycle(self,material,execute):
        self.calls.append((material,execute));self.entered.set()
        if self.hold:self.release.wait(3)
        return dict(status='empty_scene',physical_motion_attempted=False)
    def stop(self):self.stops+=1;self.blockers=['operator_stop_requested']


class Port:
    port='COM10'
    in_waiting=0
    def __init__(self):self.closed=False;self.writes=[];self.input=queue.Queue()
    def read(self,_):
        try:return self.input.get(timeout=.01)
        except queue.Empty:return b''
    def write(self,data):self.writes.append(data);return len(data)
    def flush(self):pass
    def close(self):self.closed=True


class VoiceTestBase(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.backend=Backend();self.ops=Operations(self.backend)
        self.port=Port();folder=Path(self.temp.name)
        self.voice=ResidentVoice(self.ops,folder,folder/'events.jsonl',open_port=lambda:self.port)
        self.ops.voice_connection(True)
    def tearDown(self):
        self.backend.release.set();self.voice.close();self.ops.close();self.temp.cleanup()
    def arm(self,identifier='grant-one',material='AUTO',owner='operator-page',ready=True):
        return self.ops.command(dict(token=self.ops.token,action='voice_arm',owner=owner,
            request_id=identifier,ready=ready,material=material))
    def wait(self,predicate):
        end=time.monotonic()+2
        while not predicate():
            if time.monotonic()>end:self.fail('condition timeout')
            time.sleep(.005)


class ResidentTests(VoiceTestBase):
    def test_startup_and_repeated_words_do_not_grant_ready(self):
        for word in ('PAPER','AUTO','PLASTIC'):
            self.voice.handle(request_frame(word));self.assertEqual(self.voice.replies.get_nowait(),'REJECTED')
        self.assertEqual(self.backend.calls,[])
        with self.assertRaisesRegex(ValueError,'ready'):self.arm(ready=False)
    def test_two_batches_keep_listener_and_require_two_distinct_page_grants(self):
        for index in range(2):
            self.arm('grant-batch-'+str(index));self.assertEqual(len(self.backend.calls),index)
            self.voice.handle(request_frame('PAPER'))
            self.wait(lambda:len(self.ops.voice_completed)>0)
            self.voice.poll_result();self.assertEqual(self.voice.replies.get_nowait(),'EMPTY')
            self.assertFalse(self.ops.status()['voice_service']['armed'])
            self.voice.handle(request_frame('PAPER'))
            self.assertEqual(self.voice.replies.get_nowait(),'REJECTED')
        self.assertEqual(self.backend.calls,[('PAPER',True),('PAPER',True)])
        self.assertTrue(self.ops.voice_connected)
    def test_grant_replay_and_foreign_heartbeat_do_not_rearm(self):
        self.arm();self.voice.handle(request_frame('AUTO'))
        self.wait(lambda:self.ops.state=='finished')
        with self.assertRaisesRegex(ValueError,'duplicate'):self.arm()
        with self.assertRaisesRegex(ValueError,'owner'):
            self.ops.command(dict(token=self.ops.token,action='heartbeat',owner='other-page',operation_id=self.ops.operation_id))
    def test_first_wrong_category_consumes_scope_without_motion(self):
        self.arm(material='PAPER');self.voice.handle(request_frame('METAL'))
        self.assertEqual(self.voice.replies.get_nowait(),'REJECTED')
        self.voice.handle(request_frame('PAPER'));self.assertEqual(self.voice.replies.get_nowait(),'REJECTED')
        self.assertEqual(self.backend.calls,[])
    def test_expired_page_grant_never_moves(self):
        for kind in ('heartbeat','deadline'):
            self.arm('grant-'+kind)
            if kind=='heartbeat':self.ops.last_heartbeat-=100
            else:self.ops.deadline-=1000
            self.voice.handle(request_frame('AUTO'))
            self.assertEqual(self.voice.replies.get_nowait(),'REJECTED')
        self.assertEqual(self.backend.calls,[]);self.assertEqual(self.backend.stops,0)
    def test_station_blocker_is_rechecked_at_utterance(self):
        self.arm();self.backend.blockers=['unresolved_previous_task']
        self.voice.handle(request_frame('AUTO'))
        self.assertEqual(self.voice.replies.get_nowait(),'REJECTED');self.assertEqual(self.backend.calls,[])
    def test_disconnect_cancels_grant_and_reconnect_does_not_rearm(self):
        self.arm();self.ops.voice_connection(False);self.ops.voice_connection(True)
        self.voice.handle(request_frame('AUTO'))
        self.assertEqual(self.voice.replies.get_nowait(),'REJECTED');self.assertEqual(self.backend.stops,0)
    def test_disconnect_during_own_motion_persists_stop(self):
        self.backend.hold=True;self.arm();self.voice.handle(request_frame('AUTO'))
        self.assertTrue(self.backend.entered.wait(1))
        self.ops.voice_connection(False)
        self.assertEqual(self.backend.stops,1)
        self.ops.voice_connection(True);self.backend.release.set()
        self.wait(lambda:self.ops.state=='stopped')
        with self.assertRaisesRegex(ValueError,'station_not_ready'):self.arm('new-grant')
    def test_busy_word_is_not_queued_and_page_remains_motion_owner(self):
        self.backend.hold=True;self.arm();owner=self.ops.owner
        self.voice.handle(request_frame('AUTO'));self.assertTrue(self.backend.entered.wait(1))
        self.voice.handle(request_frame('METAL'));self.assertEqual(self.voice.replies.get_nowait(),'BUSY')
        self.assertEqual(self.ops.owner,owner);self.assertEqual(len(self.backend.calls),1)
    def test_hardware_stop_is_global_and_never_cleared(self):
        self.voice.handle(request_frame('STOP'));self.assertEqual(self.voice.replies.get_nowait(),'STOPPED')
        self.assertEqual(self.backend.stops,1)
        with self.assertRaisesRegex(ValueError,'station_not_ready'):self.arm()
    def test_listener_keeps_running_after_a_reply_without_one_hour_limit(self):
        self.voice.start();self.wait(lambda:self.ops.voice_connected)
        self.port.input.put(request_frame('REPORT'))
        self.wait(lambda:bool(self.port.writes))
        self.assertTrue(self.voice.thread.is_alive())
        self.assertEqual(self.port.writes,[b'$KSP1:NO_RESULT!'])
        self.voice.close();self.assertTrue(self.port.closed)
    def test_completion_survives_page_rearming_before_audio_reply(self):
        self.arm();self.voice.handle(request_frame('AUTO'))
        self.wait(lambda:bool(self.ops.voice_completed))
        self.arm('next-grant')
        self.voice.poll_result();self.assertEqual(self.voice.replies.get_nowait(),'EMPTY')
        self.assertTrue(self.ops.status()['voice_service']['armed'])
    def test_preview_record_cannot_be_announced_as_physical_success(self):
        self.voice.pending='fake'
        self.ops.voice_completed['fake']=dict(state='finished',result=dict(preview=True,status='completed_visual_evidence'))
        self.voice.poll_result();self.assertEqual(self.voice.replies.get_nowait(),'REJECTED')
    def test_serial_open_checks_unique_identity_and_clears_buffer(self):
        fake=SimpleNamespace(port=None,open=lambda:None,close=lambda:None,reset_input_buffer=lambda:None)
        with patch('serial.Serial',return_value=fake),patch('serial.tools.list_ports.comports',return_value=[
                SimpleNamespace(vid=0x1a86,pid=0x7522,device='COM10')]):
            with patch.object(self.voice.done,'wait',return_value=False):self.voice._open_serial()
            self.assertEqual(fake.port,'COM10');self.assertFalse(fake.dtr);self.assertFalse(fake.rts)
        with patch('serial.tools.list_ports.comports',return_value=[
                SimpleNamespace(vid=0x1a86,pid=0x7523,device='COM9')]):
            with self.assertRaisesRegex(ValueError,'unique_reviewed'):self.voice._open_serial()


class DirectVoiceTests(VoiceTestBase):
    """A fresh spoken request directly dispatches through the original guards."""
    def setUp(self):
        super().setUp()
        self.voice.direct_commands=True

    def test_direct_startup_does_not_submit_motion(self):
        self.voice.start();time.sleep(.05)
        self.assertEqual(self.backend.calls,[])

    def test_direct_two_fresh_utterances_need_no_page_between_batches(self):
        identifiers=[]
        for material in ('PAPER','METAL'):
            self.voice.handle(request_frame(material));identifiers.append(self.voice.pending)
            self.wait(lambda:bool(self.ops.voice_completed))
            self.voice.poll_result();self.assertEqual(self.voice.replies.get_nowait(),'EMPTY')
        self.assertEqual(self.backend.calls,[('PAPER',True),('METAL',True)])
        self.assertNotEqual(identifiers[0],identifiers[1]);self.assertEqual(self.ops.owner,self.voice.owner)

    def test_direct_requires_fresh_connection_and_runtime_checks(self):
        self.ops.voice_connection(False)
        self.voice.handle(request_frame('AUTO'));self.assertEqual(self.voice.replies.get_nowait(),'OFFLINE')
        self.ops.voice_connection(True);self.backend.blockers=['unresolved_previous_task']
        self.voice.handle(request_frame('AUTO'));self.assertEqual(self.voice.replies.get_nowait(),'REJECTED')
        self.assertEqual(self.backend.calls,[])

    def test_direct_heartbeat_does_not_require_an_open_page(self):
        self.ops.lease_s=.08;self.backend.hold=True
        self.voice.start();self.port.input.put(request_frame('AUTO'))
        self.assertTrue(self.backend.entered.wait(1));time.sleep(.4)
        self.assertEqual(self.backend.stops,0);self.assertEqual(self.ops.state,'executing')
        self.backend.release.set();self.wait(lambda:bool(self.port.writes))

    def test_direct_does_not_take_ownership_from_a_web_task(self):
        self.backend.hold=True
        self.ops.command(dict(token=self.ops.token,owner='web-page',request_id='web-request',
                              action='start',ready=True,material='PAPER'))
        self.assertTrue(self.backend.entered.wait(1))
        heartbeat=self.ops.last_heartbeat
        self.voice.handle(request_frame('AUTO'));self.assertEqual(self.voice.replies.get_nowait(),'BUSY')
        self.voice.heartbeat()
        self.assertEqual(self.ops.owner,'web-page');self.assertEqual(self.ops.last_heartbeat,heartbeat)
        self.assertEqual(len(self.backend.calls),1)

    def test_direct_stop_is_latched_and_not_cleared_by_another_word(self):
        self.voice.handle(request_frame('STOP'));self.assertEqual(self.voice.replies.get_nowait(),'STOPPED')
        self.voice.handle(request_frame('AUTO'));self.assertEqual(self.voice.replies.get_nowait(),'REJECTED')
        self.assertEqual(self.backend.calls,[]);self.assertEqual(self.backend.stops,1)

    def test_direct_reconnect_does_not_replay_but_accepts_a_new_word(self):
        self.ops.voice_connection(False);self.ops.voice_connection(True)
        self.assertEqual(self.backend.calls,[])
        self.voice.handle(request_frame('PLASTIC'));self.wait(lambda:bool(self.ops.voice_completed))
        self.voice.poll_result();self.assertEqual(self.voice.replies.get_nowait(),'EMPTY')
        self.assertEqual(self.backend.calls,[('PLASTIC',True)])




if __name__=='__main__':unittest.main()
