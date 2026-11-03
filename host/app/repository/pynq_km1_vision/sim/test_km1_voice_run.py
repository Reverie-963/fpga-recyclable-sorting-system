"""Voice serial sessions must not bypass one-cycle motion and result gates."""
import threading
import unittest
from unittest.mock import patch

from src.km1_voice import request_frame
from src.km1_voice_run import Frames, VoiceSession
from src.km1_voice_station import VoiceController


def passed(label='paper'):
    return dict(mode='known_physical',status='completed_visual_evidence',choice={'label':label})


class VoiceRunTests(unittest.TestCase):
    def test_fragmented_serial_and_noise_never_become_arbitrary_commands(self):
        f=Frames()
        self.assertEqual(f.feed(b'boot log\r\n@KSV1:PA'),[])
        self.assertEqual(f.feed(b'PER!\r\n$DGT:1-2,1!'),[request_frame('PAPER')])
        self.assertEqual(f.feed(b'@KSV1:UNKNOWN!@'+b'A'*200+b'!'),[])
        self.assertEqual(f.feed(request_frame('REPORT')+request_frame('STOP')),
                         [request_frame('REPORT'),request_frame('STOP')])

    def test_execution_needs_explicit_ready_before_worker(self):
        with self.assertRaisesRegex(ValueError,'explicit_ready'):
            VoiceSession(lambda *_:None,lambda:None,lambda:None,execute=True)

    def test_only_one_execution_attempt_even_after_success(self):
        calls=[]
        def run(*args):calls.append(args);return passed()
        s=VoiceSession(run,lambda:None,lambda:None,execute=True,ready=True)
        s.handle(request_frame('PAPER'));s.worker.join(2)
        self.assertEqual(s.replies.get_nowait(),'PAPER_PASS')
        s.handle(request_frame('AUTO'))
        self.assertEqual(s.replies.get_nowait(),'REJECTED')
        self.assertEqual(calls,[('PAPER',True,True)])

    def test_busy_request_not_queued_and_stop_reaches_motion_guard(self):
        started=threading.Event();release=threading.Event();stops=[];calls=[]
        def run(*args):
            calls.append(args);started.set();release.wait(3)
            return dict(mode='known_physical',status='needs_review')
        s=VoiceSession(run,lambda:passed(),lambda:stops.append(True),True,True)
        try:
            s.handle(request_frame('AUTO'));self.assertTrue(started.wait(2))
            s.handle(request_frame('PAPER'));s.handle(request_frame('REPORT'))
            self.assertEqual([s.replies.get_nowait(),s.replies.get_nowait()],['BUSY','BUSY'])
            s.handle(request_frame('STOP'))
            self.assertEqual(s.replies.get_nowait(),'STOPPED');self.assertEqual(stops,[True])
        finally:release.set();s.worker.join(2)
        self.assertEqual(s.replies.get_nowait(),'REVIEW')
        s.handle(request_frame('METAL'));self.assertEqual(s.replies.get_nowait(),'REJECTED')
        self.assertEqual(len(calls),1)

    def test_preview_cannot_announce_real_success(self):
        s=VoiceSession(lambda *_:passed(),lambda:None,lambda:None)
        s.handle(request_frame('PAPER'));s.worker.join(2)
        self.assertEqual(s.replies.get_nowait(),'NO_RESULT')

    def test_valid_preview_is_announced_as_preview(self):
        s=VoiceSession(lambda *_:dict(mode='preview',status='preview_only'),lambda:None,lambda:None)
        s.handle(request_frame('AUTO'));s.worker.join(2)
        self.assertEqual(s.replies.get_nowait(),'PREVIEW')

    def test_unavailable_report_has_explicit_no_result_reply(self):
        def broken():raise OSError('record unavailable')
        s=VoiceSession(lambda *_:self.fail('unexpected sort'),broken,lambda:None)
        s.handle(request_frame('REPORT'))
        self.assertEqual(s.replies.get_nowait(),'NO_RESULT')

    def test_report_query_does_not_start_sort_and_review_stays_review(self):
        s=VoiceSession(lambda *_:self.fail('query started sort'),
            lambda:dict(mode='known_physical',status='needs_review',operator_review={'resolved':True}),
            lambda:None)
        s.handle(request_frame('REPORT'))
        self.assertEqual(s.replies.get_nowait(),'REVIEW');self.assertIsNone(s.worker)

    def test_material_mismatch_rejected_before_existing_planner(self):
        c=VoiceController.__new__(VoiceController);c.voice_request='PAPER'
        with patch('src.km1_voice_station.RelocatedController.plan') as planner:
            with self.assertRaisesRegex(ValueError,'requested_material_not_observed'):
                c.plan({},({'label':'metal'},None))
            planner.assert_not_called()
            c.plan({},({'label':'paper'},None));planner.assert_called_once()


if __name__=='__main__':unittest.main()
