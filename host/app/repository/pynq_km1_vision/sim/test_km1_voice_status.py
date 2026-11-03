import hashlib
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from src.km1_voice_status import ConsoleStatus,ConsoleMonitor,SpeechQueue
from src.km1_voice_dashboard import VoiceRecords
from src.km1_voice import result_reply


def state(operation='operation-one',status='idle',result=None,**values):
    return dict(operation_id=operation,state=status,busy=status in ('executing','checking_preview'),
                result=result,blockers=[],**values)


class ConsoleStateTests(unittest.TestCase):
    def setUp(self):
        self.records=SimpleNamespace(selected=Mock(return_value=dict(mode='known_physical',
            status='completed_visual_evidence',choice=dict(label='paper'))))
        self.client=SimpleNamespace(endpoint='http://127.0.0.1:8810',token='process-token',records=self.records)
        self.mapper=ConsoleStatus()
    def baseline(self):self.assertIsNone(self.mapper.observe(self.client,state()))
    def test_startup_does_not_replay_old_success_or_old_review(self):
        for status in ('completed_visual_evidence','needs_review'):
            self.assertIsNone(ConsoleStatus().observe(self.client,state(status='finished',result=dict(status=status))))
    def test_processing_is_announced_once_then_exact_result(self):
        self.baseline()
        working=state(status='executing')
        self.assertEqual(self.mapper.observe(self.client,working),'BUSY')
        self.assertIsNone(self.mapper.observe(self.client,working))
        final=state(status='finished',result=dict(status='completed_visual_evidence',task_id='task'))
        self.assertEqual(self.mapper.observe(self.client,final),'PAPER_PASS')
        self.assertIsNone(self.mapper.observe(self.client,final))
    def test_preview_is_never_physical_success(self):
        self.baseline()
        self.assertEqual(self.mapper.observe(self.client,state(status='checking_preview')),'BUSY')
        self.assertEqual(self.mapper.observe(self.client,state(status='finished',result=dict(status='preview_only',preview=True))),'PREVIEW')
        self.records.selected.assert_not_called()
    def test_rejection_empty_review_and_stop_are_distinct(self):
        for result,expected in ((dict(status='rejected',preview=True),'REJECTED'),
                (dict(status='empty_scene',preview=True),'EMPTY'),(dict(status='needs_review'),'REVIEW')):
            self.mapper=ConsoleStatus();self.baseline()
            self.assertEqual(self.mapper.observe(self.client,state(status='finished',result=result)),expected)
        self.assertEqual(self.mapper.observe(self.client,state(status='stopped',reason='operator_stop')),'STOPPED')
    def test_cancelling_wait_is_not_claimed_as_global_stop(self):
        self.baseline()
        self.assertIsNone(self.mapper.observe(self.client,state(status='stopped',reason='standby_cancelled')))
    def test_new_controller_does_not_replay_its_historical_result(self):
        self.baseline();self.client.token='new-process'
        self.assertIsNone(self.mapper.observe(self.client,state(status='finished',result=dict(status='completed_visual_evidence'))))
    def test_missing_or_simulated_evidence_cannot_generate_pass(self):
        self.baseline();self.records.selected.return_value=dict(mode='simulation',status='completed_visual_evidence',choice=dict(label='paper'))
        self.assertEqual(self.mapper.observe(self.client,state(status='finished',result=dict(status='completed_visual_evidence'))),'NO_RESULT')
        self.records.selected.side_effect=ValueError('hash mismatch')
        self.assertEqual(self.mapper.observe(self.client,state(operation='operation-two',status='finished',result=dict(status='completed_visual_evidence'))),'REVIEW')
    def test_connection_loss_is_an_event_not_a_repeated_poll(self):
        self.assertEqual(self.mapper.lost(),'OFFLINE');self.assertIsNone(self.mapper.lost())
        self.baseline();self.assertEqual(self.mapper.lost(),'OFFLINE')
    def test_passive_monitor_never_starts_motion_or_renews_a_page(self):
        self.client._status=Mock(return_value=state());self.client._command=Mock(side_effect=AssertionError('mutation'))
        control=SimpleNamespace(lock=threading.Lock(),client=self.client,select=Mock())
        speech=SpeechQueue();monitor=ConsoleMonitor(control,speech,lambda *a,**k:None)
        monitor.sample();self.client._status.return_value=state(status='executing');monitor.sample()
        self.assertEqual(speech.next()['token'],'BUSY')
        self.client._command.assert_not_called();control.select.assert_not_called()


class SpeechQueueTests(unittest.TestCase):
    def setUp(self):self.now=10;self.queue=SpeechQueue(clock=lambda:self.now)
    def test_voice_reply_and_console_final_are_merged(self):
        self.assertTrue(self.queue.offer('REVIEW','same-task',source='console'))
        self.assertFalse(self.queue.offer('REVIEW','same-task',source='voice_reply'))
        self.assertEqual(self.queue.next()['token'],'REVIEW');self.assertIsNone(self.queue.next())
    def test_final_drops_unsent_progress(self):
        self.queue.offer('BUSY','task');self.queue.offer('EMPTY','task')
        self.assertEqual(self.queue.next()['token'],'EMPTY');self.assertIsNone(self.queue.next())
    def test_stop_can_interrupt_progress_without_repeating(self):
        self.queue.offer('BUSY','task');self.queue.next();self.now+=.1
        self.queue.offer('STOPPED','task');self.assertEqual(self.queue.next()['token'],'STOPPED')
        self.assertFalse(self.queue.offer('STOPPED','task'))
    def test_new_explicit_report_can_repeat_after_cooldown(self):
        self.queue.offer('PAPER_PASS','task');self.queue.next();self.now+=3
        self.assertTrue(self.queue.offer('PAPER_PASS','task',explicit=True))
        self.assertEqual(self.queue.next()['token'],'PAPER_PASS')
    def test_stale_unknown_and_excess_events_are_bounded(self):
        self.assertFalse(self.queue.offer('arbitrary serial command','task'))
        for n in range(100):self.queue.offer('EMPTY',str(n))
        self.assertEqual(len(self.queue.pending),16)
        self.now+=9;self.assertIsNone(self.queue.next())
    def test_second_outage_after_recovery_is_not_permanently_muted(self):
        self.queue.offer('OFFLINE');self.queue.next();self.now+=20
        self.assertTrue(self.queue.offer('OFFLINE'));self.assertEqual(self.queue.next()['token'],'OFFLINE')


class DeferredEvidenceTests(unittest.TestCase):
    def test_final_batch_evidence_can_be_spoken_without_rewriting_item_status(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);records=VoiceRecords(folder)
            item=dict(task_id='item',batch_id='batch',mode='known_physical',
                status='motion_completed_awaiting_batch_verification',choice=dict(label='paper',track_id='track',slot_id='slot'))
            raw=json.dumps(item).encode();(folder/'item.json').write_bytes(raw)
            final=dict(status='batch_visual_evidence_pass',frames=[dict(frame_id=i,placements=[dict(
                track_id='track',slot_id='slot',label='paper')]) for i in (1,2,3)])
            batch=dict(batch_id='batch',status='completed',final_verification=final,items=[dict(
                task_id='item',physical_motion_attempted=True,report_sha256=hashlib.sha256(raw).hexdigest())])
            (folder/'batch.batch.json').write_text(json.dumps(batch))
            proof=folder/'batch.batch-final-verification.json';proof.write_text(json.dumps(final))
            report=records.selected(dict(batch_id='batch',status='completed'))
            self.assertEqual(result_reply(report),'PAPER_PASS')
            self.assertEqual(report['voice_source_status'],item['status'])
            self.assertEqual((folder/'item.json').read_bytes(),raw)
            final['frames'][1]['placements'][0]['label']='metal';proof.write_text(json.dumps(final))
            with self.assertRaisesRegex(ValueError,'final_evidence_mismatch'):
                records.selected(dict(batch_id='batch',status='completed'))


if __name__=='__main__':unittest.main()
