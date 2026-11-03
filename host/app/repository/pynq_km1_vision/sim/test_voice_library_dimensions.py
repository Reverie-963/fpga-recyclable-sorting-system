import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.board_voice_host import SingleAttemptVoiceClient
from src.km1_voice import result_reply
from src.km1_voice_dashboard import VoiceRecords,VoiceCommandNotSubmitted
from test_voice_height_registration import RegisteredOperations,measured_plan


class VoiceLibraryTests(unittest.TestCase):
    def setUp(self):
        self.ops=RegisteredOperations();self.addCleanup(self.ops.close)
        ops=self.ops;self.submissions=[];submissions=self.submissions;plan=measured_plan()
        class Client(SingleAttemptVoiceClient):
            def _status(self):self.token=ops.token;return ops.status()
            def _request(self,route,body=None):
                if body:submissions.append(copy.deepcopy(body));return ops.command(body)
                return ops.status()
            @staticmethod
            def dimensions_prepare(data):return dict(data,height_plan=copy.deepcopy(plan),library_revision=4,frame_id='frame')
        self.client=Client('http://127.0.0.1:18810',None)

    def test_explicit_voice_registers_library_plan_then_submits_once_consumed(self):
        self.client._command('start',material='AUTO',ready=True,request_id='voice-library-once')
        self.ops.worker.join(2)
        self.assertEqual([s['action'] for s in self.submissions],['register_heights','start'])
        self.assertEqual([i['object_height_mm'] for i in self.ops.calls[0][2]['items']],[40,80])
        self.assertIsNone(self.ops.height_registration())

    def test_preparation_failure_or_cancel_does_not_submit_start_or_stop(self):
        for fail in (ValueError('missing_or_ambiguous'),TimeoutError('prepare_unreachable')):
            with patch.object(self.client,'dimensions_prepare',side_effect=fail),patch.object(self.client,'stop') as stop:
                self.assertEqual(self.client.run('AUTO',True,True),dict(voice_transport_reply='REJECTED'));stop.assert_not_called()
        self.client.cancelled.set()
        with self.assertRaises(VoiceCommandNotSubmitted):self.client._command('start',material='AUTO',ready=True,request_id='cancelled')
        self.assertEqual(self.submissions,[])

    def test_model_change_after_preparation_rejects_before_registration(self):
        prepare=self.client.dimensions_prepare
        def changed(data):
            result=prepare(data);self.ops.binding['revision']+=1;return result
        with patch.object(self.client,'dimensions_prepare',side_effect=changed):
            with self.assertRaises(VoiceCommandNotSubmitted):self.client._command('start',material='AUTO',ready=True,request_id='changed-model')
        self.assertEqual(self.submissions,[])

    def test_lost_registration_reply_never_sends_start_or_stop(self):
        request=self.client._request
        def lost(route,body=None):
            if body and body['action']=='register_heights':request(route,body);raise TimeoutError('registration_reply_lost')
            return request(route,body)
        with patch.object(self.client,'_request',side_effect=lost),patch.object(self.client,'stop') as stop:
            self.assertEqual(self.client.run('AUTO',True,True),dict(voice_transport_reply='REJECTED'));stop.assert_not_called()
        self.assertEqual([s['action'] for s in self.submissions],['register_heights']);self.assertEqual(self.ops.calls,[])


class CurrentBatchVoiceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.folder=Path(self.temp.name);self.reader=VoiceRecords(self.folder)
        self.records=[dict(task_id='child-'+str(i),batch_id='batch',mode='known_physical',status='motion_completed_awaiting_batch_verification',
            choice=dict(label=label,track_id='track-'+str(i),slot_id='slot-'+str(i),bin_id='bin-'+label)) for i,label in enumerate(('paper','plastic'))]
        self.items=[]
        for record in self.records:
            raw=json.dumps(record).encode();(self.folder/(record['task_id']+'.json')).write_bytes(raw)
            self.items.append(dict(task_id=record['task_id'],report_sha256=hashlib.sha256(raw).hexdigest(),physical_motion_attempted=True))
        self.final=dict(status='batch_visual_evidence_pass',mode='bin_occupied',scope='visited_bin_occupancy_only',attempted_count=2,
            frames=[dict(frame_id='final')],bins=[dict(bin_id='bin-'+l,status='bin_occupied_pass',frames=[dict(frame_id='final',occupied=True)]) for l in ('paper','plastic')],
            physical_success_verified=False,class_verified=False,new_item_verified=False,source_removal_verified=False)
        self.batch=dict(batch_id='batch',status='completed',items=self.items,motion_completed_count=2,final_verification=self.final)
        self.result=dict(self.batch,status='completed_visual_evidence',task_id='batch',mode='known_physical',batch_status='completed')
        self.save()

    def save(self):
        for name,value in (('batch.batch.json',self.batch),('batch.json',self.result),('batch.batch-final-verification.json',self.final)):
            (self.folder/name).write_text(json.dumps(value))

    def test_current_single_frame_occupancy_is_readable_without_promoting_success(self):
        record=self.reader.selected(self.result)
        self.assertEqual(record['voice_source_status'],'completed_visual_evidence')
        self.assertEqual(record['voice_evidence_basis'],'visited_bin_occupancy_only');self.assertEqual(result_reply(record),'PAPER_PASS')
        self.assertTrue(record['voice_verification_completed'])
        for key in ('physical_success_verified','class_verified','new_item_verified','source_removal_verified'):
            self.assertIs(record[key],False)
        self.assertEqual(json.loads((self.folder/'batch.json').read_text())['status'],'completed_visual_evidence')

    def test_earlier_child_tampering_and_wrong_final_rejected(self):
        path=self.folder/'child-0.json';raw=path.read_bytes();path.write_bytes(raw+b' ')
        with self.assertRaisesRegex(ValueError,'hash_mismatch'):self.reader.selected(self.result)
        path.write_bytes(raw);self.final['bins'][0]['frames'][0]['occupied']=False;self.save()
        with self.assertRaisesRegex(ValueError,'occupancy_evidence'):self.reader.selected(self.result)

    def test_semantic_multi_material_does_not_announce_only_last_item(self):
        self.final.clear();self.final.update(status='batch_visual_evidence_pass',attempted_count=2,
            frames=[dict(frame_id=str(n),placements=[dict(r['choice']) for r in self.records]) for n in range(3)])
        self.save();record=self.reader.selected(self.result)
        self.assertEqual(record['voice_materials'],['paper','plastic']);self.assertEqual(result_reply(record),'PAPER_PASS')

if __name__=='__main__':unittest.main()
