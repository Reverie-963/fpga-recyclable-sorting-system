"""Page registration through the real operation dispatcher; no hardware I/O."""
import copy
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.monitor_prompted_operation import PromptedOperations
from src.board_voice_host import SingleAttemptVoiceClient
from src.km1_voice_dashboard import VoiceCommandNotSubmitted


def measured_plan():
    return dict(kind='instance-heights-v1', model_sha256='a'*64, configuration_sha256='b'*64,
        source_stream_id='stream', image_size=[40,30], items=[
            dict(item_id=1,label='metal',official_label='Drink can',mask_rle=[0,20,1180],object_height_mm=40),
            dict(item_id=2,label='metal',official_label='Drink can',mask_rle=[400,20,780],object_height_mm=80)])


class RegisteredOperations(PromptedOperations):
    def __init__(self):
        self.binding=dict(profile_sha256='c'*64,revision=7,model_sha256='a'*64,configuration_sha256='b'*64)
        self.calls=[]
        super().__init__(SimpleNamespace(check=lambda:[],object_height_mm=None,height_plan=None,stop=lambda:None))

    def status(self):
        return dict(super().status(),execution_model=dict(self.binding))

    def _run(self,action,material):
        with self.lock:
            self.calls.append((action,material,copy.deepcopy(self.backend.height_plan),self.backend.object_height_mm))
            self.state='finished';self.result={'status':'empty_scene'}


class VoiceHeightRegistrationTests(unittest.TestCase):
    def setUp(self):
        self.policy=patch('src.monitor_prompted_operation.calibrated_settings',side_effect=ValueError('fixture'))
        self.policy.start();self.addCleanup(self.policy.stop)
        self.ops=RegisteredOperations();self.addCleanup(self.ops.close)
        self.plan=measured_plan()

    def save(self,**values):
        return self.ops.command(dict(action='register_heights',owner='web-page',token=self.ops.token,
            request_id='register-0001',height_plan=self.plan,ready=True,**values))

    def client(self):
        ops=self.ops
        class Client(SingleAttemptVoiceClient):
            def _status(self):
                self.token=ops.token
                return ops.status()
            def _request(self,route,body=None):
                return ops.command(body) if body else ops.status()
        return Client('http://127.0.0.1:18810',None)

    def test_page_to_voice_freezes_each_height_and_consumes_once(self):
        self.plan['items'][0]['grasp_width_mm']=10
        self.plan['items'][0]['measurement_profile']=dict(id='d'*32,revision=2,name='measured specimen',pose='flat')
        status=self.save();self.assertEqual(self.ops.calls,[])
        self.assertFalse(status['busy']);self.assertIsNone(status['operation_id'])
        self.plan['items'][0]['object_height_mm']=99
        client=self.client();client._command('start',material='AUTO',ready=True,request_id='voice-start-0001')
        self.ops.worker.join(2)
        call=self.ops.calls[0]
        self.assertEqual([i['object_height_mm'] for i in call[2]['items']],[40,80]);self.assertIsNone(call[3])
        self.assertEqual(call[2]['items'][0]['grasp_width_mm'],10)
        self.assertEqual(call[2]['items'][0]['measurement_profile']['revision'],2)
        self.assertIsNone(self.ops.status()['voice_height_registration'])
        with self.assertRaises(VoiceCommandNotSubmitted):client._command('start',material='AUTO')
        self.assertEqual(len(self.ops.calls),1)

    def test_missing_registration_never_uses_old_height_or_stops(self):
        self.ops.backend.object_height_mm=56
        client=self.client()
        with patch.object(client,'fallback_stop') as stop:
            self.assertEqual(client.run('METAL',True,True),{'voice_transport_reply':'REJECTED'})
            stop.assert_not_called()
        self.assertEqual(self.ops.calls,[])

    def test_page_confirmation_invalid_height_busy_and_wrong_model_reject(self):
        base=dict(action='register_heights',owner='web-page',token=self.ops.token,request_id='register-0001',height_plan=self.plan,ready=True)
        bad=copy.deepcopy(self.plan);bad['items'][1]['object_height_mm']=None
        changed=copy.deepcopy(self.plan);changed['configuration_sha256']='d'*64
        for change in ({'ready':False},{'height_plan':bad},{'height_plan':changed},{'token':'old'}):
            with self.subTest(change=change),self.assertRaises(ValueError):self.ops.command(dict(base,**change))
        self.ops.state='executing'
        with self.assertRaises(ValueError):self.ops.command(base)
        self.ops.state='idle';self.assertIsNone(self.ops.height_registration());self.assertEqual(self.ops.calls,[])

    def test_expiry_and_model_revision_never_fall_back(self):
        self.save();self.ops._height_registration['deadline']=0
        with self.assertRaises(VoiceCommandNotSubmitted):self.client()._command('start',material='METAL')
        self.ops.seen.clear();self.save();self.ops.binding['revision']+=1
        with self.assertRaises(VoiceCommandNotSubmitted):self.client()._command('start',material='METAL')
        self.assertEqual(self.ops.calls,[])

    def test_revoke_or_replace_between_voice_read_and_submit_rejects(self):
        old=self.save()['voice_height_registration']['id']
        self.ops.command(dict(action='clear_heights',token=self.ops.token,owner='web-page',height_registration_id=old))
        self.ops.seen.clear();new=self.save()['voice_height_registration']['id']
        self.assertNotEqual(old,new)
        for action in ('start','clear_heights'):
            with self.assertRaises(ValueError):self.ops.command(dict(action=action,token=self.ops.token,
                owner='voice-client',request_id='voice-request',ready=True,height_registration_id=old))
        self.assertEqual(self.ops.height_registration()['id'],new);self.assertEqual(self.ops.calls,[])

    def test_preview_keeps_registration_manual_start_consumes_it(self):
        self.save()
        self.ops.command(dict(action='preview',token=self.ops.token,owner='web-page',request_id='preview-0001',height_plan=self.plan))
        self.ops.worker.join(2);self.assertIsNotNone(self.ops.height_registration())
        self.ops.command(dict(action='start',token=self.ops.token,owner='web-page',request_id='manual-0001',ready=True,height_plan=self.plan))
        self.ops.worker.join(2);self.assertIsNone(self.ops.height_registration())

    def test_pre_submission_failure_retains_registration_and_prior_backend(self):
        saved=self.save()['voice_height_registration']['id']
        with self.assertRaises(ValueError):self.ops.command(dict(action='start',token=self.ops.token,
            owner='voice-client',request_id='voice-0001',ready=False,height_registration_id=saved))
        self.assertIsNone(self.ops.backend.height_plan)
        self.assertEqual(self.ops.height_registration()['id'],saved)

    def test_server_rechecks_binding_and_rejects_legacy_voice_scalar(self):
        saved=self.save()['voice_height_registration']['id'];self.ops.binding['revision']+=1
        with self.assertRaisesRegex(ValueError,'binding_changed'):
            self.ops.command(dict(action='start',token=self.ops.token,owner='voice-client',
                request_id='voice-0001',ready=True,height_registration_id=saved))
        with self.assertRaisesRegex(ValueError,'requires_page'):
            self.ops.command(dict(action='start',token=self.ops.token,owner='voice-old',
                request_id='voice-0002',ready=True,object_height_mm=56))
        self.assertEqual(self.ops.calls,[])

    def test_error_after_acceptance_cannot_restore_registration_or_change_frozen_plan(self):
        saved=self.save()['voice_height_registration']['id']
        status=self.ops.status;calls=[]
        def fail_after_acceptance():
            calls.append(1)
            if len(calls)==2:raise ValueError('reply_lost_after_acceptance')
            return status()
        with patch.object(self.ops,'status',side_effect=fail_after_acceptance):
            with self.assertRaisesRegex(ValueError,'reply_lost'):
                self.ops.command(dict(action='start',token=self.ops.token,owner='voice-client',
                    request_id='voice-accepted',ready=True,height_registration_id=saved))
        self.ops.worker.join(2)
        self.assertIsNone(self.ops.height_registration())
        self.assertEqual([i['object_height_mm'] for i in self.ops.calls[0][2]['items']],[40,80])

    def test_stop_clears_registration_without_changing_height_or_starting(self):
        self.save();self.ops.command(dict(action='stop',token=self.ops.token,owner='web-page'))
        self.assertIsNone(self.ops.height_registration());self.assertEqual(self.ops.calls,[])

    def test_audit_write_failure_cannot_leave_a_revoked_registration_usable(self):
        self.save()
        with patch.object(self.ops,'_record',side_effect=OSError('disk_unavailable')):
            with self.assertRaises(OSError):self.ops.invalidate_height_registration('start_consumed')
        self.assertIsNone(self.ops.height_registration())

    def test_duplicate_old_start_does_not_consume_new_registration(self):
        command=dict(action='start',token=self.ops.token,owner='web-page',request_id='manual-old',ready=True,height_plan=self.plan)
        self.ops.command(command);self.ops.worker.join(2)
        saved=self.save()['voice_height_registration']['id'];previous=copy.deepcopy(self.ops.backend.height_plan)
        changed=copy.deepcopy(command);changed['height_plan']['items'][0]['object_height_mm']=90
        with self.assertRaisesRegex(ValueError,'duplicate_request'):self.ops.command(changed)
        self.assertEqual(self.ops.height_registration()['id'],saved)
        self.assertEqual(self.ops.backend.height_plan,previous)


if __name__=='__main__':unittest.main()
