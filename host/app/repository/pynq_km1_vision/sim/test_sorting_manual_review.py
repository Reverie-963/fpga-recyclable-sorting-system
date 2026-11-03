import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.board_control_ownership import BoardOwnership
from src.sorting_manual_review import acknowledge,review_status
from src.monitor_operation import Operations


class ManualReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.folder=self.root/'journal';self.folder.mkdir();self.lease=self.root/'device.lease';self.stop=self.root/'STOP'
        self.owner=BoardOwnership(self.root/'owner.db');self.addCleanup(self.owner.close)
        self.owner.finish(self.owner.claim('station-console','failed-request'),'needs_review')
        self.original=json.dumps(dict(task_id='child',status='needs_review',reason='verification_failed')).encode()
        (self.folder/'child.json').write_bytes(self.original)
        (self.folder/'pending.json').write_bytes(self.original)
        self.batch=json.dumps(dict(batch_id='batch',status='needs_review',items=['child'])).encode();(self.folder/'batch-pending.json').write_bytes(self.batch)

    def review(self,**values):
        return acknowledge(self.folder,self.lease,self.owner,**dict(epoch=self.owner.epoch,request_id='failed-request',confirmed=True,**values))

    def test_confirmed_review_archives_exact_pending_releases_only_review_and_never_promotes_report(self):
        self.stop.write_text('independent operator stop')
        result=self.review();self.assertEqual(result['status'],'operator_reviewed')
        archive=self.folder/'reviews'/result['review_id']
        self.assertEqual((archive/'pending.json').read_bytes(),self.original);self.assertEqual((archive/'batch-pending.json').read_bytes(),self.batch)
        self.assertEqual((self.folder/'child.json').read_bytes(),self.original);self.assertEqual(json.loads(self.original)['status'],'needs_review')
        self.assertFalse((self.folder/'pending.json').exists());self.assertFalse((self.folder/'controller.lock').exists())
        self.assertEqual(self.owner.status()['status'],'idle');self.assertEqual(self.stop.read_text(),'independent operator stop')
        record=json.loads((archive/'review.json').read_text());self.assertFalse(record['automatic_success']);self.assertFalse(record['physical_motion'])
        self.assertFalse(result['automatic_rearm']);self.assertFalse(result['physical_motion'])
        with self.assertRaisesRegex(ValueError,'already_used'):self.owner.claim('voice','failed-request')

    def test_missing_confirmation_or_stale_identity_preserves_every_protection(self):
        for values in (dict(confirmed=False),dict(epoch='old-service'),dict(request_id='other-task')):
            params=dict(epoch=self.owner.epoch,request_id='failed-request',confirmed=True);params.update(values)
            with self.subTest(values=values),self.assertRaises(ValueError):acknowledge(self.folder,self.lease,self.owner,**params)
            self.assertEqual((self.folder/'pending.json').read_bytes(),self.original);self.assertEqual(self.owner.status()['status'],'blocked')

    def test_active_unknown_or_operator_stop_owner_cannot_be_acknowledged_as_finished_review(self):
        for status,reason in (('active',None),('blocked','outcome_unknown'),('blocked','operator_stop'),('blocked','board_owner_heartbeat_lost')):
            with self.owner.transaction() as state:state['status']=status;state['reason']=reason
            with self.subTest(reason=reason),self.assertRaisesRegex(ValueError,'not_eligible'):self.review()
            self.assertTrue((self.folder/'pending.json').exists())

    def test_any_device_or_controller_lock_is_preserved_including_unknown_stale_lock(self):
        for path in (self.lease,self.folder/'controller.lock'):
            path.write_bytes(b'unknown owner')
            self.assertFalse(review_status(self.folder,self.lease,self.owner)['available'])
            with self.assertRaisesRegex(ValueError,'still_owned'):self.review()
            self.assertEqual(path.read_bytes(),b'unknown owner');path.unlink()

    def test_audit_commit_failure_restores_pending_and_keeps_owner_blocked(self):
        with patch.object(self.owner,'event',side_effect=OSError('audit unavailable')):
            with self.assertRaises(OSError):self.review()
        self.assertEqual((self.folder/'pending.json').read_bytes(),self.original)
        self.assertEqual((self.folder/'batch-pending.json').read_bytes(),self.batch)
        self.assertEqual(self.owner.status()['status'],'blocked');self.assertEqual((self.folder/'child.json').read_bytes(),self.original)

    def test_owner_only_review_is_audited_without_inventing_pending(self):
        (self.folder/'pending.json').unlink();(self.folder/'batch-pending.json').unlink()
        self.assertTrue(review_status(self.folder,self.lease,self.owner)['required'])
        self.assertEqual(self.review()['archived'],[])

    def test_uncertain_transport_evidence_cannot_hide_behind_generic_needs_review_owner(self):
        for extra in (dict(reason='TimeoutError: Reply lost; inspect physical state before any new motion'),
            dict(steps=[dict(transport_receipt=dict(echo_matched=False))]),dict(outcome_unknown=True)):
            raw=json.dumps(dict(task_id='child',status='needs_review',**extra)).encode();(self.folder/'pending.json').write_bytes(raw)
            self.assertFalse(review_status(self.folder,self.lease,self.owner)['available'])
            with self.assertRaisesRegex(ValueError,'requires_recovery'):self.review()
            self.assertEqual((self.folder/'pending.json').read_bytes(),raw);self.assertEqual(self.owner.status()['status'],'blocked')

    def backend(self):
        outer=self
        class Backend:
            def check(self):return ['needs_review'] if outer.owner.status()['status']=='blocked' else []
            def review_status(self):return review_status(outer.folder,outer.lease,outer.owner)
            def review(self,**values):return acknowledge(outer.folder,outer.lease,outer.owner,**values)
            def cycle(self,*args):raise AssertionError('no motion')
            def home(self,*args):raise AssertionError('no home')
            def reset(self,*args):raise AssertionError('no reset')
            def stop(self):raise AssertionError('no automatic STOP')
        return Backend()

    def operations(self):
        op=Operations(self.backend());self.addCleanup(op.close)
        op.operation_id='current-web-preview';op.state='finished';op.result=dict(status='needs_review',task_id='batch')
        return op

    def command(self,op):return dict(action='review',token=op.token,owner='cloud-review-page',request_id='review-request-001',
        expected_operation_id='current-web-preview',review_epoch=self.owner.epoch,review_request_id='failed-request',ready=True,review_confirmed=True)

    def test_dispatch_is_nonphysical_single_use_and_returns_idle_without_clearing_original_report(self):
        op=self.operations();result=op.command(self.command(op))
        self.assertEqual(result['state'],'idle');self.assertIsNone(result['operation_id']);self.assertFalse(result['busy'])
        self.assertEqual(result['result']['status'],'operator_reviewed');self.assertFalse(result['manual_review']['required'])
        self.assertIsNone(op.worker);self.assertEqual((self.folder/'child.json').read_bytes(),self.original)
        with self.assertRaisesRegex(ValueError,'duplicate'):op.command(self.command(op))

    def test_dispatch_requires_its_own_fresh_confirmation_and_operation_context(self):
        op=self.operations()
        for values in (dict(review_confirmed=False),dict(ready=False),dict(expected_operation_id='another-preview')):
            command=self.command(op);command.update(values)
            with self.subTest(values=values),self.assertRaises(ValueError):op.command(command)
        self.assertEqual(self.owner.status()['status'],'blocked');self.assertTrue((self.folder/'pending.json').exists())

    def test_dispatch_audit_failure_precedes_unlock(self):
        op=self.operations()
        with patch.object(op,'_record',side_effect=OSError('operation audit unavailable')):
            with self.assertRaises(OSError):op.command(self.command(op))
        self.assertTrue((self.folder/'pending.json').exists());self.assertEqual(self.owner.status()['status'],'blocked')

if __name__=='__main__':unittest.main()
