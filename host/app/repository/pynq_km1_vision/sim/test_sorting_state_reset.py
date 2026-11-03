"""Reset is an audited, idle-only software operation; no serial imports."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from src.sorting_state_reset import reset_state,process_alive


class StateResetTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.ledger=self.root/'ledger';self.ledger.mkdir()
        self.lease=self.root/'device.lock';self.stop=self.root/'STOP'

    def reset(self,confirmed=False,alive=lambda pid:False):
        return reset_state(self.ledger,self.lease,[self.stop],confirmed,'synthetic operator confirmation',alive)

    def test_idle_ui_reset_needs_no_physical_confirmation_or_motion(self):
        result=self.reset()
        self.assertEqual(result['cleared'],[]);self.assertFalse(result['physical_motion'])
        self.assertFalse(result['automatic_rearm'])
        self.assertFalse((self.ledger/'controller.lock').exists())

    def test_stop_and_pending_require_confirmation_and_preserve_original_report(self):
        report=self.ledger/'task.json';report.write_text('{"status":"needs_review"}')
        original=hashlib.sha256(report.read_bytes()).hexdigest()
        (self.ledger/'pending.json').write_text('{"task_id":"task"}')
        (self.ledger/'batch-pending.json').write_text('{"batch_id":"batch"}')
        self.stop.write_text('operator stopped')
        with self.assertRaisesRegex(ValueError,'confirmation_required'):self.reset()
        self.assertTrue(self.stop.exists())
        result=self.reset(True)
        self.assertEqual(set(result['cleared']),{'motion_pending','batch_pending','stop_protection'})
        self.assertEqual(hashlib.sha256(report.read_bytes()).hexdigest(),original)
        archive=self.ledger/'resets'/result['reset_id']
        record=json.loads((archive/'completed.json').read_text())
        self.assertEqual(record['outcome'],'state_reset_not_automatic_success')
        for item in record['files']:
            self.assertEqual(hashlib.sha256((archive/item['snapshot']).read_bytes()).hexdigest(),item['sha256'])

    def test_live_owner_and_unknown_legacy_lease_are_never_forced_clear(self):
        for content in ('{"pid":123,"token":"owned"}','legacy-uuid-without-owner'):
            self.lease.write_text(content)
            with self.assertRaisesRegex(ValueError,'reset_device_still_owned|reset_lock_owner_unknown'):
                self.reset(True,alive=lambda pid:True)
            self.assertEqual(self.lease.read_text(),content)

    def test_proven_dead_owner_can_be_archived_after_confirmation(self):
        self.lease.write_text('{"pid":456,"token":"dead"}')
        (self.ledger/'controller.lock').write_text('456')
        result=self.reset(True)
        self.assertEqual(set(result['cleared']),{'device_lease','controller_lock'})
        self.assertFalse(self.lease.exists());self.assertFalse((self.ledger/'controller.lock').exists())

    def test_process_probe_recognizes_this_process_without_signalling_it(self):
        import os
        self.assertTrue(process_alive(os.getpid()))
