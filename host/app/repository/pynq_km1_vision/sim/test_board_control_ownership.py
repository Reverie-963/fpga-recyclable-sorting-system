import json
from pathlib import Path
import tempfile
import threading
import unittest

from src.board_control_ownership import BoardOwnership


class OwnershipTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'candidate-only.db'
        self.now = 100.
        self.owner = BoardOwnership(self.path, clock=lambda:self.now)
        self.addCleanup(self.owner.close)

    def test_only_one_host_claims_and_completed_request_is_never_reused(self):
        lease = self.owner.claim('pc-web', 'request-one')
        for client in ('board-web', 'voice', 'cli'):
            with self.assertRaisesRegex(ValueError, 'board_control_active'):
                self.owner.claim(client, 'other-'+client)
        self.owner.finish(lease, 'completed')
        with self.assertRaisesRegex(ValueError, 'board_request_already_used'):
            self.owner.claim('voice', 'request-one')
        self.owner.check(self.owner.claim('voice', 'new-request'))

    def test_concurrent_clients_do_not_both_own_control(self):
        accepted, rejected = [], []
        barrier = threading.Barrier(3)
        def claim(client):
            barrier.wait()
            try: accepted.append(self.owner.claim(client, 'req-'+client))
            except ValueError as exc: rejected.append(str(exc))
        workers = [threading.Thread(target=claim, args=(name,)) for name in ('host-a', 'host-b')]
        for worker in workers: worker.start()
        barrier.wait()
        for worker in workers: worker.join(2)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(rejected, ['board_control_active'])

    def test_owner_token_and_epoch_cannot_be_substituted(self):
        lease = self.owner.claim('pc-web', 'request-one')
        for key in ('owner', 'request_id', 'token', 'epoch'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.owner.check(dict(lease, **{key:'wrong'}), heartbeat=True)
        self.owner.check(lease)

    def test_timeout_latches_and_no_other_client_inherits(self):
        lease = self.owner.claim('pc-web', 'request-one')
        self.now += 12
        with self.assertRaisesRegex(ValueError, 'board_control_blocked'):
            self.owner.check(lease, heartbeat=True)
        self.assertEqual(self.owner.status()['reason'], 'board_owner_heartbeat_lost')
        with self.assertRaisesRegex(ValueError, 'board_control_blocked'):
            self.owner.claim('voice', 'request-two')

    def test_heartbeat_uses_board_clock_and_preserves_owner(self):
        lease = self.owner.claim('pc-web', 'request-one')
        self.now += 10
        self.owner.check(lease, heartbeat=True)
        self.now += 10
        self.owner.check(lease)
        self.assertEqual(self.owner.status()['owner'], 'pc-web')

    def test_restart_latches_active_owner_and_fences_old_service(self):
        lease = self.owner.claim('pc-web', 'request-one')
        restarted = BoardOwnership(self.path, clock=lambda:self.now)
        try:
            self.assertEqual(restarted.status()['reason'], 'board_service_restarted_with_owner')
            with self.assertRaisesRegex(ValueError, 'board_owner_service_epoch_changed'):
                self.owner.check(lease)
            with self.assertRaises(ValueError): restarted.claim('voice', 'request-two')
        finally: restarted.close()

    def test_unknown_result_stop_and_restart_preserve_original_requests(self):
        lease = self.owner.claim('pc-web', 'request-one')
        self.owner.finish(lease, 'outcome_unknown')
        self.owner.stop('test simulated stop only')
        with self.assertRaises(ValueError): self.owner.review('not confirmed')
        self.owner.review('synthetic operator checked stopped scene', confirmed=True)
        with self.assertRaisesRegex(ValueError, 'board_request_already_used'):
            self.owner.claim('voice', 'request-one')
        self.assertEqual(self.owner.status()['status'], 'idle')
        rows = [json.loads(row[0]) for row in self.owner.db.execute('SELECT value FROM events')]
        self.assertEqual(rows[-1]['action'], 'operator_review_not_automatic_success')
        self.assertFalse(any(lease['token'] in json.dumps(row) for row in rows))

    def test_review_cannot_release_live_owner(self):
        lease = self.owner.claim('pc-web', 'request-one')
        with self.assertRaisesRegex(ValueError, 'only_blocked'):
            self.owner.review('synthetic statement', confirmed=True)
        self.owner.check(lease)

    def test_invalid_requests_do_not_claim_or_clear_state(self):
        for value in ('', '../bad', 'x'*129, None, '队长'):
            with self.assertRaises(ValueError): self.owner.claim(value, 'request-one')
        self.assertEqual(self.owner.status()['status'], 'idle')
        lease = self.owner.claim('pc-web', 'request-one')
        with self.assertRaises(ValueError): self.owner.finish(lease, 'retry')
        self.owner.check(lease)


if __name__ == '__main__':
    unittest.main()
