"""Telemetry must follow guards, preserve errors and freeze elapsed time."""
import unittest
from unittest.mock import patch

from src.operation_progress import Timeline, install_runtime
from src.monitor_operation import Operations
from src.sorting_io import Motion
from src.arm_transport import ArmConnection
from src.sorting_unified import UnifiedController
from src import sorting_batch_verification


class TimelineTests(unittest.TestCase):
    def test_durations_freeze_and_do_not_reclassify_failed_outcomes(self):
        now = [10.0]
        timeline = Timeline('one', 'start', clock=lambda: now[0])
        now[0] = 14
        timeline.mark('checking')
        now[0] = 17
        timeline.finish('finished', {'status': 'rejected'})
        now[0] = 100
        result = timeline.snapshot()
        self.assertEqual(result['elapsed_s'], 7)
        self.assertEqual(result['outcome'], 'rejected')
        self.assertFalse(result['sound_confirmed'])
        timeline.mark('moving')
        self.assertEqual(timeline.snapshot()['phase'], 'ended')

    def run_hooks(self, fail_guard=False, fail_wait=False,batch=False):
        calls = []
        def run(op, *args):
            motion, connection = Motion.__new__(Motion), ArmConnection.__new__(ArmConnection)
            def guard():
                calls.append('guard')
                if fail_guard:
                    raise ValueError('guard_failed')
            for _ in range(2 if batch else 1):
                Motion.open(motion, [('home', b'x', 1000), ('close', b'y', 1000)])
                for data in ([b'x',b'y'] if batch else [b'x']):
                    ArmConnection.send(connection, data, 1000, guard)
                    Motion.wait(motion, 1)
            if batch:sorting_batch_verification.verify_batch_bin_occupied(None,None,None)
            else:UnifiedController.verify_within_window(None)
            op.result = {'status': 'needs_review'}
            op.state = 'finished'
            return 'original-return'
        def send(connection, data, duration, guard):
            guard()
            calls.append('write')
            return 123
        def wait(motion, seconds):
            calls.append('wait')
            if fail_wait:
                raise TimeoutError('echo_missing')
        with patch.multiple(Operations, _record=lambda *a, **k: None,
                            status=lambda op: dict(operation_id=op.operation_id), _run=run), \
             patch.object(Operations, '_progress_installed', False, create=True), \
             patch.object(Motion, 'open', lambda *a: calls.append('open')), \
             patch.object(Motion, 'wait', wait), patch.object(ArmConnection, 'send', send), \
             patch.object(UnifiedController, 'verify_within_window', lambda *a: calls.append('verify')), \
             patch.object(sorting_batch_verification,'verify_batch_bin_occupied',lambda *a:calls.append('batch-verify')):
            install_runtime()
            op = Operations.__new__(Operations)
            op.operation_id, op.active_action, op.state, op.result = 'one', 'start', 'executing', None
            op._record('accepted')
            try:
                value = op._run('start', 'AUTO')
            except (ValueError, TimeoutError) as error:
                value = str(error)
            result = op.status()['progress']
        return calls, value, result

    def test_real_nodes_follow_original_calls_and_keep_return_value(self):
        calls, value, result = self.run_hooks()
        self.assertEqual(calls, ['open', 'guard', 'write', 'wait', 'verify'])
        self.assertEqual(value, 'original-return')
        self.assertEqual([e['phase'] for e in result['events']],
                         ['submitted', 'checking', 'connecting', 'ready', 'moving', 'verifying', 'ended'])
        self.assertEqual(result['completed_commands'], 1)
        self.assertEqual(result['total_commands'], 2)
        self.assertEqual(result['outcome'], 'needs_review')

    def test_failed_guard_never_claims_uart_or_motion(self):
        calls, value, result = self.run_hooks(fail_guard=True)
        self.assertEqual(value, 'guard_failed')
        self.assertNotIn('write', calls)
        self.assertNotIn('moving', [e['phase'] for e in result['events']])
        self.assertTrue(result['terminal'])

    def test_failed_wait_never_counts_step_or_starts_verification(self):
        calls, value, result = self.run_hooks(fail_wait=True)
        self.assertEqual(value, 'echo_missing')
        self.assertEqual(result['completed_commands'], 0)
        self.assertNotIn('verify', calls)

    def test_batch_progress_counts_all_routes_and_verifies_only_at_end(self):
        calls,value,result=self.run_hooks(batch=True)
        self.assertEqual(calls.count('batch-verify'),1)
        self.assertEqual(calls[-1],'batch-verify')
        self.assertEqual(result['completed_commands'],4)
        self.assertEqual(result['total_commands'],4)
        phases=[e['phase'] for e in result['events']]
        self.assertEqual(phases.count('connecting'),1)
        self.assertEqual(phases.count('ready'),1)
        self.assertEqual(phases.count('verifying'),1)
