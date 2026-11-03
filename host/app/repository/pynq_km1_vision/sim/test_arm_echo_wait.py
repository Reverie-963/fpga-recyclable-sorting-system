from pathlib import Path
import tempfile
import unittest

from src.static_supervised_trial import execute


class Clock:
    def __init__(self): self.now = 0
    def monotonic(self): return self.now
    def sleep(self, seconds): self.now += seconds


class Port:
    def __init__(self, clock, delay=.01, echo=True):
        self.clock, self.delay, self.echo = clock, delay, echo
        self.times, self.payload, self.closed = [], b'', False
    def open(self): pass
    def close(self): self.closed = True
    def reset_input_buffer(self): pass
    def write(self, payload):
        self.times.append(self.clock.now)
        self.payload = payload
        return len(payload)
    def read(self, n):
        self.clock.now += .01
        if self.echo and self.clock.now-self.times[-1] >= self.delay:
            out, self.payload = self.payload, b''
            return out
        return b''


class EchoWaitTests(unittest.TestCase):
    def run_trial(self, policy=None, delay=.01, echo=True, settle=0):
        clock = Clock()
        port = Port(clock, delay, echo)
        plan = dict(task_id='timing-test')
        if policy is not None: plan['wait_policy'] = policy
        stage = dict(name='cycle', steps=[dict(command='{#000P1500T1500!}',
            duration_ms=1500, extra_settle_s=settle)]*2)
        with tempfile.TemporaryDirectory() as folder:
            report = execute(Path(folder), plan, stage, 'offline-test', port, clock)
        return report, port

    def test_early_echo_never_shortens_motion_or_settling(self):
        report, port = self.run_trial('motion_and_echo', settle=.2)
        self.assertEqual(report['status'], 'sent_pending_observation')
        self.assertGreaterEqual(port.times[1]-port.times[0], 1.7)
        self.assertLess(port.times[1]-port.times[0], 1.73)

    def test_default_wait_is_backwards_compatible(self):
        _, port = self.run_trial()
        self.assertGreaterEqual(port.times[1]-port.times[0], 2.)

    def test_echo_after_motion_end_is_still_required(self):
        report, port = self.run_trial('motion_and_echo', delay=1.8)
        self.assertEqual(report['status'], 'sent_pending_observation')
        self.assertGreaterEqual(port.times[1]-port.times[0], 1.8)

    def test_missing_echo_stops_without_next_motion(self):
        report, port = self.run_trial('motion_and_echo', echo=False)
        self.assertEqual(report['status'], 'stopped_needs_review')
        self.assertEqual(len(port.times), 1)
        self.assertTrue(port.closed)

    def test_unknown_wait_policy_rejected_before_open(self):
        with self.assertRaises(ValueError): self.run_trial('fast_unknown')


if __name__ == '__main__': unittest.main()
