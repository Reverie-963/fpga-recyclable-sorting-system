import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.static_supervised_trial import execute, load_plan, prerequisites, save


class FakeClock:
    def __init__(self): self.now = 0
    def sleep(self, seconds): self.now += seconds
    def monotonic(self): self.now += 1; return self.now


class Port:
    def __init__(self, echo=True, partial=False, fail_open=False):
        self.echo, self.partial, self.fail_open = echo, partial, fail_open
        self.writes, self.closed, self.payload = [], False, b''
    def open(self):
        if self.fail_open: raise IOError('open failed')
    def close(self): self.closed = True
    def reset_input_buffer(self): pass
    def write(self, data):
        self.writes.append(data); self.payload = data
        return len(data) - int(self.partial)
    def read(self, size): return self.payload if self.echo else b''


class SupervisedTrialTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        step = dict(command='{#000P1500T1500!}', duration_ms=1500)
        self.plan = dict(kind='supervised_fixed_path_trial_v1', task_id='test', stages=[
            dict(name='approach', observe='aligned', steps=[step, step]),
            dict(name='grasp', observe='held', steps=[step])])
        self.raw = json.dumps(self.plan).encode()
        self.digest = hashlib.sha256(self.raw).hexdigest()
        self.path = self.root / 'plan.json'
        self.path.write_bytes(self.raw)

    def run_stage(self, port):
        return execute(self.root, self.plan, self.plan['stages'][0], self.digest, port, FakeClock())

    def test_complete_echo_still_requires_human_observation(self):
        port = Port(); result = self.run_stage(port)
        self.assertEqual(result['status'], 'sent_pending_observation')
        self.assertEqual(result['physical_result'], 'pending_operator_observation')
        self.assertTrue(port.closed)
        with self.assertRaises(FileNotFoundError):
            prerequisites(self.root, self.plan, 'grasp', self.digest)

    def test_missing_echo_stops_before_second_command_and_cannot_retry(self):
        port = Port(echo=False); result = self.run_stage(port)
        self.assertEqual(result['status'], 'stopped_needs_review')
        self.assertEqual(len(port.writes), 1)
        with self.assertRaises(ValueError):
            prerequisites(self.root, self.plan, 'approach', self.digest)
        with self.assertRaises(FileExistsError): self.run_stage(Port())

    def test_partial_write_stops(self):
        port = Port(partial=True); self.run_stage(port)
        self.assertEqual(len(port.writes), 1)
        self.assertTrue(port.closed)

    def test_open_failure_consumes_attempt(self):
        self.assertEqual(self.run_stage(Port(fail_open=True))['status'], 'stopped_needs_review')
        with self.assertRaises(ValueError):
            prerequisites(self.root, self.plan, 'approach', self.digest)

    def test_observation_binds_exact_execution(self):
        self.run_stage(Port())
        path = self.root / 'approach.json'
        observation = dict(task_id='test', plan_sha256=self.digest, ok=True,
                           execution_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        save(self.root / 'approach.observation.json', observation)
        self.assertEqual(prerequisites(self.root, self.plan, 'grasp', self.digest)['name'], 'grasp')
        path.write_text(path.read_text() + ' ')
        with self.assertRaises(ValueError): prerequisites(self.root, self.plan, 'grasp', self.digest)

    def test_reviewed_plan_hash_and_duplicate_servo_rejected(self):
        self.assertEqual(load_plan(self.path, self.digest)['task_id'], 'test')
        with self.assertRaises(ValueError): load_plan(self.path, '0'*64)
        self.plan['stages'][0]['steps'][0]['command'] = '{#000P1500T1500!#000P1500T1500!}'
        raw = json.dumps(self.plan).encode(); self.path.write_bytes(raw)
        with self.assertRaises(ValueError): load_plan(self.path, hashlib.sha256(raw).hexdigest())

    def test_invalid_extra_settle_rejected_before_motion(self):
        for settle in (-1, float('nan'), True, 11):
            self.plan['stages'][0]['steps'][0]['extra_settle_s'] = settle
            raw = json.dumps(self.plan).encode(); self.path.write_bytes(raw)
            with self.assertRaises(ValueError): load_plan(self.path, hashlib.sha256(raw).hexdigest())

    def test_settle_extends_wait_before_next_command(self):
        class FineClock:
            def __init__(self): self.now = 0
            def sleep(self, seconds): self.now += seconds
            def monotonic(self): self.now += .1; return self.now
        clock = FineClock()
        class TimedPort(Port):
            def __init__(self): super().__init__(); self.times = []
            def write(self, data): self.times.append(clock.now); return super().write(data)
        self.plan['stages'][0]['steps'][0]['extra_settle_s'] = 2
        port = TimedPort()
        result = execute(self.root, self.plan, self.plan['stages'][0], self.digest, port, clock)
        self.assertEqual(result['status'], 'sent_pending_observation')
        self.assertGreaterEqual(port.times[1] - port.times[0], 3.5)
