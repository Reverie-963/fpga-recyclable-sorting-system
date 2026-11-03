import copy
import unittest

from src.station_history import audit_phase


class StationHistoryTests(unittest.TestCase):
    def setUp(self):
        self.start = [1500]*5+[1000]
        command = '{#000P1600T2000!#005P1300T2000!}'
        step = dict(command=command, duration_ms=2000)
        self.plan = dict(start_pwm=self.start, stages=[dict(steps=[step])])
        self.execution = dict(plan_sha256='a'*64, status='sent_pending_observation', serial_closed=True,
            steps=[dict(step, echo_matched=True, bytes_written=len(command), received_hex=command.encode().hex())])

    def test_recovers_sparse_final_state_without_motion(self):
        before = copy.deepcopy(self.start)
        commands, end = audit_phase(self.plan, self.execution, 'a'*64, self.start)
        self.assertEqual(len(commands), 1)
        self.assertEqual(end, [1600, 1500, 1500, 1500, 1500, 1300])
        self.assertEqual(self.start, before)

    def test_incomplete_or_wrongly_bound_execution_rejected(self):
        for key, value in (('plan_sha256', 'wrong'), ('status', 'stopped_needs_review'), ('serial_closed', False)):
            bad = copy.deepcopy(self.execution); bad[key] = value
            with self.assertRaises(ValueError): audit_phase(self.plan, bad, 'a'*64, self.start)

    def test_echo_flag_alone_cannot_forge_complete_echo(self):
        for key, value in (('bytes_written', 1), ('received_hex', b'partial'.hex()), ('echo_matched', False)):
            bad = copy.deepcopy(self.execution); bad['steps'][0][key] = value
            with self.assertRaises(ValueError): audit_phase(self.plan, bad, 'a'*64, self.start)

    def test_phase_discontinuity_and_truncated_steps_rejected(self):
        start = list(self.start); start[0] = 1600
        with self.assertRaises(ValueError): audit_phase(self.plan, self.execution, 'a'*64, start)
        self.execution['steps'] = []
        with self.assertRaises(ValueError): audit_phase(self.plan, self.execution, 'a'*64, self.start)


if __name__ == '__main__': unittest.main()
