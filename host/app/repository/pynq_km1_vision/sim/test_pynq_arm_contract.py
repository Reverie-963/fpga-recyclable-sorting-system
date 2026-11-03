"""Compose the teammate's real prediction/planning with the new arm boundary."""
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from src.arm import execute, packet, plan_pick
from src.arm_plan import inspect_plan
from src.moving_targets import intercept
from src.recycling_run import MovingArm, moving_plan, validate_dynamic


def config():
    c = json.loads(Path('config/cap_deformation_board.json').read_text())
    c['arm'].update(transport='openmv_v3', commissioned=True,
                    joint_limits_pwm=[[700, 2300]] * 5 + [[1000, 1700]])
    # Broad limits here are simulation fixtures, never device calibration.
    return c


def choice(c, label='cap_deformed'):
    track = {'id': 17, 'xy': [0, 150], 'belt': 0, 'stamp': 10,
             'seen': 5, 'label': label, 'semantic_stamp': 10, 'confidence': .99}
    sample = {'velocity_mm_s': 2, 'position_mm': 0, 'time': 10, 'uncertainty_s': .001}
    horizon = 2 * (c['arm']['move_ms'] / 1000 + c['arm']['settle_s']) + c['dynamic']['grip_contact_s']
    return intercept(track, sample, 10, horizon, c)


class PynqArmContractTests(unittest.TestCase):
    def test_prediction_to_dynamic_plan_keeps_ps_timing_and_steps(self):
        c = config()
        target = choice(c)
        self.assertIsNotNone(target)
        self.assertEqual(target['track_id'], 17)
        self.assertAlmostEqual(target['xy'][1], 157.8)
        self.assertAlmostEqual(target['contact_time'], 13.9)
        plan = moving_plan(target, c)
        self.assertEqual([s[0] for s in plan], ['above_target', 'descend', 'close', 'lift',
                                             'above_bin', 'lower_bin', 'release', 'leave_bin', 'return'])
        close = next(s for s in plan if s[0] == 'close')
        self.assertEqual(close[2], c['dynamic']['grip_ms'])
        self.assertEqual(close[1], packet([(5, c['arm']['close_pwm'])], c['dynamic']['grip_ms']))
        self.assertEqual(plan[0][1], plan_pick(*target['xy'], target['label'], c)[3][1])

    def test_normal_and_stale_results_do_not_become_motion_tasks(self):
        c = config()
        self.assertIsNone(choice(c, 'cap_ok'))
        c['dynamic']['semantic_max_age_s'] = .1
        track = {'id': 17, 'xy': [0, 150], 'belt': 0, 'stamp': 10, 'seen': 5,
                 'label': 'cap_deformed', 'semantic_stamp': 9, 'confidence': .99}
        sample = {'velocity_mm_s': 2, 'position_mm': 0, 'time': 10, 'uncertainty_s': 0}
        with patch('serial.Serial') as serial:
            self.assertIsNone(intercept(track, sample, 10, 3.9, c))
            serial.assert_not_called()

    def test_dynamic_preview_reports_bridge_blocker_without_opening_serial(self):
        c = config()
        with patch('serial.Serial') as serial:
            report = inspect_plan(moving_plan(choice(c), c), c, dynamic=True)
            serial.assert_not_called()
        self.assertEqual(report['execution_blockers'], ['Dynamic OpenMV bridge timing is unverified'])
        self.assertFalse(report['software_checks_passed'])

    def test_dynamic_execution_cannot_bypass_gate_with_commissioned_flags(self):
        c = config()
        c['dynamic']['commissioned'] = True
        with patch('serial.Serial') as serial:
            with self.assertRaisesRegex(ValueError, 'timing'):
                validate_dynamic(c, True)
            with self.assertRaisesRegex(ValueError, 'timing'):
                MovingArm(c)
            serial.assert_not_called()

    def test_ps_executor_preserves_guard_and_poll_during_motion(self):
        c = config()
        c['arm']['transport'] = 'direct'
        clock = {'t': 20.0}
        guard = Mock()
        link = Mock()
        def send(data, duration, check):
            check()
            return clock['t']
        link.send.side_effect = send
        def poll():
            clock['t'] += .1
        with patch('src.recycling_run.ArmConnection', return_value=link), \
                patch('time.monotonic', side_effect=lambda: clock['t']), patch('time.sleep'):
            arm = MovingArm(c)
            stamp = arm.send(('above_target', packet([(0, 1500)], 300), 300), poll, guard)
            arm.close()
        self.assertEqual(stamp, 20)
        self.assertGreaterEqual(clock['t'], 20 + .3 + c['arm']['settle_s'])
        guard.assert_called_once()
        link.close.assert_called_once()

    def test_malformed_later_direct_step_stops_before_open(self):
        c = config()
        c['arm']['transport'] = 'direct'
        plan = [('open', packet([(5, 1000)], 800), 800),
                ('bad_later', packet([(0, 1500)], 800), 900)]
        with patch('serial.Serial') as serial:
            with self.assertRaisesRegex(ValueError, 'duration differs'):
                execute(plan, c, True)
            serial.assert_not_called()
