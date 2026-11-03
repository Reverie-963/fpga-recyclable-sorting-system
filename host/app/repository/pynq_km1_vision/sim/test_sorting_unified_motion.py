"""Timing candidates must preserve geometry, protocol and complete waits."""
import json
from pathlib import Path
import re
import unittest

from sim.test_sorting_unified import test_config, setup_policy
from sim import test_sorting_unified as unified_tests
from src.sorting_unified_plan import compile_route, validate_policy
from src.sorting_unified_motion import movement_wait
from openmv_gripper.km1_arm_bridge import validate_packet


def preset():
    return json.loads((Path(__file__).parents[1]/'config/sorting_motion_20s.example.json').read_text())


class TimingTests(unittest.TestCase):
    def setUp(self):
        self.c = test_config()
        self.policy = setup_policy(self.c)

    def test_all_angles_keep_waypoints_joint_targets_and_height_audit(self):
        fast = dict(self.policy, motion_timing=preset())
        for label in self.c['profiles']:
            for angle in range(0, 360, 30):
                with self.subTest(label=label, angle=angle):
                    slot = self.policy['bin_slots'][label][0]
                    old = compile_route(self.c, label, [0, 5], angle, self.policy, slot)
                    new = compile_route(self.c, label, [0, 5], angle, fast, slot)
                    self.assertEqual(old['nominal_interpolation'], new['nominal_interpolation'])
                    self.assertEqual([s['name'] for s in old['steps']], [s['name'] for s in new['steps']])
                    for a, b, audit in zip(old['steps'], new['steps'], new['nominal_interpolation']):
                        self.assertEqual(re.sub(r'T\d{4}', '', a['command']), re.sub(r'T\d{4}', '', b['command']))
                        self.assertEqual(validate_packet(b['command'], [[500, 2500]]*6), b['duration_ms'])
                        self.assertGreaterEqual(movement_wait(fast, b['name'], b['duration_ms']), b['duration_ms']/1000)
                        for i, rate in enumerate(fast['motion_timing']['max_pwm_per_s']):
                            self.assertLessEqual(abs(audit['target_pwm'][i]-audit['start_pwm'][i]) /
                                                 (b['duration_ms']/1000), rate+1e-8)
                    self.assertLess(new['motion_timing']['candidate_motion_wait_s'], 20)
                    self.assertFalse(new['motion_timing']['physical_speed_verified'])

    def test_legacy_policy_retains_original_timing(self):
        route = compile_route(self.c, 'paper', [0, 5], 90, self.policy, self.policy['bin_slots']['paper'][0])
        self.assertNotIn('motion_timing', route)
        for s in route['steps']:
            self.assertEqual(s['duration_ms'], 2000 if s['name'] in ('close', 'release') else 5000)
        self.assertEqual(movement_wait(self.policy, 'home', 5000), 5.5)
        self.assertEqual(movement_wait(self.policy, 'release', 2000), 3)

    def test_invalid_or_unreviewed_timing_is_rejected(self):
        for key, value in [('min_move_ms', True), ('close_ms', 0), ('release_settle_s', 0),
                           ('max_pwm_per_s', [10000]*6), ('max_pwm_per_s', [900]*5),
                           ('physical_speed_verified', True), ('operator_statement', ''),
                           ('target_cycle_s', float('nan'))]:
            with self.subTest(key=key, value=value):
                timing = preset(); timing[key] = value
                with self.assertRaises(ValueError):validate_policy(dict(self.policy, motion_timing=timing), self.c)


class FastCycleTests(unified_tests.UnifiedBatchTests):
    """Run the real controller's existing failure/identity checks at the new speed."""
    def setUp(self):
        super().setUp()
        self.policy['motion_timing'] = preset()


if __name__ == '__main__':
    unittest.main()
