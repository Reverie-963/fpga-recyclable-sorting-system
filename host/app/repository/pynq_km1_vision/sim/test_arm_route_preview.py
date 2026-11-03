import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.arm import packet
from src.arm_route_preview import make_preview
from src.static_supervised_trial import load_plan


def step(pairs, ms=3000, settle=0):
    return dict(command=packet(pairs, ms).decode(), duration_ms=ms, extra_settle_s=settle)


class RoutePreviewTests(unittest.TestCase):
    def setUp(self):
        self.plan = dict(kind='supervised_fixed_path_trial_v1', task_id='offline',
            start_pwm=[1500, 1632, 1987, 928, 1500, 1000], stages=[
                dict(name='cycle', observe='inspect', steps=[
                    step([(0, 1450)]), step([(0, 1400)]),
                    step([(5, 1500)], 2000, .5),
                    step([(0, 1450)]), step([(0, 1500)]),
                    step([(5, 1000)], 2000, .5)])])

    def preview(self, plan=None, **policy):
        return make_preview(plan or self.plan, policy)

    def test_shortest_route_preserves_grip_sequence_and_final_state(self):
        before = copy.deepcopy(self.plan)
        p = self.preview()
        self.assertEqual(p['candidate_step_count'], 4)
        rows = p['stages'][0]['steps']
        self.assertEqual([r['source_indices'] for r in rows], [[0, 1], [2], [3, 4], [5]])
        self.assertEqual(rows[-1]['target_pwm'], self.plan['start_pwm'])
        self.assertEqual(rows[1]['duration_ms'], 2000)
        self.assertEqual(rows[1]['extra_settle_s'], .5)
        self.assertEqual(before, self.plan)

    def test_sparse_commands_accumulate_all_joint_targets(self):
        self.plan['stages'][0]['steps'] = [step([(1, 1628)]), step([(2, 1983)])]
        p = self.preview(max_tip_deviation_mm=10)
        rows = p['stages'][0]['steps']
        self.assertEqual(len(rows), 1)
        self.assertIn('#001P1628', rows[0]['command'])
        self.assertIn('#002P1983', rows[0]['command'])

    def test_explicit_clearance_pose_and_stages_cannot_be_skipped(self):
        self.assertEqual(self.preview(keep_after={'cycle': [0, 3]})['candidate_step_count'], 6)
        self.plan['stages'].append(dict(name='return_home', observe='inspect again',
            steps=[step([(0, 1550)])]))
        p = self.preview()
        self.assertEqual(len(p['stages']), 2)
        self.assertEqual(p['stages'][1]['steps'][0]['source_indices'], [0])

    def test_settle_is_a_barrier(self):
        self.plan['stages'][0]['steps'][0]['extra_settle_s'] = .1
        self.assertEqual(self.preview()['candidate_step_count'], 5)

    def test_large_base_rotation_cannot_merge_with_arm_motion(self):
        self.plan['stages'][0]['steps'] = [step([(0, 800)]), step([(1, 1615)])]
        self.assertEqual(self.preview(max_tip_deviation_mm=100)['candidate_step_count'], 2)

    def test_tip_detour_cannot_be_erased_by_identical_endpoints(self):
        self.plan['stages'][0]['steps'] = [step([(1, 1700)]), step([(1, 1632)])]
        self.assertEqual(self.preview(max_tip_deviation_mm=.1)['candidate_step_count'], 2)

    def test_rate_limit_applies_to_merged_delta(self):
        self.plan['stages'][0]['steps'] = [step([(0, 1300)]), step([(0, 1100)])]
        p = self.preview(max_arm_pwm_per_s=100)
        self.assertEqual(p['candidate_step_count'], 1)
        self.assertEqual(p['stages'][0]['steps'][0]['duration_ms'], 4000)
        self.assertAlmostEqual(p['candidate_executor_budget_s'], 4.5)

    def test_overlapped_echo_budget_preserves_grip_settling(self):
        old = self.preview()
        fast = self.preview(wait_policy='motion_and_echo')
        self.assertEqual(old['candidate_step_count'], fast['candidate_step_count'])
        self.assertAlmostEqual(old['candidate_executor_budget_s']-fast['candidate_executor_budget_s'], 1.)

    def test_unknown_start_or_invalid_policy_fails(self):
        for policy in ({'max_arm_pwm_per_s': True}, {'max_tip_deviation_mm': float('nan')},
                       {'min_move_ms': 1000}, {'keep_after': {'wrong_stage': [1]}},
                       {'keep_after': {'cycle': [99]}}):
            with self.assertRaises(ValueError): self.preview(**policy)
        self.plan.pop('start_pwm')
        with self.assertRaises(ValueError): self.preview()

    def test_preview_kind_is_rejected_by_physical_executor(self):
        p = self.preview()
        self.assertFalse(p['automatic_motion_allowed'])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'preview.json'
            raw = json.dumps(p).encode()
            path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, 'explicit supervised physical plan'):
                load_plan(path, hashlib.sha256(raw).hexdigest())


if __name__ == '__main__': unittest.main()
