"""Supervised fast-route integration; synthetic scene and serial only."""
import copy
from pathlib import Path
import tempfile
import unittest

from src.arm import packet
from src.arm_route_preview import prepare
from src.sorting_demo import DemoClock, DemoMotion, DemoSource
from src.sorting_io import Statistics
from src.sorting_known import KnownController, validate_known_config
from src.sorting_loop import Journal
from src.sorting_motion import compact_plan, motion_wait
from sim.test_sorting_known import config
from openmv_gripper.km1_arm_bridge import validate_packet


class CompactCycleTests(unittest.TestCase):
    def cycle(self, label, profile='compact', failure=None, wrong_bin=False):
        c = config()
        clock = DemoClock()
        source = DemoSource(c, clock)
        source.items = [item for item in source.items if item['label'] == label]
        motion = DemoMotion(source, clock)
        motion.fail_step, motion.wrong_bin = failure, wrong_bin
        with tempfile.TemporaryDirectory() as folder:
            journal = Journal(Path(folder))
            controller = KnownController(c, source, Statistics(), motion, journal,
                source.reference, source.reference, folder, mode='simulation',
                motion_profile=profile, clock=clock.monotonic, sleep=clock.sleep)
            with journal:
                report = controller.cycle()
                pending = journal.pending()
                if failure:
                    with self.assertRaisesRegex(ValueError, 'unresolved_previous_task'):
                        controller.cycle()
        return report, motion, pending

    def test_three_materials_remain_correct_and_use_actual_shorter_packets(self):
        for label in ('paper', 'metal', 'plastic'):
            with self.subTest(label=label):
                baseline, _, _ = self.cycle(label, 'qualified')
                fast, motion, pending = self.cycle(label)
                self.assertEqual(fast['status'], 'simulated_visual_pass', fast)
                evidence = fast['choice']['motion_optimization']
                self.assertLess(evidence['candidate_motion_wait_s'], evidence['baseline_motion_wait_s']*.5)
                self.assertLess(fast['elapsed_s'], baseline['elapsed_s'])
                self.assertFalse(evidence['physical_speed_verified'])
                self.assertTrue(fast['supervised_motion_trial'])
                self.assertFalse(fast['production_quality_claimed'])
                self.assertIsNone(pending)
                names = motion.sent
                self.assertLess(names.index('source_hover'), names.index('source_grip'))
                self.assertLess(names.index('source_grip'), names.index('close'))
                self.assertLess(names.index('close'), names.index('release'))
                self.assertEqual(names[-1], 'return')
                for step, executed in zip(fast['plan'], fast['steps']):
                    ms = validate_packet(step['command'], [[500,2500]]*6)
                    self.assertEqual(ms, step['duration_ms'])
                    self.assertGreaterEqual(executed['required_wait_s'], ms/1000)

    def test_new_profile_does_not_rewrite_qualified_config_or_route(self):
        c = config()
        before = copy.deepcopy(c)
        plan = [('home',packet(list(enumerate(c['known']['prepare_pwm']))+[(4,1500),(5,1000)],5000),5000),
                ('source_hover',packet([(0,1500)],5000),5000),
                ('close',packet([(5,1300)],2000),2000),
                ('release',packet([(5,1000)],2000),2000),
                ('return',packet(list(enumerate(c['known']['initial_pwm'][:4])),5000),5000)]
        original = copy.deepcopy(plan)
        compact, _ = compact_plan(c, plan)
        self.assertEqual(c, before)
        self.assertEqual(plan, original)
        def end_state(p):
            obj = dict(start_pwm=c['known']['initial_pwm'], stages=[dict(steps=[
                dict(command=d.decode(),duration_ms=ms) for _,d,ms in p])])
            return prepare(obj)[0][-1]['end']
        self.assertEqual(end_state(compact), end_state(plan))

    def test_partial_write_stops_candidate_without_retry(self):
        result, motion, pending = self.cycle('paper', failure='close')
        self.assertEqual(result['status'], 'needs_review')
        self.assertEqual(motion.sent[-1], 'close')
        self.assertIsNotNone(pending)

    def test_faster_route_keeps_wrong_bin_detection(self):
        result, _, pending = self.cycle('metal', wrong_bin=True)
        self.assertEqual(result['status'], 'needs_review')
        self.assertIn('no_unique_new_target_in_correct_bin', result['reason'])
        self.assertIsNotNone(pending)

    def test_settle_budget_and_unknown_profile(self):
        arm = config()['arm']
        self.assertEqual(motion_wait(arm,'release',2000),3)
        self.assertEqual(motion_wait(arm,'release',1000,'compact'),1.5)
        self.assertEqual(motion_wait(arm,'close',1000,'compact'),1.3)
        with self.assertRaises(ValueError): motion_wait(arm,'close',1000,'unknown')

    def test_unknown_config_profile_is_rejected(self):
        c = config()
        c['known']['motion_profile'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'unknown_known_motion_profile'):
            validate_known_config(c)


if __name__ == '__main__': unittest.main()
