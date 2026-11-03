"""Check deployment isolation and byte-for-byte route equivalence without devices."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sim.test_sorting_unified import test_config, setup_policy
from src.board_control_shadow import ledger_checks, plan_case, run


class BoardShadowTests(unittest.TestCase):
    def fixture(self):
        config = test_config()
        policy = setup_policy(config)
        case = dict(id='paper', label='paper', card_xy=[0, 5], axis_deg=90, slot_index=0)
        case['expected_sha256'] = plan_case(config, policy, case)[0]
        return dict(kind='board-control-shadow-fixture-v1', config=config, policy=policy,
                    cases=[case], source=dict(synthetic=True))

    def test_real_routes_standby_and_ledger_without_devices(self):
        fixture = self.fixture()
        before = copy.deepcopy(fixture)
        with tempfile.TemporaryDirectory() as scratch, \
             patch('serial.Serial', side_effect=AssertionError('no UART')), \
             patch('cv2.VideoCapture', side_effect=AssertionError('no camera')), \
             patch('src.monitor_operation.StationBackend', side_effect=AssertionError('no live backend')):
            report = run(fixture, Path(scratch))
        self.assertEqual(report['status'], 'passed')
        self.assertEqual(report['standby']['passed'], 18)
        self.assertFalse(report['execution_enabled'])
        self.assertFalse(report['production_state_changed'])
        self.assertEqual(before, fixture)

    def test_wrong_route_digest_fails(self):
        fixture = self.fixture()
        fixture['cases'][0]['expected_sha256'] = '0'*64
        with tempfile.TemporaryDirectory() as scratch:
            report = run(fixture, Path(scratch))
        self.assertEqual(report['status'], 'failed')

    def test_ledger_exclusive_and_pending_survives(self):
        with tempfile.TemporaryDirectory() as scratch:
            result = ledger_checks(scratch)
        self.assertTrue(result['passed'])
        self.assertFalse(result['production_ledger_used'])

    def test_unbounded_or_wrong_fixtures_rejected(self):
        for fixture in ({'kind':'execute'}, dict(self.fixture(), cases=[]),
                        dict(self.fixture(), cases=[{}]*121)):
            with self.assertRaises(ValueError):
                run(fixture, '.')


if __name__ == '__main__':
    unittest.main()
