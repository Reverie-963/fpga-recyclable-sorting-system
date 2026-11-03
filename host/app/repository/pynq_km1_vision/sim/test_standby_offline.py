"""The runnable offline rehearsal must stay independent of the live station."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.standby_offline import run_suite


class OfflineStandbyTests(unittest.TestCase):
    def test_real_state_machine_scenarios_use_no_network_station_or_files(self):
        forbidden = AssertionError('Offline rehearsal accessed a live dependency')
        with patch('socket.socket', side_effect=forbidden), \
             patch('src.monitor_operation.StationBackend', side_effect=forbidden), \
             patch('builtins.open', side_effect=forbidden), \
             patch('pathlib.Path.read_text', side_effect=forbidden):
            report = run_suite()
        self.assertEqual(report['status'], 'passed', report)
        self.assertEqual(report['passed'], report['total'])
        self.assertFalse(report['real_devices_used'])
        self.assertFalse(report['physical_motion'])
        by_name = {v['name']: v for v in report['cases']}
        for label in ('auto', 'paper', 'metal', 'plastic'):
            case = by_name['clear_then_new_'+label]
            self.assertEqual([c['simulated_attempt'] for c in case['calls']], [False, False, False, True])
            self.assertTrue(case['duplicate_request_rejected'])
        self.assertEqual(by_name['stop_during_observation']['simulated_stop_requests'], 1)
        self.assertEqual(by_name['cancel_during_observation']['simulated_stop_requests'], 0)
        self.assertEqual(by_name['failed_attempt_not_retried']['result']['status'], 'needs_review')
        self.assertEqual(by_name['old_target_never_clears']['simulated_attempts'], 0)
        self.assertEqual(by_name['page_heartbeat_lost']['reason'], 'page_disconnected')

    def test_command_line_writes_report_and_refuses_to_replace_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'offline.json'
            cmd = [sys.executable, '-m', 'src.standby_offline', '--output', str(path)]
            first = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            self.assertEqual(first.returncode, 0, first.stderr)
            report = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(report['status'], 'passed')
            original = path.read_bytes()
            second = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
