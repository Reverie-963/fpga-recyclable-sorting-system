import copy
import json
from pathlib import Path
import tempfile
import unittest

from src.monitor_fpga_benchmark import fpga_benchmark_summary


class FPGABenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        root=Path(self.temp.name);self.report=root/'report.json';self.config=root/'config.json'
        assets={'vision.bit':'a'*64,'vision.hwh':'b'*64}
        self.config.write_text(json.dumps(dict(statistics=dict(backend='pl_http',assets=assets))))
        cpu=dict(samples=20,median_ms=2,mean_ms=2.5,p95_ms=3,max_ms=4,sequential_jobs_per_s=400)
        pl=dict(samples=20,median_ms=100,mean_ms=125,p95_ms=150,max_ms=200,sequential_jobs_per_s=8)
        role=dict(cpu=cpu,pl_roundtrip=pl,with_cpu_check=pl,correct=20,total=20,median_speedup=999)
        self.data=dict(assets=assets,image_size=[640,480],all_statistics_and_output_hashes_equal=True,
            kernel_only_measured=False,cpu_utilization_measured=False,cases=5,repeats=4,
            roles=dict(geometry=role,visual=copy.deepcopy(role)),comparisons=40,
            completed_at='2026-10-29T09:44:26+00:00',cpu_device='CPU',pl_device='PL')

    def result(self):
        self.report.write_text(json.dumps(self.data))
        return fpga_benchmark_summary(self.report,self.config)

    def test_recompute_cost_instead_of_trusting_speedup_and_no_current_board_claim(self):
        result=self.result()
        self.assertTrue(result['available'])
        self.assertFalse(result['all_roles_faster'])
        self.assertEqual(result['rows'][0]['median_speedup'],0.02)
        self.assertFalse(result['current_board_verified'])

    def test_asset_mismatch_cannot_show_earlier_board_measurements(self):
        self.data['assets']['vision.bit']='c'*64
        self.assertFalse(self.result()['available'])

    def test_incomplete_or_invalid_metrics_are_hidden(self):
        for key,value in [('median_ms',0),('mean_ms',float('nan')),('samples',19)]:
            with self.subTest(key=key):
                original=self.data['roles']['visual']['cpu'][key]
                self.data['roles']['visual']['cpu'][key]=value
                self.assertFalse(self.result()['available'])
                self.data['roles']['visual']['cpu'][key]=original
        self.data['roles']['visual']['correct']=19
        self.assertFalse(self.result()['available'])

    def test_missing_report_is_unavailable(self):
        self.assertFalse(fpga_benchmark_summary(self.report,self.config)['available'])


if __name__=='__main__':unittest.main()
