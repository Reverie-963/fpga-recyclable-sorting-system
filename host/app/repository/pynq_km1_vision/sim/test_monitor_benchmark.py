"""Recorded speedups must match the actual model, precision and inference policy."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from src.monitor_benchmark import benchmark_summary


class BenchmarkDisplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'result.json'
        self.health = dict(model_family='yolov8s', model_name='YOLOv8s-seg', model_sha256='v8-hash',
            inference_policy=dict(image_size=640), execution=dict(device='cuda:0', precision='fp16',
                torch='torch-build', hip='hip-build', device_name='gpu'))
        def profile(median, mean):
            return dict(pipeline=dict(median_ms=median, mean_ms=mean, max_ms=1000, p95_ms=900, samples=40))
        self.report = dict(models={'yolov8s':dict(model_sha256='v8-hash')}, policy=dict(image_size=640),
            batch=1, torch='torch-build', hip='hip-build', gpu='gpu', repeats=5, image_paths=['image']*8,
            profiles={'yolov8s/cpu_fp32':profile(100,100), 'yolov8s/gpu_fp16':profile(25,80),
                      'yolov8s/gpu_fp32':profile(50,120)}, completed_at='2026-10-29T00:00:00Z',
            cpu_threads=4, training_at_start=dict(process_running=True))

    def tearDown(self):
        self.temp.cleanup()

    def result(self, report=None, health=None):
        self.path.write_text(json.dumps(self.report if report is None else report), encoding='utf-8')
        return benchmark_summary(self.health if health is None else health, self.path)

    def test_typical_and_whole_batch_use_different_measured_statistics(self):
        result = self.result()
        self.assertTrue(result['available'])
        self.assertEqual(result['median_speedup'], 4)
        self.assertEqual(result['batch_speedup'], 1.25)
        self.assertFalse(result['live_speedup'])
        self.assertTrue(result['concurrent_training'])

    def test_fp32_uses_fp32_measurements_and_preserves_slowdown(self):
        health = copy.deepcopy(self.health)
        health['execution']['precision'] = 'fp32'
        result = self.result(health=health)
        self.assertEqual(result['median_speedup'], 2)
        self.assertLess(result['batch_speedup'], 1)
        self.assertEqual(result['gpu_median_ms'], 50)

    def test_another_checkpoint_policy_or_runtime_has_no_matching_claim(self):
        for field in ('model_sha256', 'inference_policy', 'model_family'):
            health = copy.deepcopy(self.health)
            health[field] = 'changed'
            self.assertFalse(self.result(health=health)['available'])
        for field in ('device', 'precision', 'torch', 'hip', 'device_name', 'miopen_enabled'):
            health = copy.deepcopy(self.health)
            health['execution'][field] = 'changed'
            self.assertFalse(self.result(health=health)['available'])

    def test_invalid_numbers_or_sample_counts_cannot_be_displayed(self):
        for value in (0, -1, float('nan'), float('inf'), True):
            report = copy.deepcopy(self.report)
            report['profiles']['yolov8s/gpu_fp16']['pipeline']['median_ms'] = value
            self.assertFalse(self.result(report=report)['available'])
        report = copy.deepcopy(self.report)
        report['profiles']['yolov8s/gpu_fp16']['pipeline']['samples'] = 39
        self.assertFalse(self.result(report=report)['available'])

    def test_missing_or_broken_report_never_breaks_monitor(self):
        self.assertFalse(benchmark_summary(self.health, self.path)['available'])
        self.path.write_text('broken', encoding='utf-8')
        self.assertFalse(benchmark_summary(self.health, self.path)['available'])
        self.assertFalse(self.result(health={})['available'])


if __name__ == '__main__':
    unittest.main()
