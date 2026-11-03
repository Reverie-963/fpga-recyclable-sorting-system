"""Show recorded device measurements only for the exact monitored model/policy."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = ROOT / 'local-only/rocm-cpu-benchmark/20261029-v8-preparation/result.json'


def benchmark_summary(health, path=DEFAULT_REPORT):
    unavailable = dict(available=False, reason='no_matching_device_benchmark')
    try:
        report = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        family = health['model_family']
        execution = health['execution']
        precision = execution['precision']
        if (family not in ('yolo11s', 'yolov8s') or execution['device'] != 'cuda:0' or
            precision not in ('fp32', 'fp16') or
            report['models'][family]['model_sha256'] != health['model_sha256'] or
            report['policy'] != health['inference_policy'] or report['batch'] != 1 or
            report['torch'] != execution['torch'] or report['hip'] != execution['hip'] or
            report.get('miopen_enabled') != execution.get('miopen_enabled') or
            report['gpu'] != execution['device_name']):
            return unavailable
        cpu = report['profiles'][family+'/cpu_fp32']['pipeline']
        gpu = report['profiles'][family+'/gpu_'+precision]['pipeline']
        same_precision = report['profiles'][family+'/gpu_fp32']['pipeline']
        for profile in (cpu, gpu, same_precision):
            for key in ('median_ms', 'mean_ms', 'max_ms', 'p95_ms'):
                value = profile[key]
                if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                    return unavailable
        if (type(cpu['samples']) is not int or cpu['samples'] != gpu['samples'] or
            cpu['samples'] != report['repeats'] * len(report['image_paths']) or cpu['samples'] < 1):
            return unavailable
        return dict(available=True, schema_version='monitor-device-benchmark-v1',
            source='offline_benchmark', model_sha256=health['model_sha256'],
            model_name=health['model_name'], gpu_precision=precision,
            measured_at=report['completed_at'], cpu_threads=report['cpu_threads'],
            samples_per_profile=cpu['samples'], batch=1,
            cpu_median_ms=cpu['median_ms'], gpu_median_ms=gpu['median_ms'],
            cpu_mean_ms=cpu['mean_ms'], gpu_mean_ms=gpu['mean_ms'],
            median_speedup=cpu['median_ms']/gpu['median_ms'],
            batch_speedup=cpu['mean_ms']/gpu['mean_ms'],
            same_fp32_median_speedup=cpu['median_ms']/same_precision['median_ms'],
            gpu_max_ms=gpu['max_ms'], gpu_p95_ms=gpu['p95_ms'],
            concurrent_training=any(isinstance(report.get(k), dict) and
                report[k].get('process_running') is True for k in ('training_at_start', 'training_at_end')),
            includes=['preprocess', 'model', 'postprocess', 'cpu_mask_rle'],
            live_speedup=False)
    except (KeyError, TypeError, ValueError, OSError):
        return unavailable
