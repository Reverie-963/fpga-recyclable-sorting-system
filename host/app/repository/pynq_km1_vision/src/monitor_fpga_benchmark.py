"""Read recorded PL roundtrip measurements without contacting the board."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT/'local-only/fpga-device-benchmark/latest.json'
CONFIG = ROOT/'pynq_km1_vision/config/sorting_station.local.json'


def fpga_benchmark_summary(path=REPORT, config_path=CONFIG):
    unavailable = dict(available=False, reason='no_matching_fpga_benchmark')
    try:
        report=json.loads(Path(path).read_text(encoding='utf-8-sig'))
        config=json.loads(Path(config_path).read_text(encoding='utf-8-sig'))['statistics']
        assets=report['assets']
        if (config['backend']!='pl_http' or assets!=config['assets'] or
            set(assets)!={'vision.bit','vision.hwh'} or
            any(len(v)!=64 or any(c not in '0123456789abcdef' for c in v) for v in assets.values()) or
            report['image_size']!=[640,480] or report['all_statistics_and_output_hashes_equal'] is not True or
            report['kernel_only_measured'] is not False or report['cpu_utilization_measured'] is not False):
            return unavailable
        rows=[]
        for role in ('geometry','visual'):
            item=report['roles'][role]
            count=item['total']
            if type(count) is not int or count<1 or item['correct']!=count or count!=report['cases']*report['repeats']:
                return unavailable
            for name in ('cpu','pl_roundtrip','with_cpu_check'):
                stats=item[name]
                if stats['samples']!=count:return unavailable
                for key in ('median_ms','mean_ms','p95_ms','max_ms','sequential_jobs_per_s'):
                    value=stats[key]
                    if type(value) not in (int,float) or not math.isfinite(value) or value<=0:return unavailable
            rows.append(dict(role=role,correct=count,total=count,cpu=item['cpu'],
                pl_roundtrip=item['pl_roundtrip'],with_cpu_check=item['with_cpu_check'],
                median_speedup=item['cpu']['median_ms']/item['pl_roundtrip']['median_ms'],
                batch_speedup=item['cpu']['mean_ms']/item['pl_roundtrip']['mean_ms']))
        if report['comparisons']!=sum(r['total'] for r in rows):return unavailable
        return dict(available=True,source='recorded_device_benchmark',measured_at=report['completed_at'],
            image_size=report['image_size'],assets=assets,rows=rows,comparisons=report['comparisons'],
            current_board_verified=False,kernel_only_measured=False,cpu_utilization_measured=False,
            cpu_device=report['cpu_device'],pl_device=report['pl_device'],
            all_roles_faster=all(r['median_speedup']>1 for r in rows),
            cases=report['cases'],repeats=report['repeats'])
    except (OSError,KeyError,TypeError,ValueError):
        return unavailable
