"""Pure stdlib recalculation from archived observations; no network or motion."""
from pathlib import Path
import csv,collections,json,math,statistics
root=Path(__file__).resolve().parents[2]
def quantile(values,q):
 x=sorted(values);p=(len(x)-1)*q;i=int(p);j=min(i+1,len(x)-1)
 return x[i]+(x[j]-x[i])*(p-i)
rows=[json.loads(s)for s in(root/'metrics/logs/online_raw.jsonl').read_text(encoding='utf-8').splitlines()if s.strip()]
summary=json.loads((root/'metrics/logs/online_summary.json').read_text(encoding='utf-8'))
assert len(rows)==800
checks={}
for group,group_rows in __import__('itertools').groupby(sorted(rows,key=lambda r:r['kind']),key=lambda r:r['kind']):
 values=list(group_rows);assert len(values)==200 and all(r['ok']for r in values)
 for name,q in [('p50',.5),('p95',.95),('p99',.99),('max',1)]:
  calculated=quantile([r['rtt_ms']for r in values],q)
  assert math.isclose(calculated,summary[group]['rtt_ms'][name],abs_tol=1e-6)
 checks[group]=dict(samples=len(values),rtt_ms={k:quantile([r['rtt_ms']for r in values],q)for k,q in [('p50',.5),('p95',.95),('p99',.99),('max',1)]})
with(root/'metrics/logs/task_trials.csv').open(encoding='utf-8-sig')as f:trials=list(csv.DictReader(f))
assert len(trials)==11
passed=sum(r['strict_physical_truth']=='pass'for r in trials)
assert passed==8
groups={}
for material in ['metal','paper','plastic']:
 group=[r for r in trials if r['material']==material]
 groups[material]=dict(attempts=len(group),strict_pass=sum(r['strict_physical_truth']=='pass'for r in group))
assert groups=={'metal':{'attempts':3,'strict_pass':1},'paper':{'attempts':5,'strict_pass':5},'plastic':{'attempts':3,'strict_pass':2}}
ai=json.loads((root/'metrics/logs/ai_batch1.json').read_text(encoding='utf-8'))
assert len(ai['rows'])==162
for key,stats in ai['summary'].items():
 for stat,q in [('p50_ms',.5),('p95_ms',.95),('p99_ms',.99),('max_ms',1)]:
  assert math.isclose(quantile([r[key]for r in ai['rows']],q),stats[stat],abs_tol=1e-6)
quality=json.loads((root/'metrics/logs/quality_gpu_derived.json').read_text(encoding='utf-8'))
parts=[p for p in quality['GPU_quality_components']if p['variant']=='reference']
f1=statistics.mean(2*p['TP']/(2*p['TP']+p['FP']+p['FN'])for p in parts)
miou=statistics.mean(p['pixel_TP']/(p['pixel_TP']+p['pixel_FP']+p['pixel_FN'])for p in parts)
assert math.isclose(f1,quality['GPU_quality']['reference']['macro_F1'],abs_tol=1e-12)
assert math.isclose(miou,quality['GPU_quality']['reference']['mIoU'],abs_tol=1e-12)
result=dict(passed=True,online=checks,physical_exploratory=dict(attempts=11,strict_pass=passed,groups=groups),ai_samples=162,quality=dict(macro_f1=f1,foreground_miou=miou),overall_success_rate=None,population_motion_p99=None,new_measurements=False)
print(json.dumps(result,ensure_ascii=False,indent=2))
