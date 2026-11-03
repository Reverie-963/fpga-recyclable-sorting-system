
"""Summarize the latest run in a board JSONL event log; never call read-completion time exposure time."""
import argparse,json
from pathlib import Path
import numpy as np

def summarize(records):
    starts=[i for i,r in enumerate(records) if r.get('event')=='start']
    if starts:records=records[starts[-1]:]
    observations=[r for r in records if r.get('event') in ('observation','tracking','semantic_result')]
    result={'observations':len(observations),'session':records[0].get('session') if records else None}
    def stats(values):return {'mean':float(np.mean(values)),'p50':float(np.percentile(values,50)),'p95':float(np.percentile(values,95)),'p99':float(np.percentile(values,99)),'max':float(max(values))}
    for field in ['segment_ms','geometry_ms','vision_ms','frame_age_ms']:
        vals=[r[field] for r in observations if field in r]
        if vals:result[field]=stats(vals)
    for field in ['rpc_ms','inference_ms','request_bytes']:
        vals=[r['prediction'][field] for r in observations if r.get('prediction') and field in r['prediction']]
        if vals:result[field]=stats(vals)
    for field in ['preprocess_ms','inference_ms','postprocess_ms','total_ms']:
        vals=[r['prediction']['timing'][field] for r in observations if field in r.get('prediction',{}).get('timing',{})]
        if vals:result['yolo_'+field]=stats(vals)
    result['events']={e:sum(r.get('event')==e for r in records) for e in ['plan','intercept_plan','track_lost','action_sent','action_failed','inference_rejected','verification','stopped','task_completed','batch_finished']}
    result['note']='Latest session only. Times are host software measurements, not sensor exposure or hardware worst-case bounds.'
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--log',default='board/events.jsonl');p.add_argument('--output',default='report/session_summary.json');a=p.parse_args()
    records=[]
    for line in Path(a.log).read_text(encoding='utf-8-sig').splitlines():
        try:records.append(json.loads(line))
        except json.JSONDecodeError:raise ValueError('Truncated/corrupt log; preserve the original before repair')
    result=summarize(records);Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
