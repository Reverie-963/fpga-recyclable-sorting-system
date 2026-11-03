
"""PC-side independent test-set evaluation, optional CPU vs accelerator comparison."""
import argparse,json,time
from pathlib import Path
import cv2,numpy as np
from .inference import Predictor

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--data',default='data/crops/test')
    p.add_argument('--compare');p.add_argument('--output',default='report/model_evaluation.json');a=p.parse_args()
    model=Predictor(json.loads(Path(a.config).read_text(encoding='utf-8-sig')));reference=Predictor(json.loads(Path(a.compare).read_text(encoding='utf-8-sig'))) if a.compare else None
    rows=[]
    try:
        for label in model.labels:
            for path in sorted((Path(a.data)/label).glob('*')):
                im=cv2.imread(str(path))
                if im is None:continue
                t=time.perf_counter();result=model.predict(im);elapsed=(time.perf_counter()-t)*1000
                row={'file':str(path),'actual':label,'predicted':result['label'],'margin':result['margin'],'ms':elapsed}
                if reference:
                    other=reference.predict(im)
                    if other['score_kind']!=result['score_kind']:raise ValueError('Comparison requires same score semantics')
                    row.update(reference_label=other['label'],margin_difference=abs(other['margin']-result['margin']))
                rows.append(row)
        if not rows:raise ValueError('No test images')
        if set(row['actual'] for row in rows)!=set(model.labels):raise ValueError('Each class needs test images')
        times=[row['ms'] for row in rows]
        result={'samples':len(rows),'accuracy':float(np.mean([v['actual']==v['predicted'] for v in rows])),
                'mean_ms':float(np.mean(times)),'p50_ms':float(np.percentile(times,50)),'p95_ms':float(np.percentile(times,95)),'p99_ms':float(np.percentile(times,99)),'max_ms':max(times),
                'model_id':model.model_id,'providers':model.providers,'score_kind':model.metric,'rows':rows,
                'note':'Preprocess+inference time, excludes camera/network. Provider registration does not prove all operators run on accelerator.'}
        if reference:result.update(label_agreement=float(np.mean([v['predicted']==v['reference_label'] for v in rows])),max_margin_difference=max(v['margin_difference'] for v in rows))
        Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    finally:
        for m in [model,reference]:
            if m:
                profile=m.close()
                if profile:print('ORT profile:',profile)
if __name__=='__main__':main()
