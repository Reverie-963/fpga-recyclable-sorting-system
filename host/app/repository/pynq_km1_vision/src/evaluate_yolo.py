"""Independent detection quality and batch=1 phase timings on real test images."""
import argparse,json,time
from pathlib import Path
import numpy as np
from .detection import YOLODetector

def main():
    from ultralytics import YOLO
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--data',default='data/dataset.yaml')
    p.add_argument('--images',default='data/detection/images/test');p.add_argument('--repeat',type=int,default=5)
    p.add_argument('--output',default='metrics/logs/yolo_evaluation.json');a=p.parse_args()
    if a.repeat<1:p.error('repeat must be positive')
    cfg=json.loads(Path(a.config).read_text(encoding='utf-8-sig'));detector=YOLODetector(cfg)
    metrics=detector.model.val(data=a.data,split='test',device=detector.device,imgsz=cfg['imgsz'],batch=1,
        conf=.001,iou=.7,half=False,plots=True,project='runs/evaluate',name='test',exist_ok=False)
    import cv2
    paths=sorted(p for p in Path(a.images).rglob('*') if p.suffix.lower() in ('.jpg','.jpeg','.png'))
    if not paths:raise ValueError('No test images')
    rows=[];started=time.perf_counter()
    for _ in range(a.repeat):
        for path in paths:
            frame=cv2.imread(str(path))
            if frame is None:raise ValueError('Unreadable '+str(path))
            result=detector.predict(frame);rows.append({'file':str(path),**result})
    elapsed=time.perf_counter()-started
    summary={}
    for field in rows[0]['timing']:
        values=[r['timing'][field] for r in rows]
        summary[field]={'mean':float(np.mean(values)),**{'p'+str(q):float(np.percentile(values,q)) for q in [50,95,99]},'max':max(values)}
    report={'model_id':detector.model_id,'runtime_id':detector.runtime_id,'config':cfg,
        'test_quality':{k:float(v) for k,v in metrics.results_dict.items()},'phase_timings_ms':summary,
        'sustained_samples_s_including_file_read':len(rows)/elapsed,'samples':len(rows),'duration_s':elapsed,'rows':rows}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
