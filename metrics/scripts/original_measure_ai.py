"""Bounded GPU image benchmark. No actuator API, service changes or model switch."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).parent
catalog=json.loads((ROOT/'local-only/model-switch-console/models.json').read_text(encoding='utf-8-sig'))
os.environ.update(YOLO_AUTOINSTALL='false',YOLO_CONFIG_DIR=str(OUT/'gpu-settings'),
                  OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
sys.path.insert(0,str(ROOT/'pynq_km1_vision'))
sys.path.insert(0,catalog['runtime'])
import cv2
import numpy as np
import psutil
import torch
import ultralytics
from src.console_candidate_monitor import CandidateDetector

opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))

def state():
    try:
        with opener.open('http://127.0.0.1:8810/operation',timeout=5) as r:d=json.load(r)
        return {k:d.get(k) for k in ['busy','state','blockers','operation_id','board_root','board_source_sha256','execution_model']}
    except Exception as exc:
        return dict(available=False,busy=None,error=type(exc).__name__+': '+str(exc))

def save(d):
    (OUT/'ai-results.json').write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def percentile(v):
    a=np.asarray(v,float)
    return dict(n=len(a),p50_ms=float(np.percentile(a,50)),p95_ms=float(np.percentile(a,95)),
                p99_ms=float(np.percentile(a,99)) if len(a)>=100 else None,max_ms=float(a.max()))

report=dict(schema='competition-gpu-batch1-measurement-v1',started_at=datetime.datetime.now().astimezone().isoformat(),
            physical_motion=False,actuator_requests=0,services_restarted=0,model_switches=0,
            source='stored known-inventory photos, current registered detector and reference bank in separate process',
            warmups_excluded=5,batch=1,background_services_left_running=True,rows=[])
report['before']=state()
assert report['before']['busy'] is not True
entry=next(m for m in catalog['models'] if m['id']=='yoloe26_reference')
with opener.open('http://127.0.0.1:8831/health',timeout=5) as r:worker=json.load(r)
assert entry['sha256']==worker['model_sha256']
assert entry['configuration_sha256']==worker['configuration_sha256']
report['existing_PC_worker_identity']={k:worker.get(k) for k in ['model_sha256','configuration_sha256','runtime_id','execution']}
report['state_probe_errors']=[]
free,total=torch.cuda.mem_get_info()
assert psutil.virtual_memory().available>4*2**30 and free>2**30
report['preflight_memory']=dict(available_ram_bytes=psutil.virtual_memory().available,gpu_free_bytes=free,gpu_total_bytes=total)
previous=json.loads((ROOT/'local-only/competition-audit-20261101/metrics-summary.json').read_text(encoding='utf-8'))
frames=previous['AI_quality']['rows']
assert len(frames)==131
dataset=ROOT/'local-only/recyclables-final-16-v1'
frozen=json.loads((ROOT/'local-only/yoloe-public16-web-20261031/reference-comparison/inputs.json').read_text(encoding='utf-8'))
frozen={str(Path(k)).lower():v for k,v in frozen.items()}
images=[]
for f in frames:
    p=dataset/f['image'];assert hashlib.sha256(p.read_bytes()).hexdigest()==frozen[str(p).lower()]
    image=cv2.imread(str(p));assert image.shape==(480,640,3)
    images.append(image)
detector=CandidateDetector(entry,'cuda:0')
captured={}
original_predict=detector.model.predict

def timed_predict(*args,**kwargs):
    raw=original_predict(*args,**kwargs)
    captured.clear();captured.update({k:float(v) for k,v in raw[0].speed.items()})
    return raw

detector.model.predict=timed_predict
for image in [im for im,f in zip(images,frames) if f['scene']=='source'][:5]:detector.predict_control(image)
report['runtime']=dict(detector.health(),ultralytics=ultralytics.__version__,python=sys.version.split()[0],
                       opencv=cv2.__version__,numpy=np.__version__,torch_file=torch.__file__)
assert report['runtime']['tensor_dtype']=='torch.float16'
outputs=[]
start=time.perf_counter()
try:
    for i in range(2048):
        if i%16==0:
            s=state()
            if s.get('available') is False:report['state_probe_errors'].append(dict(index=i,**s))
            if s['busy'] is True or (s.get('execution_model') and s['execution_model']['configuration_sha256']!=entry['configuration_sha256']):
                report['stopped_reason']='another_operation_started_or_model_binding_changed';break
        image=images[i%len(images)]
        t=time.perf_counter();value=detector.predict_control(image);wall=(time.perf_counter()-t)*1000
        assert value['model_version']==entry['sha256'] and value['configuration_sha256']==entry['configuration_sha256']
        assert value['source_bgr_sha256']==hashlib.sha256(image.tobytes()).hexdigest()
        row=dict(index=i,image=frames[i%len(frames)]['image'],instances=len(value['instances']),
                 preprocess_ms=captured['preprocess'],forward_ms=captured['inference'],
                 model_postprocess_ms=captured['postprocess'],detector_total_ms=wall,
                 application_extra_ms=wall-sum(captured.values()))
        assert row['application_extra_ms']>=-1
        report['rows'].append(row)
        if i<len(frames):outputs.append(dict(image=frames[i]['image'],result=value))
        if (i+1)%25==0:
            print(f'GPU batch1: {i+1} samples, {time.perf_counter()-start:.1f} s',flush=True)
            save(report)
        if i+1>=len(frames) and time.perf_counter()-start>=60:break
        if time.perf_counter()-start>180:
            report['stopped_reason']='bounded_benchmark_duration_reached';break
    report['duration_s']=time.perf_counter()-start
    report['sustained_loop_fps']=len(report['rows'])/report['duration_s']
    report['throughput_scope']='sequential preloaded image detector calls including periodic read-only status checks and loop bookkeeping; excludes image file decode, camera, HTTP, PS planning, PL verification and mechanics'
    report['summary']={k:percentile([r[k] for r in report['rows']]) for k in
                       ['preprocess_ms','forward_ms','model_postprocess_ms','application_extra_ms','detector_total_ms']}
    report['postprocessing_total_summary']=percentile([r['model_postprocess_ms']+r['application_extra_ms'] for r in report['rows']])
    report['after']=state()
    report['finished_at']=datetime.datetime.now().astimezone().isoformat()
    save(report)
    (OUT/'gpu-first-pass-predictions.json').write_text(json.dumps(outputs,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'samples':len(report['rows']),'duration_s':report['duration_s'],'fps':report['sustained_loop_fps'],'summary':report['summary']},ensure_ascii=False),flush=True)
except Exception as exc:
    report['error']=dict(type=type(exc).__name__,message=str(exc));save(report);raise
