"""Bounded image/statistics measurement, no actuator/control writes."""
import json,time,datetime,urllib.request,base64,hashlib,cv2,numpy as np
from pathlib import Path
O=Path(__file__).parent;out=O/'online-benchmark-repeat';out.mkdir(exist_ok=True);assert not (out/'raw.jsonl').exists(),'Existing raw cohort must not be appended or overwritten'
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
rows=[];summary={'timestamp':datetime.datetime.now().astimezone().isoformat(),'scope':'current services running, no new mechanical commands; sequential bounded samples','automatic_retries':0}
def request(url,payload=None):
 t=time.perf_counter();body=None if payload is None else json.dumps(payload).encode()
 with opener.open(urllib.request.Request(url,body,{'Content-Type':'application/json'}),timeout=15) as r:raw=r.read();headers=dict(r.headers)
 return raw,headers,(time.perf_counter()-t)*1000
def stats(v):
 return dict(zip(('p50','p95','p99','max'),map(float,np.percentile(v,[50,95,99,100])))) if v else None
def run(kind,n,func,gap=0):
 start=time.perf_counter();data=[]
 for i in range(n):
  t=time.perf_counter();row=dict(kind=kind,index=i,elapsed_s=t-start)
  try:row.update(ok=True,**func(i))
  except Exception as e:row.update(ok=False,error_type=type(e).__name__,error=str(e))
  data.append(row);rows.append(row)
  with (out/'raw.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
  if gap:time.sleep(max(0,gap-(time.perf_counter()-t)))
  if i%20==0:print(kind,i,flush=True)
 duration=time.perf_counter()-start;good=[r for r in data if r['ok']]
 s=dict(samples=n,success=len(good),failures=n-len(good),duration_s=duration,rtt_ms=stats([r['rtt_ms'] for r in good]))
 if kind=='png':s.update(payload_bytes=sum(r['bytes'] for r in good),payload_mbps=sum(r['bytes'] for r in good)*8/duration/1e6)
 if kind=='pl':
  unique={r['frame_id']:r for r in good if r.get('processing_ms') is not None};s.update(unique_frames=len(unique),processing_ms=stats([r['processing_ms'] for r in unique.values()]))
  if len(unique)>1:s['observed_frame_id_advance_per_s']=(max(unique)-min(unique))/(good[-1]['elapsed_s']-good[0]['elapsed_s'])
 if kind=='capture_infer':s['service_predict_ms']=stats([r['inference_ms'] for r in good]);s['capture_upload_response_ms']=stats([r['total_ms'] for r in good])
 summary[kind]=s;(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
def control(i):
 raw,h,t=request('http://127.0.0.1:8810/operation');v=json.loads(raw)
 assert not v['busy'] and not v['blockers'],'no_motion_cohort_requires_idle_unblocked_controller'
 assert v['board_root']=='/opt/km1/runtime/service-ready-20261101-v3'
 assert v['execution_model']['profile_id']=='yoloe26_reference'
 return dict(rtt_ms=t,state=v['state'],busy=v['busy'],blockers=v['blockers'],board_root=v['board_root'],profile=v['execution_model']['profile_id'])
def png(i):
 raw,h,t=request('http://127.0.0.1:8780/snapshot.png');img=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR);assert img.shape==(480,640,3)
 h={k.lower():v for k,v in h.items()};digest=hashlib.sha256(img.tobytes()).hexdigest();expected=h.get('x-decoded-bgr-sha256');assert expected is None or digest==expected
 return dict(rtt_ms=t,bytes=len(raw),bgr_sha256=digest,header_hash_verified=expected==digest,frame_id=h.get('x-frame-id'),stream_id=h.get('x-stream-id'))
def pl(i):
 raw,h,t=request('http://127.0.0.1:8780/vision');v=json.loads(raw);e=v.get('evidence') or {}
 assert v['available'] is True and v['backend']=='pl_board_local','actual_pl_service_unavailable'
 assert v['assets']=={'station_vision.bit':'054a766d3468282cbfb355dc803b8b9e960749ffde0e80bf0d43e17adebab779','station_vision.hwh':'b40312875bb53fadec4563109a8508176c75bcf582ce42150f13692e375a5213'}
 assert isinstance(e.get('processing_ms'),(int,float)) and 0<=e['processing_ms']<10000
 return dict(rtt_ms=t,frame_id=v['frame_id'],stream_id=v['stream_id'],available=v.get('available'),processing_ms=e.get('processing_ms'),backend=v.get('backend'),assets=v.get('assets'))
def infer(i):
 started=time.perf_counter();raw,h,capture=request('http://127.0.0.1:8780/snapshot.png');img=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR);digest=hashlib.sha256(img.tobytes()).hexdigest()
 raw2,h2,rtt=request('http://127.0.0.1:8831/detect',dict(image_base64=base64.b64encode(raw).decode(),control_only=True));v=json.loads(raw2)
 assert v['source_bgr_sha256']==digest and v['model_version']=='9d5734b34871aa81412b2fa47176d8f926283bb5adfe3a3f80cc58c5b7332534'
 assert v['configuration_sha256']=='cb288271841ff81c2dcd44ea6fc9742231e658d0a68f44caa0a970aee15b792e'
 if i==0:(out/'sample-prediction.json').write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
 return dict(rtt_ms=rtt,capture_ms=capture,total_ms=(time.perf_counter()-started)*1000,inference_ms=v['inference_ms'],instances=len(v['instances']),input_bgr_sha256=digest,model=v['model_version'],configuration=v['configuration_sha256'],profile=v['profile_id'])
run('control',200,control,.1);run('png',200,png,.1);run('pl',200,pl,.25)
for i in range(5):infer(-i-1)
run('capture_infer',200,infer,.1)
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
