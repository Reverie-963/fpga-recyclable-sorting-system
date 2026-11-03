"""Board entry: local/remote inference, fresh-frame recheck, one explicit pick."""
import argparse,json,time,uuid
from pathlib import Path
import cv2
from .arm import plan_pick,execute
from .config import validate
from .vision import mask_cpu,open_mask,StableTarget
from .inference import Predictor
from .remote import RemotePredictor
from .camera import LatestCamera
from .telemetry import Recorder
from .control import objects_in_area,same_target,claim_task,verify_outcome

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='config/baseline_local.json');p.add_argument('--execute',action='store_true')
    p.add_argument('--once',action='store_true');p.add_argument('--frames',type=int,default=300);a=p.parse_args()
    c=validate(json.loads(Path(a.config).read_text(encoding='utf-8-sig')));v=c['vision'];ic=c['inference'];cal=None;predictor=None
    if a.execute and not a.once:p.error('Execution requires --once; no unattended retries')
    if a.frames<1:p.error('--frames must be positive')
    if Path(c['calibration']).exists():cal=json.loads(Path(c['calibration']).read_text(encoding='utf-8-sig'))
    if ic['mode']=='remote':predictor=RemotePredictor(ic)
    elif Path(c['model']).exists():predictor=Predictor({'kind':'ridge','model':c['model']})
    if a.execute and (cal is None or predictor is None or not c['arm']['commissioned']):p.error('Need calibration, model/server and commissioned=true')
    pl=None;cam=None;rec=None;last=0;session=uuid.uuid4().hex;lock=StableTarget(v['stable_frames'],v['stable_px'])
    def segment(frame):
        if frame.shape[:2]!=(c['camera']['height'],c['camera']['width']):raise ValueError('Camera resolution mismatch')
        if cal and cal['image_size']!=[frame.shape[1],frame.shape[0]]:raise ValueError('Calibration resolution mismatch')
        if pl:return pl.run(frame,v['background_bgr'],v['threshold'])
        mask=mask_cpu(frame,v['background_bgr'],v['threshold'])
        return open_mask(mask) if v.get('pl_kernel','threshold')=='open' else mask
    try:
        if v['backend']=='pl':
            from .pl import Segmenter
            pl=Segmenter(c['bitfile'],(c['camera']['height'],c['camera']['width']),kernel=v.get('pl_kernel','threshold'))
        cam=LatestCamera(c['camera']);rec=Recorder('board',**c['telemetry'])
        rec.event({'event':'start','session':session,'inference':ic['mode'],'segmentation':v['backend']},True)
        for _ in range(a.frames):
            fid,stamp,frame=cam.read(after=last);last=fid;t=time.monotonic()
            mask=segment(frame);seg_ms=(time.monotonic()-t)*1000;objs=objects_in_area(mask,v,cal);target=None;prediction=None
            try:
                if len(objs)==1 and predictor:
                    obj=objs[0];x,y,w,h=obj['bbox'];crop=frame[y:y+h,x:x+w]
                    prediction=predictor.predict(crop,fid,stamp) if ic['mode']=='remote' else predictor.predict(crop)
                    if prediction['score_kind']!=ic['score_kind']:raise ValueError('Score kind mismatch; set matching threshold')
                    if prediction['margin']>=v['min_margin'] and prediction['label'] in c['arm']['bins']:
                        target={**obj,**prediction}
                if time.monotonic()-stamp>ic['max_result_age_s']:raise TimeoutError('Stale result')
            except Exception as e:
                lock.reset();rec.event({'event':'inference_rejected','frame_id':fid,'reason':type(e).__name__+': '+str(e)},True)
                # No automatic local inference fallback. Explicit restart/config change is required.
                raise
            ready=lock.update(target)
            record={'event':'observation','time':time.time(),'frame_id':fid,'segmentation':v['backend'],'segment_ms':seg_ms,
                    'vision_ms':(time.monotonic()-t)*1000,'frame_age_ms':(time.monotonic()-stamp)*1000,'objects':objs,'prediction':prediction,'ready':ready}
            rec.preview(frame,mask);rec.event(record)
            if ready and cal:
                from .vision import map_point
                xy=map_point(target['center'],cal);plan=plan_pick(*xy,target['label'],c);task_id=session+'-'+str(fid)
                record.update(task_id=task_id,xy_mm=xy,plan=execute(plan,c,False))
                Path('board/last_plan.json').write_text(json.dumps(record,indent=2));rec.event({**record,'event':'plan'},True)
                print(json.dumps({'task_id':task_id,'xy_mm':xy,'label':target['label'],'execute':a.execute}),flush=True)
                if a.execute:
                    def guard():
                        # Request a frame acquired AFTER the recheck began; never accept buffered inference input.
                        started=time.monotonic();check_id=last
                        for _ in range(60):
                            nf,ns,nframe=cam.read(after=check_id);check_id=nf
                            if ns>=started:break
                        else:raise TimeoutError('No post-decision frame')
                        current=objects_in_area(segment(nframe),v,cal)
                        if time.monotonic()-ns>ic['max_result_age_s'] or len(current)!=1 or not same_target(target,frame,current[0],nframe,v['stable_px']):
                            raise RuntimeError('Target moved/disappeared/occluded before motion')
                    guard();claim_task('board/tasks',task_id,{'label':target['label'],'xy_mm':xy})
                    started=time.monotonic()
                    try:result=execute(plan,c,True,before_motion=guard)
                    except Exception as e:
                        rec.event({'event':'action_failed','task_id':task_id,'error':str(e)},True);raise
                    rec.event({'event':'action_sent','task_id':task_id,'command_sequence_s':time.monotonic()-started,**result},True)
                    finished=time.monotonic();check_id=last
                    for _ in range(60):
                        nf,ns,after=cam.read(after=check_id);check_id=nf
                        if ns>=finished:break
                    else:raise TimeoutError('No post-action frame')
                    evidence=verify_outcome(frame,after,target,target['label'],c)
                    cv2.imwrite('board/after.jpg',after);rec.event({'event':'verification','task_id':task_id,**evidence},True)
                    print(json.dumps(evidence),flush=True)
                if a.once:break
                lock.reset()
    except Exception as e:
        if rec:rec.event({'event':'stopped','error':type(e).__name__+': '+str(e)},True)
        raise
    finally:
        if cam:cam.close()
        if rec:rec.close()
        if pl:pl.close()
        if isinstance(predictor,Predictor):predictor.close()
if __name__=='__main__':main()
