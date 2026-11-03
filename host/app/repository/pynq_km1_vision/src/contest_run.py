"""Competition controller: YOLO11 + PL geometry + bounded verified multi-pick."""
import argparse
import json
import math
import time
import uuid
from pathlib import Path

from .config import validate
from .camera import LatestCamera
from .telemetry import Recorder
from .vision import mask_cpu, open_mask, map_point, StableTarget
from .arm import plan_pick, execute
from .control import claim_task
from .detect_remote import DetectionClient
from .grasp_geometry import fuse, match_target, verify_masks


def validate_contest(c):
    validate(c)
    if c['inference']['mode'] != 'remote' or c['inference']['score_kind'] != 'detection_confidence':
        raise ValueError('Contest uses remote protocol-v2 detection')
    if c['vision'].get('pl_kernel') != 'open': raise ValueError('Contest requires opening kernel')
    g = c['grasp']
    for k in ('confidence', 'min_box_coverage'):
        if not isinstance(g[k], (float,int)) or not math.isfinite(g[k]) or not 0 < g[k] <= 1: raise ValueError('Invalid '+k)
    for k in ('clearance_px', 'min_interior_px'):
        if type(g[k]) is not int or not 1 <= g[k] <= 500: raise ValueError('Invalid '+k)
    return c


def run(c, execute_enabled=False, max_tasks=20, frames=3000):
    validate_contest(c)
    if type(max_tasks) is not int or not 1 <= max_tasks <= 1000 or frames < 1: raise ValueError('Invalid run limit')
    cal = json.loads(Path(c['calibration']).read_text(encoding='utf-8-sig'))
    if execute_enabled:
        if not c['arm']['commissioned']: raise ValueError('Commission arm first')
        if not set(c['arm']['bins']).issubset(c['verification']['bin_rois']):
            raise ValueError('Every class needs measured bin ROI before execution')
        # A deposit zone must not be reused as a picking zone.
        import cv2
        import numpy as np
        hull = np.asarray(cal['pixel_hull'],np.float32)
        for x,y,w,h in c['verification']['bin_rois'].values():
            roi = np.asarray([[x,y],[x+w,y],[x+w,y+h],[x,y+h]],np.float32)
            if cv2.intersectConvexConvex(hull,roi)[0] > 0: raise ValueError('Bin ROI overlaps pick region')
    client = DetectionClient(c['inference']); cam = None; pl = None; rec = None
    session = uuid.uuid4().hex; last = 0; completed = 0; scanned = 0
    v = c['vision']; lock = StableTarget(v['stable_frames'], v['stable_px'])
    def fresh():
        nonlocal last
        started = time.monotonic()
        for _ in range(60):
            fid,stamp,frame = cam.read(after=last); last = fid
            if stamp >= started:
                if frame.shape[:2] != (c['camera']['height'],c['camera']['width']): raise ValueError('Frame size changed')
                if cal['image_size'] != [frame.shape[1],frame.shape[0]]: raise ValueError('Calibration size mismatch')
                return fid,stamp,frame
        raise TimeoutError('No fresh frame')
    def segment(frame):
        return pl.run(frame,v['background_bgr'],v['threshold']) if pl else open_mask(mask_cpu(frame,v['background_bgr'],v['threshold']))
    def observe():
        fid,stamp,frame = fresh(); t = time.monotonic(); mask = segment(frame)
        seg_ms = (time.monotonic()-t)*1000; prediction = client.predict(frame,fid,stamp)
        choices = fuse(prediction['detections'],mask,c,cal)
        if time.monotonic()-stamp > c['inference']['max_result_age_s']: raise TimeoutError('Stale geometry')
        rec.preview(frame,mask)
        rec.event({'event':'observation','frame_id':fid,'detections':prediction['detections'],
            'candidates':choices,'segment_ms':seg_ms,'frame_age_ms':(time.monotonic()-stamp)*1000,
            'vision_ms':(time.monotonic()-t)*1000,'prediction':prediction})
        return fid,frame,mask,choices
    try:
        if v['backend']=='pl':
            from .pl import Segmenter
            pl = Segmenter(c['bitfile'],(c['camera']['height'],c['camera']['width']),kernel='open')
        cam = LatestCamera(c['camera']); rec = Recorder('board',**c['telemetry'])
        rec.event({'event':'start','session':session,'controller':'contest_v4','config':c,
                   'max_tasks':max_tasks,'execute':execute_enabled},True)
        while scanned < frames and completed < max_tasks:
            fid,frame,mask,choices = observe(); scanned += 1
            target = choices[0] if choices else None
            if not lock.update(target): continue
            xy = map_point(target['center'],cal); plan = plan_pick(*xy,target['label'],c)
            task_id = session+'-'+str(fid)
            record = {'event':'plan','task_id':task_id,'target':target,'xy_mm':xy,'plan':execute(plan,c,False)}
            Path('board/last_plan.json').write_text(json.dumps(record,indent=2),encoding='utf-8');rec.event(record,True)
            if not execute_enabled:
                return {'status':'dry_run_plan','planned':1,'completed':0}
            def guard():
                _,current_frame,_,current_choices = observe()
                if match_target(target,frame,current_choices,current_frame,v['stable_px']) is None:
                    raise RuntimeError('Target identity/geometry changed before motion')
            guard(); claim_task('board/tasks',task_id,{'label':target['label'],'xy_mm':xy})
            rec.event({'event':'action_started','task_id':task_id},True)
            result = execute(plan,c,True,before_motion=guard)
            rec.event({'event':'action_sent','task_id':task_id,**result},True)
            _,_,after = fresh(); after_mask = segment(after)
            evidence = verify_masks(mask,after_mask,target,target['label'],c)
            import cv2
            cv2.imwrite('board/after.jpg',after)
            evidence_dir=Path('board/evidence')/task_id; evidence_dir.mkdir(parents=True)
            cv2.imwrite(str(evidence_dir/'before.png'),frame);cv2.imwrite(str(evidence_dir/'after.png'),after)
            rec.event({'event':'verification','task_id':task_id,**evidence},True)
            if evidence['status'] != 'visual_evidence_pass':
                raise RuntimeError('Unverified placement; stop batch for human review, no retry')
            completed += 1; lock.reset()
            rec.event({'event':'task_completed','task_id':task_id,'completed':completed},True)
        result = {'status':'task_limit' if completed>=max_tasks else 'observation_limit',
                  'completed':completed,'observations':scanned}
        rec.event({'event':'batch_finished',**result},True)
        return result
    except Exception as exc:
        if rec: rec.event({'event':'stopped','reason':str(exc),'completed':completed},True)
        raise
    finally:
        if cam: cam.close()
        if pl: pl.close()
        if rec: rec.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='config/static_board.json')
    p.add_argument('--execute',action='store_true');p.add_argument('--max-tasks',type=int,default=20)
    p.add_argument('--frames',type=int,default=3000);a=p.parse_args()
    print(json.dumps(run(json.loads(Path(a.config).read_text(encoding='utf-8-sig')),a.execute,a.max_tasks,a.frames)))

if __name__=='__main__':main()
