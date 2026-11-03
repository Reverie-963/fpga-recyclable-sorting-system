"""Continuous conveyor interception; asynchronous semantics, encoder-linked geometry.
A low-speed calibrated planar conveyor is required. No stop-and-pick fallback.
"""
import argparse,copy,json,math,time,uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import numpy as np
from .contest_run import validate_contest
from .camera import LatestCamera
from .telemetry import Recorder
from .vision import candidates,mask_cpu,open_mask
from .grasp_geometry import fuse
from .detect_remote import DetectionClient
from .conveyor import Encoder,at_time
from .moving_targets import Tracker,intercept,verify_deposit
from .arm import plan_pick,packet
from .arm_transport import ArmConnection
from .arm_plan import inspect_plan,require_plan
from .control import claim_task


def validate_dynamic(c,enabled=False):
    validate_contest(c);d=c['dynamic'];e=c['encoder']
    if enabled and c['arm'].get('transport','direct') != 'direct':
        raise ValueError('OpenMV V3 currently supports static commissioning only; dynamic bridge timing is unverified')
    if d['mode']!='continuous_intercept':raise ValueError('Only continuous_intercept is supported')
    positive=('track_max_gap_s','association_mm','max_track_age_s','semantic_max_age_s','semantic_coverage',
              'min_speed_mm_s','max_speed_mm_s','max_error_mm','max_accel_mm_s2','position_uncertainty_mm',
              'actuation_uncertainty_s','capture_uncertainty_s','max_frame_age_s','grip_contact_s','schedule_slack_s')
    for k in positive:
        if isinstance(d[k],bool) or not isinstance(d[k],(int,float)) or not math.isfinite(d[k]) or d[k]<=0:raise ValueError('Invalid dynamic '+k)
    if not isinstance(d['capture_delay_s'],(int,float)) or not math.isfinite(d['capture_delay_s']) or d['capture_delay_s']<0:raise ValueError('Invalid camera delay')
    if not isinstance(d['min_track_frames'],int) or d['min_track_frames']<2 or not 0<d['semantic_coverage']<=1:raise ValueError('Invalid tracking thresholds')
    if len(d['axis_xy'])!=2 or not np.isfinite(d['axis_xy']).all() or abs(np.linalg.norm(d['axis_xy'])-1)>1e-6:raise ValueError('Belt axis must be a unit vector')
    if len(d['pick_window_mm'])!=4 or not np.isfinite(d['pick_window_mm']).all():raise ValueError('Invalid pick window')
    x0,x1,y0,y1=d['pick_window_mm'];a,b,f,g=c['arm']['pick_bounds_mm']
    if not a<=x0<x1<=b or not f<=y0<y1<=g:raise ValueError('Window outside arm bounds')
    if d['min_speed_mm_s']>=d['max_speed_mm_s']:raise ValueError('Invalid speed range')
    if type(d['grip_ms']) is not int or not 300<=d['grip_ms']<=5000 or d['grip_contact_s']>d['grip_ms']/1000:raise ValueError('Measure jaw contact time within grip duration')
    for k in ('mm_per_count','clock_hz','snapshot_max_s','velocity_window_s','min_velocity_span_s'):
        if not isinstance(e[k],(int,float)) or not math.isfinite(e[k]) or e[k]<=0:raise ValueError('Invalid encoder '+k)
    if e['direction'] not in (-1,1) or e['clock_hz']!=100000000 or e['min_velocity_span_s']>=e['velocity_window_s']:raise ValueError('Invalid encoder scale/window')
    if enabled and (not c['arm']['commissioned'] or not d['commissioned']):raise ValueError('Commission dynamic timing, camera delay and arm before execution')
    if enabled and not set(c['arm']['bins']).issubset(c['verification']['bin_rois']):raise ValueError('Measure every bin ROI')
    return c


def geometry(mask,c,cal):
    # Extract label-free interior geometry; semantics is bound later using the original frame IDs.
    raw=candidates(mask,c['vision']['min_area'],c['vision']['max_area'],already_open=True)
    placeholder=next(iter(c['arm']['bins']))
    ds=[{'label':placeholder,'confidence':1.,'xyxy':[x,y,x+w,y+h]} for x,y,w,h in (o['bbox'] for o in raw)]
    return fuse(ds,mask,c,cal)


class MovingArm:
    """Persistent UART; preflight every point, poll tracking between writes."""
    def __init__(self,c):
        if c['arm'].get('transport','direct') != 'direct':
            raise ValueError('Dynamic OpenMV timing not commissioned; use static arm_pick first')
        self.c=c;self.connection=ArmConnection(c['arm'])
        self.connection.open()
    def send(self,step,poll=lambda:None,guard=lambda:None):
        _,data,duration=step
        start=self.connection.send(data,duration,guard)
        end=start+duration/1000+self.c['arm']['settle_s']
        while time.monotonic()<end:poll();time.sleep(.001)
        return start
    def prepare(self):
        a=self.c['arm'];label=next(iter(a['bins']))
        # Use IK checked preparation steps; no movement to the placeholder pick location.
        plan=plan_pick(0 if a['pick_bounds_mm'][0]<=0<=a['pick_bounds_mm'][1] else a['pick_bounds_mm'][0],
                       (a['pick_bounds_mm'][2]+a['pick_bounds_mm'][3])/2,label,self.c)
        require_plan(plan[:3],self.c,dynamic=True)
        for step in plan[:3]:self.send(step)
    def close(self):self.connection.close()


def moving_plan(choice,c):
    plan=plan_pick(*choice['xy'],choice['label'],c)[3:]
    return [(name,packet([(5,c['arm']['close_pwm'])],c['dynamic']['grip_ms']),c['dynamic']['grip_ms']) if name=='close' else (name,data,dur) for name,data,dur in plan]


def run(c,enabled=False,max_tasks=20,frames=3000):
    validate_dynamic(c,enabled)
    if type(max_tasks) is not int or not 1<=max_tasks<=1000 or type(frames) is not int or frames<1:raise ValueError('Invalid batch limits')
    cal=json.loads(Path(c['calibration']).read_text(encoding='utf-8-sig'))
    for x,y,w,h in c['verification']['bin_rois'].values():
        if cv2.intersectConvexConvex(np.asarray(cal['pixel_hull'],np.float32),np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]],np.float32))[0]>0:raise ValueError('Bin ROI overlaps belt observation area')
    v=c['vision'];d=c['dynamic'];cam=pl=rec=arm=None;future=None;pool=None
    tracker=Tracker(d);last=0;completed=0;scanned=0;session=uuid.uuid4().hex;latest=None;pending=None
    # Even CPU image baseline uses the same PL encoder: compare the image path without changing the clock source.
    try:
        from .pl import Segmenter
        pl=Segmenter(c['bitfile'],(c['camera']['height'],c['camera']['width']),kernel='open')
        encoder=Encoder(pl.overlay,c['encoder']);client=DetectionClient(c['inference'])
        cam=LatestCamera(c['camera']);rec=Recorder('board',**c['telemetry']);pool=ThreadPoolExecutor(max_workers=1)
        if enabled:arm=MovingArm(c);arm.prepare()
        rec.event({'event':'start','controller':'recycling_v5','config':c,'execute':enabled,'session':session},True)
        def pump():
            nonlocal last,future,pending,latest,scanned
            fid,read_stamp,frame=cam.read(after=last);last=fid;scanned+=1
            if frame.shape[:2]!=(c['camera']['height'],c['camera']['width']) or cal['image_size']!=[frame.shape[1],frame.shape[0]]:raise ValueError('Camera/calibration mismatch')
            stamp=read_stamp-d['capture_delay_s'];sample=encoder.sample()
            if sample['time']-stamp>d['max_frame_age_s']:raise TimeoutError('Frame too old for conveyor tracking')
            speed=sample['velocity_mm_s']
            if speed is not None and not d['min_speed_mm_s']<=speed<=d['max_speed_mm_s']:raise RuntimeError('Belt stopped/reversed/outside commissioned speed range')
            tick=time.monotonic()
            mask=pl.run(frame,v['background_bgr'],v['threshold']) if v['backend']=='pl' else open_mask(mask_cpu(frame,v['background_bgr'],v['threshold']))
            geom=geometry(mask,c,cal)
            old_ids=set(tracker.tracks)
            tracks=tracker.update(geom,stamp,at_time(sample,stamp),cal) if speed is not None else {}
            for lost in old_ids-set(tracks):
                rec.event({'event':'track_lost','track_id':lost,'reason':'left_or_ambiguous','frame_id':fid})
            if future is not None and future.done():
                prediction=future.result() # authentication, fingerprint and timeout errors stop the batch
                tracker.bind(pending[1],prediction['detections'],pending[0])
                rec.event({'event':'semantic_result','frame_id':pending[2],'prediction':prediction})
                future=None
            if future is None and tracks:
                pending=(stamp,copy.deepcopy(tracks),fid)
                future=pool.submit(client.predict,frame.copy(),fid,read_stamp)
            rec.preview(frame,mask)
            latest={'fid':fid,'stamp':stamp,'sample':sample,'frame':frame,'mask':mask,'tracks':copy.deepcopy(tracker.tracks)}
            rec.event({'event':'tracking','frame_id':fid,'encoder':sample,'tracks':latest['tracks'],
                       'geometry_ms':(time.monotonic()-tick)*1000,'frame_age_ms':(time.monotonic()-stamp)*1000})
            return latest
        while scanned<frames and completed<max_tasks:
            obs=pump();now=time.monotonic()
            horizon=2*(c['arm']['move_ms']/1000+c['arm']['settle_s'])+d['grip_contact_s']
            options=[intercept(t,obs['sample'],now,horizon,c) for t in tracker.tracks.values()]
            options=[o for o in options if o]
            if not options:continue
            choice=min(options,key=lambda o:o['track_id']);plan=moving_plan(choice,c);task_id=session+'-'+str(choice['track_id'])
            validation=inspect_plan(plan,c,dynamic=True)
            record={'event':'intercept_plan','task_id':task_id,**choice,'plan':[{'step':n,'command':p.decode(),'wait_ms':ms} for n,p,ms in plan],
                    'plan_validation':validation}
            Path('board/last_plan.json').write_text(json.dumps(record,indent=2));rec.event(record,True)
            if not enabled:return {'status':'dry_run_intercept','planned':1,'completed':0}
            require_plan(plan,c,dynamic=True)
            claim_task('board/tasks',task_id,choice)
            # Rebase deadline immediately before first command; reject planning/I/O jitter instead of stretching it.
            if time.monotonic()-now>d['schedule_slack_s']:raise TimeoutError('Commit deadline missed')
            before=obs['mask'].copy();before_frame=obs['frame'].copy()
            def belt_guard():
                sample=encoder.sample();speed=sample['velocity_mm_s'];remaining=choice['contact_time']-time.monotonic()
                if speed is None or not d['min_speed_mm_s']<=speed<=d['max_speed_mm_s']:raise RuntimeError('Invalid belt speed during interception')
                predicted=at_time(sample,choice['contact_time'])
                if abs(predicted-choice['belt_at_contact'])>d['max_error_mm']-choice['error_bound_mm']:raise RuntimeError('Belt prediction drift; abort remaining commands')
                if remaining < -d['schedule_slack_s']:raise TimeoutError('Grasp contact deadline missed')
            def target_guard():
                belt_guard();current=tracker.tracks.get(choice['track_id'])
                if current is None:raise RuntimeError('Target identity lost before descent')
                if current['label']!=choice['label']:raise RuntimeError('Target class changed')
                remaining=choice['contact_time']-time.monotonic()
                projected=intercept(current,latest['sample'],time.monotonic(),max(0,remaining),c)
                if projected is None or np.linalg.norm(np.array(projected['xy'])-choice['xy'])>d['max_error_mm']-choice['error_bound_mm']:raise RuntimeError('Target slipped/ambiguous before descent')
            expected=now
            for name,data,ms in plan:
                if name in ('above_target','descend','close'):
                    if time.monotonic()-expected>d['schedule_slack_s']:raise TimeoutError('Motion step late')
                    guard=target_guard if name!='close' else belt_guard
                else:guard=lambda:None
                # Vision continues until descent; arm occlusion then makes geometry invalid.
                poll=pump if name=='above_target' else (belt_guard if name in ('descend','close') else lambda:None)
                # Contact deadline applies until close command only; during jaw closure use encoder speed, not expired deadline.
                if name not in ('above_target','descend'):
                    def poll():
                        s=encoder.sample();speed=s['velocity_mm_s']
                        if speed is None or not d['min_speed_mm_s']<=speed<=d['max_speed_mm_s']:raise RuntimeError('Belt speed changed during motion')
                sent_at=arm.send((name,data,ms),poll,guard)
                rec.event({'event':'motion_step','task_id':task_id,'step':name,'expected_start':expected,'sent_at':sent_at,'duration_ms':ms},True)
                expected+=ms/1000+c['arm']['settle_s']
            # Reobserve after arm returns; reset IDs to avoid carrying semantics across occlusion.
            tracker.tracks={}
            after=pump();evidence=verify_deposit(before,after['mask'],choice['label'],c)
            dest=Path('board/evidence')/task_id;dest.mkdir(parents=True)
            cv2.imwrite(str(dest/'before.png'),before_frame);cv2.imwrite(str(dest/'after.png'),after['frame'])
            rec.event({'event':'verification','task_id':task_id,**evidence},True)
            if evidence['status']!='bin_visual_evidence_pass':raise RuntimeError('Deposit unverified; no automatic retry')
            completed+=1;rec.event({'event':'task_completed','task_id':task_id,'completed':completed},True)
        result={'status':'task_limit' if completed>=max_tasks else 'observation_limit','completed':completed,'observations':scanned}
        rec.event({'event':'batch_finished',**result},True);return result
    except Exception as exc:
        if rec:rec.event({'event':'stopped','reason':str(exc),'completed':completed},True)
        raise
    finally:
        if arm:arm.close()
        if pool:pool.shutdown(wait=True)
        if cam:cam.close()
        if pl:pl.close()
        if rec:rec.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='config.json')
    parser.add_argument('--execute',action='store_true');parser.add_argument('--max-tasks',type=int,default=20);parser.add_argument('--frames',type=int,default=3000)
    a=parser.parse_args();print(json.dumps(run(json.loads(Path(a.config).read_text(encoding='utf-8-sig')),a.execute,a.max_tasks,a.frames)))
if __name__=='__main__':main()
