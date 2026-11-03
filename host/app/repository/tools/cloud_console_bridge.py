"""Outbound-only bridge for the private cloud console. No autonomous motion.

Snapshots omit local authorization tokens. Each claimed command is submitted
at most once to the original loopback API, whose checks remain authoritative.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args):return None


def clean(value):
    private={'token','control_token','vision_token','device_secret','service_token','password','authorization','secret','private_key','board_source_sha256','board_root','config_path','weights','binding'}
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items() if k not in private}
    if isinstance(value,list):return [clean(v) for v in value]
    return value


def protect(raw,decrypt=False):
    if os.name!='nt':return raw
    from ctypes import wintypes
    class Blob(ctypes.Structure):_fields_=[('size',wintypes.DWORD),('data',ctypes.POINTER(ctypes.c_char))]
    buf=ctypes.create_string_buffer(raw);source=Blob(len(raw),ctypes.cast(buf,ctypes.POINTER(ctypes.c_char)));result=Blob()
    if decrypt:ok=ctypes.windll.crypt32.CryptUnprotectData(ctypes.byref(source),None,None,None,None,1,ctypes.byref(result))
    else:ok=ctypes.windll.crypt32.CryptProtectData(ctypes.byref(source),ctypes.c_wchar_p('Reverie cloud device bridge'),None,None,None,1,ctypes.byref(result))
    if not ok:raise OSError('device_configuration_protection_failed')
    try:return ctypes.string_at(result.data,result.size)
    finally:
        ctypes.windll.kernel32.LocalFree.argtypes=[ctypes.c_void_p]
        ctypes.windll.kernel32.LocalFree.restype=ctypes.c_void_p
        ctypes.windll.kernel32.LocalFree(result.data)


def save_config(path,value):
    path=Path(path).resolve();path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(protect(json.dumps(value).encode()));os.chmod(path,0o600)


def load_config(path):return json.loads(protect(Path(path).read_bytes(),True))


class Bridge:
    def __init__(self,config):
        self.config=config;self.local=config.get('local_url','http://127.0.0.1:8810').rstrip('/')
        parsed=urllib.parse.urlsplit(self.local)
        if parsed.scheme!='http' or parsed.hostname not in ('127.0.0.1','localhost') or parsed.path or parsed.query or parsed.fragment or parsed.username:raise ValueError('loopback_api_required')
        self.cloud=config['site_url'].rstrip('/');parsed=urllib.parse.urlsplit(self.cloud)
        if parsed.path or parsed.query or parsed.fragment or parsed.username or not (parsed.scheme=='https' and parsed.hostname.endswith('.chatgpt.site') or parsed.scheme=='http' and parsed.hostname in ('localhost','127.0.0.1')):raise ValueError('selected_site_origin_required')
        self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
        # Loopback remains direct; the hosted Site uses the workstation's
        # existing system proxy, which may be required for outbound HTTPS.
        self.cloud_opener=urllib.request.build_opener(urllib.request.ProxyHandler(),NoRedirect())
        self.snapshot={};self.snapshot_at=0;self.lock=threading.Lock();self.stop=threading.Event();self.results=[]
        self.seen=set();self.seen_lock=threading.Lock();self.pool=ThreadPoolExecutor(max_workers=6);self.last_tasks=None;self.last_task_at=0;self.tasks_future=None
        self.display_frames={}
        # Cloud command polling must continue during image tests, model loads,
        # and reset replies so STOP and genuine browser heartbeats can pass.
        self.command_pool=ThreadPoolExecutor(max_workers=1)
        self.urgent_pool=ThreadPoolExecutor(max_workers=2)
        self.results_lock=threading.Lock()

    def local_request(self,path,payload=None,timeout=3):
        request=urllib.request.Request(self.local+path,data=None if payload is None else json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        with self.opener.open(request,timeout=timeout) as response:
            raw=response.read(18000001)
            if len(raw)>18000000:raise ValueError('local_response_too_large')
            if path=='/camera.jpg':return dict(base64=base64.b64encode(raw).decode(),age_ms=float(response.headers.get('X-Frame-Age-Ms','0')))
            return json.loads(raw)

    def maybe_get(self,path):
        try:return self.local_request(path,timeout=2.5)
        except (OSError,ValueError):return None

    def collect(self):
        while not self.stop.is_set():
            started=time.monotonic()
            paths=['/model-switch/status','/health','/operation','/live-result','/camera.jpg']
            values=list(self.pool.map(self.maybe_get,paths));models,health,operation,vision,camera=values
            latest=self.maybe_get('/model-switch/status')
            if models and latest and latest.get('revision')!=models.get('revision'):vision=None
            if health and vision:
                h=health.get('model_sha256');v=vision.get('model_version',vision.get('model_sha256'))
                if h and v and h!=v:vision=None
            # Historical PS records may be slower than the camera. Never hold up
            # fresh telemetry or discard the last real records on a transient error.
            if self.tasks_future is not None and self.tasks_future.done():
                try:
                    records=self.tasks_future.result()
                    if records is not None:self.last_tasks=records
                except (OSError,ValueError):pass
                self.tasks_future=None
            if self.tasks_future is None and time.monotonic()-self.last_task_at>6:
                self.tasks_future=self.pool.submit(self.local_request,'/control-status',timeout=8);self.last_task_at=time.monotonic()
            op_keys=('state','busy','blockers','reason','operation_id','active_action','material_selection','owner','object_height_mm','voice_service','board_ownership','result','progress')
            op_keys+=('per_instance_heights_supported','selection_policy','height_changes_grasp_z',
                      'voice_height_registration_supported','voice_height_registration',
                      'voice_height_registration_session','voice_height_registration_version','per_instance_widths_supported','manual_review')
            instance_keys=('instance_id','label','official_label','confidence','model_material','reference_assisted','reference_material','reference_object_id','bbox_xyxy','mask_rle','station_zone','station_grasp_candidate')
            public_vision=None;overlay=None
            if vision:
                public_vision={k:vision.get(k) for k in ('frame_id','source_frame_id','inference_ms','model_version','configuration_sha256','image_size','display_grasp_points','display_geometry_timing','pickup_scope')}
                public_vision['source_stream_id']=vision.get('source_stream_id')
                public_vision['instances']=[{k:i.get(k) for k in instance_keys if k in i} for i in vision.get('instances',[])[:12]]
                public_vision['frame_id']=vision.get('frame_id') or vision.get('source_frame_id') or str(vision.get('generated_at'))
                if vision.get('overlay_base64'):
                    overlay=dict(base64=vision['overlay_base64'],age_ms=max(0,float(vision.get('result_age_ms',vision.get('source_frame_age_ms',0)) or 0)))
                    public_vision['frame_key']=hashlib.sha256(vision['overlay_base64'].encode()).hexdigest()
            delay=(time.monotonic()-started)*1000
            for frame in (camera,overlay):
                if frame:frame['age_ms']+=delay
            snapshot=clean(dict(models=latest or models,health={k:health.get(k) for k in ('model_name','model_sha256','execution','camera_connected')} if health else None,
                operation={k:operation.get(k) for k in op_keys} if operation else None,vision=public_vision,camera=camera,overlay=overlay,tasks=self.last_tasks))
            if len(json.dumps(snapshot))>2200000:snapshot.update(camera=None,overlay=None)
            with self.lock:
                self.snapshot=snapshot;self.snapshot_at=time.monotonic()
                if overlay:
                    key=public_vision['frame_key']
                    self.display_frames.setdefault(key,(dict(overlay),self.snapshot_at))
                self.display_frames={k:v for k,v in list(self.display_frames.items())[-6:] if (self.snapshot_at-v[1])*1000+v[0]['age_ms']<7000}
            self.stop.wait(max(.1,1.2-(time.monotonic()-started)))

    def execute(self,command):
        cid=command['id']
        with self.seen_lock:
            if cid in self.seen:return dict(id=cid,ok=False,unknown=True,error='请求已处理，不重复提交')
            self.seen.add(cid)
        if command.get('expired') or command.get('remaining_ms',6500)<=0:return dict(id=cid,ok=False,error='请求已过期，未提交')
        deadline=time.monotonic()+command.get('remaining_ms',6500)/1000
        payload=dict(command['payload']);kind=command['kind'];posted=False
        try:
            if kind=='select_model':
                current=self.local_request('/model-switch/status')
                if payload['revision']!=current['revision']:raise ValueError('模型已变化，请重新选择')
                path='/model-switch/select';payload=dict(model_id=payload['model_id'],revision=current['revision'],token=current['token'])
            elif kind=='operation':
                current=self.local_request('/operation');action=payload.get('action')
                allowed={'preview','start','arm','voice_arm','cancel','stop','reset','home','heartbeat','register_heights','clear_heights','review'}
                if action not in allowed:raise ValueError('不支持的工位动作')
                expected=payload.pop('expected_operation_id',None)
                if action not in ('stop','heartbeat','cancel','clear_heights') and current.get('operation_id')!=expected:raise ValueError('工位任务已变化，请核对现场后重新操作')
                # Review rechecks the same identity under the dispatcher lock.
                # Preserve the caller's value; never substitute a newer task.
                if action=='review':payload['expected_operation_id']=expected
                if action in ('heartbeat','cancel') and (payload.get('owner')!=current.get('owner') or payload.get('operation_id')!=current.get('operation_id')):raise ValueError('当前页面不是此任务的控制会话')
                payload['token']=current['token'];path='/operation'
            elif kind=='detect':path='/detect'
            elif kind=='task':
                data=self.local_request('/control-status?task_id='+urllib.parse.quote(payload['task_id'],safe=''))
                return dict(id=cid,ok=True,data=clean(data))
            elif kind=='operation_state':
                if payload:raise ValueError('状态读取不能附带动作')
                return dict(id=cid,ok=True,data=clean(self.local_request('/operation',timeout=4)))
            elif kind=='profiles':
                from item_profile_store import ItemProfileStore
                store=ItemProfileStore(Path(__file__).resolve().parents[1]/'local-only/item-profiles/profiles.json')
                if payload=={'action':'read'}:data=store.read()
                else:data=store.command(payload)
                return dict(id=cid,ok=True,data=data)
            else:raise ValueError('不支持的请求')
            if time.monotonic()>=deadline:raise ValueError('请求已过期，未提交')
            posted=True
            # One HTTP submission only, including after timeout or lost response.
            timeout=15 if kind=='operation' and payload.get('action')=='reset' else 8 if kind=='operation' else 35 if kind=='detect' else 8
            data=self.local_request(path,payload,timeout=timeout)
            return dict(id=cid,ok=True,data=clean(data))
        except urllib.error.HTTPError as error:
            try:detail=json.loads(error.read()).get('error','工位拒绝此操作')
            except (ValueError,OSError):detail='工位拒绝此操作'
            return dict(id=cid,ok=False,error=str(detail)[:220],code=error.code)
        except ValueError as error:return dict(id=cid,ok=False,error=str(error)[:220])
        except (OSError,TimeoutError):return dict(id=cid,ok=False,unknown=posted,error='响应未确认，不重复提交；请核对工位状态' if posted else '工位状态不可读，未提交操作')

    def _complete_command(self,command,received_at):
        command=dict(command,remaining_ms=command['remaining_ms']-(time.monotonic()-received_at)*1000)
        try:result=self.execute(command)
        except Exception:
            # A worker failure must not turn a claimed command into a retry.
            result=dict(id=command['id'],ok=False,unknown=True,error='连接器未确认本次结果，请核对工位状态，不自动重试')
        with self.results_lock:self.results.append(result)

    def exchange(self):
        with self.lock:
            data=json.loads(json.dumps(self.snapshot));age=(time.monotonic()-self.snapshot_at)*1000
        if age>6500:data.update(operation=None,vision=None,camera=None,overlay=None)
        else:
            for k in ('camera','overlay'):
                if data.get(k):data[k]['age_ms']+=age
        data['telemetry_age_ms']=age
        data['bridge']={'protocol':'cloud-relay-v2','freshness_ms':6500,'command_replay':False,
                        'client_heartbeat_required':True,'snapshot_available':bool(data.get('operation'))}
        headers={'Content-Type':'application/json','Authorization':'Bearer '+self.config['device_secret']}
        if self.config.get('service_token'):headers['OAI-Sites-Authorization']='Bearer '+self.config['service_token']
        with self.results_lock:pending=list(self.results[:4])
        request=urllib.request.Request(self.cloud+'/api/device/exchange',data=json.dumps(dict(snapshot=data,results=pending)).encode(),headers=headers)
        sent=time.monotonic()
        with self.cloud_opener.open(request,timeout=12) as response:reply=json.load(response)
        with self.results_lock:
            acknowledged={r['id'] for r in pending}
            self.results=[r for r in self.results if r['id'] not in acknowledged]
        command=reply.get('command')
        if command:
            # Expiry comes from the server's own clock, independent of PC time.
            command['remaining_ms']=command['expires_at']-reply['server_time']-(time.monotonic()-sent)*1000
            urgent=command.get('kind')=='operation' and command.get('payload',{}).get('action') in ('stop','heartbeat','cancel')
            (self.urgent_pool if urgent else self.command_pool).submit(self._complete_command,command,time.monotonic())
        return reply

    def run(self,status_path):
        threading.Thread(target=self.collect,daemon=True).start()
        while not self.stop.is_set():
            try:self.exchange();report=dict(connected=True,at=time.time(),pending_results=len(self.results));delay=.5
            except urllib.error.HTTPError as error:
                report=dict(connected=False,at=time.time(),reason='cloud_http_error',http_status=error.code,
                            edge_request_id=error.headers.get('cf-ray'));delay=2
            except (OSError,ValueError):report=dict(connected=False,at=time.time(),reason='cloud_connection_unavailable');delay=2
            try:Path(status_path).write_text(json.dumps(report),encoding='utf8')
            except OSError:pass
            self.stop.wait(delay)

    def current_snapshot(self):
        with self.lock:data=json.loads(json.dumps(self.snapshot));age=(time.monotonic()-self.snapshot_at)*1000
        if age>6500:data.update(operation=None,vision=None,camera=None,overlay=None)
        else:
            for key in ('camera','overlay'):
                if data.get(key):data[key]['age_ms']+=age
        return data

    def display_frame(self,key):
        with self.lock:
            entry=self.display_frames.get(key)
            if not entry:return None
            frame,stamp=entry
            return dict(frame,age_ms=frame['age_ms']+(time.monotonic()-stamp)*1000)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--init-config',action='store_true');p.add_argument('--start',action='store_true');args=p.parse_args()
    if args.init_config:
        value=json.load(sys.stdin);Bridge(value);save_config(args.config,value);print('Protected cloud device configuration saved');return
    if args.start:
        folder=args.config.resolve().parent;folder.mkdir(parents=True,exist_ok=True)
        with (folder/'bridge.stdout.log').open('ab') as out,(folder/'bridge.stderr.log').open('ab') as err:
            argv=[sys.executable,'-u',str(Path(__file__).resolve()),'--config',str(args.config.resolve())]
            proc=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        print(json.dumps(dict(launcher_pid=proc.pid,mechanical_commands_on_start=False)));return
    args.config=args.config.resolve();lock_path=args.config.with_suffix('.lock');lock=lock_path.open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    try:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:raise SystemExit('A bridge already owns this configuration')
    bridge=Bridge(load_config(args.config));print('Private cloud bridge starting; no autonomous operation',flush=True)
    try:bridge.run(args.config.with_suffix('.status.json'))
    finally:
        bridge.stop.set()
        for pool in (bridge.pool,bridge.command_pool,bridge.urgent_pool):pool.shutdown(wait=False,cancel_futures=True)
        lock.close()


if __name__=='__main__':main()
