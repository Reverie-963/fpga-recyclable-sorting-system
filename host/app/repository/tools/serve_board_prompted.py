"""Deploy and serve a read-only board preview, preserving the current PC model."""
import argparse
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path,PurePosixPath
import secrets
import shlex
import sys
import threading
import time

from stage_board_control import ROOT,PROJECT,require_idle,sha
from prepare_board_preview import freeze_preview_inputs,verify_preview_inputs
from board_control_gateway import BoardRPC,make_gateway


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--connection-module',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--remote-root',required=True)
    p.add_argument('--station-token-file',required=True)
    p.add_argument('--ownership-db',required=True,
                   help='Existing authoritative board database; never create a fresh owner for a cutover.')
    p.add_argument('--upstream',default='http://127.0.0.1:8811')
    p.add_argument('--policy-source',type=Path)
    p.add_argument('--trial-scope-source',type=Path)
    p.add_argument('--board-port',type=int,default=18812)
    p.add_argument('--gateway-port',type=int,default=18813)
    args=p.parse_args()
    output=args.output.resolve();remote=PurePosixPath(args.remote_root)
    if not output.is_relative_to(ROOT/'local-only'):raise ValueError('local_only_output_required')
    if remote.parent!=PurePosixPath('/home/xilinx') or not remote.name.startswith('km1-board-preview-'):
        raise ValueError('new_board_preview_directory_required')
    station_token=PurePosixPath(args.station_token_file)
    ownership_db=PurePosixPath(args.ownership_db)
    if (ownership_db.parent.parent.parent!=PurePosixPath('/home/xilinx') or
        not ownership_db.parent.parent.name.startswith('km1-board-preview-') or
        ownership_db.parent.name!='state-service' or ownership_db.name!='ownership.db'):
        raise ValueError('existing_board_ownership_database_required')
    if (station_token.parent.parent!=PurePosixPath('/home/xilinx') or
        not station_token.parent.name.startswith('km1-station-vision-') or station_token.name!='token.txt'):
        raise ValueError('existing_board_station_token_path_required')
    if not 1024<=args.board_port<=65535 or not 1024<=args.gateway_port<=65535:raise ValueError('invalid_preview_port')
    output.mkdir(parents=True,exist_ok=False)
    os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
    sys.path.insert(0,str(PROJECT))
    from src.board_detector_tunnel import DetectorTunnel
    from src.sorting_known import bound_file
    from src.sorting_relocated_run import current_source_config
    from src.sorting_unified_run import BatchLease
    from src.station_geometry import detect_markers
    import cv2
    import numpy as np
    pointer=ROOT/'local-only/station-grasp-integration/active.json'
    bpath=bound_file(pointer.parent,json.loads(pointer.read_text(encoding='utf-8-sig')))
    b=json.loads(bpath.read_text(encoding='utf-8-sig'));cp=bound_file(bpath.parent,b['files']['config'])
    camera_path=bound_file(bpath.parent,b['files']['camera'])
    camera=json.loads(camera_path.read_text(encoding='utf-8-sig'));camera_sha=sha(camera_path)
    def marker_detector(payload):
        if sha(camera_path)!=camera_sha:raise ValueError('board_marker_camera_snapshot_changed')
        image=cv2.imdecode(np.frombuffer(base64.b64decode(payload['image_base64'],validate=True),np.uint8),cv2.IMREAD_COLOR)
        if image is None:raise ValueError('cannot_decode_marker_input')
        return dict(location='pc_vision',camera_sha256=camera_sha,
                    source_bgr_sha256=hashlib.sha256(image.tobytes()).hexdigest(),detections=detect_markers(image,camera))
    fixture=dict(config=current_source_config(cp),source=dict(pointer_sha256=sha(pointer)))
    require_idle(fixture)
    frozen=freeze_preview_inputs(output,pointer,args.policy_source,args.trial_scope_source)
    bundle,files=frozen['bundle'],frozen['files']
    originals,digests=frozen['originals'],frozen['digests']
    motion_options_source=PROJECT/'config/prompted_motion_policy.json'
    motion_options_sha=frozen['motion_options_sha']
    motion_options=frozen['motion_options']
    trial_scope_source=originals['prompted-trial-scope.json']
    trial_scope_sha=frozen['trial_scope_sha']
    def current_motion_options():
        if sha(motion_options_source)!=motion_options_sha:raise ValueError('current_prompted_motion_options_changed')
        if sha(trial_scope_source)!=trial_scope_sha:raise ValueError('current_prompted_trial_scope_changed')
        if any(sha(path)!=digests[name] for name,path in originals.items() if name.endswith('.py')):
            raise ValueError('current_prompted_runtime_source_changed')
        return motion_options
    timing_source=PROJECT/'src/sorting_prompted_motion.py'
    timing_sha,timing=frozen['timing_sha'],frozen['timing']
    def current_timing():
        if sha(timing_source)!=timing_sha:raise ValueError('current_prompted_motion_timing_source_changed')
        return timing
    spec=importlib.util.spec_from_file_location('private_connection',args.connection_module.resolve())
    private=importlib.util.module_from_spec(spec);spec.loader.exec_module(private)
    client=private.connect();rpc=gateway=None
    stop=threading.Event()
    manifest=dict(kind='board-preview-service-deployment-v1',remote_root=str(remote),status='preparing',
                  physical_motion=False,execution_enabled=False)
    relay_token,control_token,gateway_token=(secrets.token_hex(32) for _ in range(3))
    for name,token in (('relay-token.txt',relay_token),('control-token.txt',control_token),('gateway-token.txt',gateway_token)):
        (output/name).write_text(token,encoding='ascii')
    files.update({'relay-token.txt':output/'relay-token.txt','control-token.txt':output/'control-token.txt'})
    def execute(command,timeout=15):
        _,out,err=client.exec_command(command,timeout=timeout)
        raw,error=out.read().decode(),err.read().decode()
        if out.channel.recv_exit_status():raise RuntimeError('board_preview_command_failed:'+error[-1000:])
        return raw
    try:
        ownership_probe=('import sqlite3,json,pathlib; p=pathlib.Path('+repr(str(ownership_db))+'); '
            'assert p.is_file(),"authoritative_ownership_database_missing"; '
            'db=sqlite3.connect("file:"+str(p)+"?mode=ro",uri=True); '
            'row=db.execute("SELECT value FROM state WHERE id=1").fetchone(); '
            'assert row and json.loads(row[0])["status"]=="idle","existing_board_owner_not_idle"; db.close()')
        execute('python3 -c '+shlex.quote(ownership_probe))
        probe='import os; m={k:int(v.split()[0]) for k,v in (line.split(":",1) for line in open("/proc/meminfo"))}; assert m["MemAvailable"]>=200000 and os.getloadavg()[0]<2.5,"board_busy_defer"'
        execute('python3 -c '+shlex.quote(probe))
        with DetectorTunnel(private.connect,relay_token,args.upstream,max_channels=3,max_inflight=3,strip_overlay=True,marker_detector=marker_detector) as tunnel:
            sftp=client.open_sftp()
            try:
                sftp.mkdir(str(remote),mode=0o700)
                parents={'state-service'}
                for name in files:
                    parents.update(str(p) for p in PurePosixPath(name).parents if str(p)!='.')
                for parent in sorted(parents,key=lambda v:(len(PurePosixPath(v).parts),v)):
                    sftp.mkdir(str(remote/parent))
                for name,path in files.items():
                    sftp.put(str(path),str(remote/name))
                    if name.endswith('token.txt'):sftp.chmod(str(remote/name),0o600)
                    with sftp.open(str(remote/name),'rb') as stream:
                        if hashlib.sha256(stream.read()).hexdigest()!=sha(path):raise ValueError('board_preview_source_hash_mismatch')
            finally:sftp.close()
            manifest['source_sha256']={name:sha(path) for name,path in files.items() if not name.endswith('token.txt')}
            require_idle(fixture)
            verify_preview_inputs(output)
            listener='import subprocess,json; print(json.dumps([r.split()[3] for r in subprocess.check_output(["ss","-lntH"],text=True).splitlines() if r.split()[3].endswith('+repr(':'+str(tunnel.port))+')]))'
            if json.loads(execute('python3 -c '+shlex.quote(listener)))!=['127.0.0.1:'+str(tunnel.port)]:
                raise ValueError('board_detector_forward_not_exclusively_loopback')
            command=('cd '+shlex.quote(str(remote/'pynq_km1_vision'))+' && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '
                'timeout 7200s nice -n 19 /usr/local/share/pynq-venv/bin/python3 -u -m src.board_control_service '
                '--binding '+shlex.quote('../'+bundle['binding']['path'])+' --policy ../prompted-policy.json '
                '--relay-url '+shlex.quote(tunnel.board_url)+' --relay-token-file ../relay-token.txt '
                '--station-token-file '+shlex.quote(str(station_token))+' --control-token-file ../control-token.txt '
                '--observation-timing-file ../observation-timing.json --state ../state-service --port '+str(args.board_port))
            command+=' --ownership-db '+shlex.quote(str(ownership_db))
            _,stdout,stderr=client.exec_command(command,timeout=20)
            ready=json.loads(stdout.readline())
            if ready.get('status')!='board_preview_ready':raise ValueError('board_preview_not_ready')
            def drain(stream,path):
                with path.open('ab') as log:
                    while True:
                        try:chunk=stream.read(4096)
                        except OSError:return
                        if not chunk:return
                        log.write(chunk);log.flush()
            for stream,name in ((stdout,'board-stdout.log'),(stderr,'board-stderr.log')):
                threading.Thread(target=drain,args=(stream,output/name),daemon=True).start()
            rpc=BoardRPC(client,args.board_port,control_token)
            health=rpc.request('/health')
            if health['mode']!='prompted_preview_only' or health['execution_enabled'] is not False:
                raise ValueError('board_preview_execution_contract_invalid')
            verify_preview_inputs(output)
            def pc_preview_lease():
                c=fixture['config']
                return BatchLease((cp.parent/c['known']['device_lease']).resolve(),
                    [(cp.parent/name).resolve() for name in c['known']['stop_files']],time.monotonic()+90)
            gateway=make_gateway(rpc,gateway_token,args.upstream,args.gateway_port,idle_check=lambda:require_idle(fixture),
                                 timing_provider=current_timing,lease_factory=pc_preview_lease,motion_options_provider=current_motion_options)
            thread=threading.Thread(target=gateway.serve_forever,daemon=True);thread.start()
            active=ROOT/'local-only/board-control-integration/active.json';active.parent.mkdir(exist_ok=True)
            config=dict(mode='preview_only',url='http://127.0.0.1:'+str(gateway.server_port),
                        token_file=str(output/'gateway-token.txt'),remote_root=str(remote),
                        ownership_db=str(ownership_db),
                        policy_source_path=str(originals['prompted-policy.json']),
                        trial_scope_source_path=str(trial_scope_source),
                        policy_source_sha256=digests['prompted-policy.json'],timing_source_sha256=timing_sha,
                        motion_options_source_sha256=motion_options_sha,
                        trial_scope_source_sha256=trial_scope_sha,
                        station_pointer_sha256=fixture['source']['pointer_sha256'])
            temp=active.with_suffix('.tmp');temp.write_text(json.dumps(config,indent=2),encoding='utf-8');temp.replace(active)
            manifest.update(status='board_preview_serving',gateway_url=config['url'],active_config=str(active),
                            board_port=args.board_port,relay_port=tunnel.port,files=len(files))
            (output/'deployment.json').write_text(json.dumps(manifest,indent=2))
            print(json.dumps(dict(status=manifest['status'],gateway_url=config['url'],files=len(files),execution_enabled=False)),flush=True)
            while not stop.wait(10):
                tunnel.require_connected();rpc.request('/health')
    except KeyboardInterrupt:
        manifest['status']='stopped'
    except Exception as exc:
        manifest.update(status='failed',error_type=type(exc).__name__,reason=str(exc))
        raise
    finally:
        if gateway is not None:gateway.shutdown();gateway.server_close()
        if rpc is not None:
            try:rpc.request('/shutdown',{})
            except (OSError,ValueError):pass
        client.close()
        (output/'deployment.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
