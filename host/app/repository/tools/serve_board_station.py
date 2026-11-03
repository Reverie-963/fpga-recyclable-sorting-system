"""Serve PC vision links and start the native PS controller from verified stage.

The initial local port is a review port. Main-port cutover is a separate step.
No action or scene-ready request is sent by this deployment process.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path,PurePosixPath
import shlex
import sys
import threading
import time

import hashlib
ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT/'pynq_km1_vision'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
from board_station_links import make_visual_relay,ReverseVisualLink,PinnedBoardHTTP,make_control_proxy
from board_link_monitor import BoardLinkMonitor


def voice_entrypoint(remote,stage):
    bootstrap=stage.get('voice_bootstrap')
    if not bootstrap:return '-m src.board_voice_host'
    path=PurePosixPath(bootstrap['path'])
    if (path.is_absolute() or '..' in path.parts or
            stage['remote_sha256'].get(str(path))!=bootstrap['sha256']):
        raise ValueError('verified_voice_bootstrap_required')
    return shlex.quote(str(remote/path))+' --voice-module src.board_voice_host'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--connection-module',type=Path,required=True)
    p.add_argument('--upstream',default='http://127.0.0.1:8824')
    p.add_argument('--station-token-file',required=True)
    p.add_argument('--ownership-db',required=True)
    p.add_argument('--port',type=int,default=8815)
    p.add_argument('--board-port',type=int,default=18810)
    p.add_argument('--visual-port',type=int,default=8826)
    p.add_argument('--visual-console-port',type=int,default=8828)
    p.add_argument('--execution-enabled',action='store_true')
    p.add_argument('--voice',action='store_true')
    p.add_argument('--attach-existing',action='store_true',help='Resume links only after exact idle native process identity checks.')
    p.add_argument('--preserve-blocked',action='store_true',help='Upgrade an idle stopped host without clearing its authoritative blocked state.')
    args=p.parse_args()
    stage=json.loads(args.stage.read_text(encoding='utf-8'));output=args.stage.resolve().parent
    if not output.is_relative_to(ROOT/'local-only'):raise ValueError('local_only_stage_required')
    if stage.get('board_verification',{}).get('files_verified')!=stage['files']:
        raise ValueError('stage_not_verified')
    for name,digest in stage['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('native_stage_source_changed:'+name)
    remote=PurePosixPath(stage['remote_root']);state=stage.get('state_root',str(remote/'state'))
    deployment=output/'native-deployment.json'
    record=dict(status='starting',board_root=str(remote),board_port=args.board_port,
        execution_enabled=args.execution_enabled,physical_motion_on_startup=False,ownership_db=args.ownership_db,
        state_root=state,control_location='pynq_ps',perception_location='pc')
    deployment.write_text(json.dumps(record,indent=2),encoding='utf-8')
    os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        YOLOE_RECYCLABLE_POLICY=str(output/'perception-policy.local.json'))
    # Use the exact policy already frozen into the stage, without copying weights.
    from tarfile import open as open_tar
    with open_tar(output/'candidate.tar.gz') as tar:
        raw=tar.extractfile('perception/policy.json').read()
    policy_file=output/'perception-policy.local.json';policy_file.write_bytes(raw)
    sys.path.insert(0,str(PROJECT))
    from src.sorting_prompted_console import create_console
    from src.monitor_operation import Operations
    binding=output/'bundle'/stage['binding']['path']
    b=json.loads(binding.read_text(encoding='utf-8-sig'))
    camera_path=(binding.parent/b['files']['camera']['path']).resolve()
    token=(output/'generated/vision-token.txt').read_text().strip()
    control_token=(output/'generated/control-token.txt').read_text().strip()
    if stage.get('execution_models'):
        with open_tar(output/'candidate.tar.gz') as tar:
            profiles=json.loads(tar.extractfile(stage['execution_models']).read())
        from execution_visual_relay import make_execution_visual_relay
        visual=make_execution_visual_relay(args.visual_port,token,profiles,camera_path)
    else:visual=make_visual_relay(args.visual_port,token,args.upstream,camera_path)
    dummy=Operations()
    visual_console=create_console(args.visual_console_port,args.upstream,operations=dummy)
    original=visual_console.RequestHandlerClass
    class VisualOnly(original):
        def do_POST(self):
            if self.path=='/prompted/plan' or not (self.path in ('/live','/detect') or self.path.startswith('/prompted/')):
                return self.reply(403,dict(error='pc_visual_only_control_is_on_board'))
            return super().do_POST()
        def do_GET(self):
            if self.path.startswith(('/operation','/voice','/grasp-record','/control-status')):
                return self.reply(403,dict(error='pc_visual_only_control_is_on_board'))
            return super().do_GET()
    visual_console.RequestHandlerClass=VisualOnly
    for server in (visual,visual_console):
        threading.Thread(target=server.serve_forever,daemon=True).start()
    spec=importlib.util.spec_from_file_location('private_connection',args.connection_module.resolve())
    private=importlib.util.module_from_spec(spec);spec.loader.exec_module(private)
    client=private.connect();control_client=None;reverse=None;proxy=None;board=None
    try:
        def execute(code,timeout=15):
            stdin,out,err=client.exec_command('timeout '+str(timeout)+'s python3 -',timeout=timeout+5)
            stdin.write(code);stdin.flush();stdin.channel.shutdown_write()
            raw,error=out.read().decode(),err.read().decode()
            if out.channel.recv_exit_status():raise RuntimeError('native_preflight_failed:'+error[-700:])
            return raw
        probe='''import sqlite3,json,pathlib,os
p=pathlib.Path('''+repr(args.ownership_db)+''');assert p.is_file()
db=sqlite3.connect('file:'+str(p)+'?mode=ro',uri=True)
row=db.execute('SELECT value FROM state WHERE id=1').fetchone();assert row
state=json.loads(row[0]);assert state['status']=='idle' or ('''+repr(args.preserve_blocked)+''' and state['status']=='blocked'),'owner_not_idle';db.close()
m={k:int(v.split()[0]) for k,v in (line.split(':',1) for line in open('/proc/meminfo'))}
assert m['MemAvailable']>=200000 and os.getloadavg()[0]<2.5,'board_resource_margin_insufficient'
print(json.dumps({'memory_available_kb':m['MemAvailable'],'load':os.getloadavg(),'existing_owner_status':state['status'],'existing_owner_reason':state.get('reason')}))
'''
        record['resource_preflight']=json.loads(execute(probe))
        existing={}
        if args.attach_existing:
            bootstrap=stage.get('bootstrap',{}).get('path')
            bootstrap_path=str(remote/bootstrap) if bootstrap else None
            probe='''import pathlib,json
root='''+repr(str(remote))+'''
bootstrap='''+repr(bootstrap_path)+'''
found={}
for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
 try:parts=p.read_bytes().decode().split('\\0')
 except (OSError,UnicodeError):continue
 for module in ('src.board_station_host','src.board_voice_host'):
  if module not in parts and not (module=='src.board_station_host' and bootstrap in parts):continue
  assert '--root' in parts and parts[parts.index('--root')+1]==root,'different_native_host_active'
  assert module not in found,'duplicate_native_host'
  found[module]=int(p.parent.name)
assert 'src.board_station_host' in found,'native_host_missing'
print(json.dumps(found))
'''
            existing=json.loads(execute(probe))
        reverse=ReverseVisualLink(client,args.visual_port);reverse.start()
        command=('cd '+shlex.quote(str(remote/'pynq_km1_vision'))+
            ' && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 '
            '/usr/local/share/pynq-venv/bin/python3 -u -m src.board_station_host')
        if stage.get('bootstrap'):
            bootstrap=PurePosixPath(stage['bootstrap']['path'])
            if bootstrap.is_absolute() or '..' in bootstrap.parts or stage['remote_sha256'].get(str(bootstrap))!=stage['bootstrap']['sha256']:
                raise ValueError('verified_candidate_bootstrap_required')
            command=command.removesuffix('-m src.board_station_host')+shlex.quote(str(remote/bootstrap))
        options={'root':str(remote),'binding':str(remote/stage['binding']['path']),
            'policy':str(remote/'perception/policy.json'),'settings':str(remote/'perception/settings.json'),
            'calibration':str(remote/'perception/calibration/deployment.json'),
            'station-token-file':args.station_token_file,'vision-token-file':str(remote/'vision-token.txt'),
            'control-token-file':str(remote/'control-token.txt'),'ownership-db':args.ownership_db,
            'state-root':state,'vision-url':'http://127.0.0.1:'+str(args.visual_port),'port':str(args.board_port)}
        options['station-config']=str(PurePosixPath(args.station_token_file).parent/'config/station_vision.local.json')
        if stage.get('station_config'):
            station_config=PurePosixPath(stage['station_config'])
            if station_config.is_absolute() or '..' in station_config.parts or str(station_config) not in stage['remote_sha256']:
                raise ValueError('verified_stage_station_configuration_required')
            options['station-config']=str(remote/station_config)
        for key,value in options.items():command+=' --'+key+' '+shlex.quote(value)
        if args.execution_enabled:command+=' --execution-enabled'
        streams=[]
        if not args.attach_existing:
            _,stdout,stderr=client.exec_command(command,timeout=90)
            line=stdout.readline()
            try:ready=json.loads(line)
            except (ValueError,TypeError):
                raise RuntimeError('native_station_did_not_start:'+stderr.read().decode()[-1200:])
            if ready.get('status')!='board_station_ready':raise ValueError('native_station_not_ready')
            streams=[(stdout,'board.stdout.log'),(stderr,'board.stderr.log')]
        else:record['attached_existing_processes']=existing
        def drain(stream,name):
            with (output/name).open('ab') as log:
                while True:
                    try:line=stream.readline()
                    except OSError:return
                    if not line:return
                    if isinstance(line,str):line=line.encode()
                    log.write(line);log.flush()
        for stream,name in streams:
            threading.Thread(target=drain,args=(stream,name),daemon=True).start()
        control_client=private.connect()
        board=PinnedBoardHTTP(client,args.board_port,control_token,control_client=control_client)
        code,raw,_=board.request('/board-control')
        if code!=200 or json.loads(raw)['compute_location']!='pynq_ps':raise ValueError('native_board_health_invalid')
        proxy=make_control_proxy(args.port,board,'http://127.0.0.1:'+str(args.visual_console_port))
        threading.Thread(target=proxy.serve_forever,daemon=True).start()
        config=dict(mode='board_station',connection_module=str(args.connection_module.resolve()),
            board_port=args.board_port,board_root=str(remote),ownership_db=args.ownership_db,state_root=state,
            control_token_file=str(output/'generated/control-token.txt'),
            visual_console_url='http://127.0.0.1:'+str(args.visual_console_port),
            review_url='http://127.0.0.1:'+str(args.port),execution_enabled=args.execution_enabled)
        config_path=ROOT/'local-only/board-control-integration/native-active.json'
        config_path.write_text(json.dumps(config,indent=2),encoding='utf-8')
        if args.voice and 'src.board_voice_host' not in existing:
            voice=('cd '+shlex.quote(str(remote/'pynq_km1_vision'))+
                ' && { nohup env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 '
                '/usr/local/share/pynq-venv/bin/python3 -u '+voice_entrypoint(remote,stage)+' --root '+shlex.quote(str(remote))+
                ' --control-token-file '+shlex.quote(str(remote/'control-token.txt'))+
                ' --controller-port '+str(args.board_port)+' --voice-port '+shlex.quote(stage['voice_device'])+
                ' --state-root '+shlex.quote(state)+' > '+shlex.quote(str(remote/'voice.log'))+' 2>&1 < /dev/null & echo $!; }')
            _,out,err=client.exec_command(voice,timeout=10);pid=out.readline().strip()
            if not pid.isdigit():raise RuntimeError('voice_start_outcome_unknown')
            out.channel.close()
            record['voice_launcher_pid']=int(pid)
        record.update(status='native_board_station_serving',review_url=config['review_url'],config_path=str(config_path))
        deployment.write_text(json.dumps(record,indent=2),encoding='utf-8')
        print(json.dumps(record),flush=True)
        monitor=BoardLinkMonitor(board,str(remote))
        while True:
            time.sleep(5)
            # A diagnostic timeout must not close healthy local vision servers.
            # The independent main controller owns its lease and stop handling.
            record['board_link_health']=monitor.poll()
            deployment.write_text(json.dumps(record,indent=2),encoding='utf-8')
    except KeyboardInterrupt:record['status']='host_exit_requested'
    except Exception as exc:
        record.update(status='failed',reason=type(exc).__name__+':'+str(exc));raise
    finally:
        if board is not None:
            try:
                code,raw,_=board.request('/operation')
                status=json.loads(raw)
                if code==200 and status.get('busy'):
                    board.request('/operation',json.dumps(dict(action='stop',owner='native-host-exit',token=status['token'])).encode())
            except Exception:pass
        if proxy:proxy.shutdown();proxy.server_close()
        if reverse:reverse.close()
        for server in (visual,visual_console):server.shutdown();server.server_close()
        dummy.close()
        if control_client is not None:control_client.close()
        client.close()
        deployment.write_text(json.dumps(record,indent=2),encoding='utf-8')


if __name__=='__main__':main()
