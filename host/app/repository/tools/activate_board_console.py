"""Switch an idle console's preview adapter while holding its existing lease.

No mechanical endpoint is called. The prior console is restored if startup
fails. Only exact project console modules/ports are selected for replacement.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

from stage_board_control import ROOT,PROJECT,require_idle,sha


def selected_process(info,ports):
    args=info.get('cmdline') or []
    if not info.get('cwd') or Path(info['cwd']).resolve()!=PROJECT.resolve():return False
    try:
        module=args[args.index('-m')+1]
        port=int(args[args.index('--port')+1])
    except (ValueError,IndexError,TypeError):return False
    return module in ('src.sorting_prompted_console','src.board_prompted_console') and port in ports


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',type=int,default=8810)
    p.add_argument('--retire-preview-port',type=int,default=0)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--disable-pc-motion',action='store_true')
    p.add_argument('--upstream',default='http://127.0.0.1:8811')
    p.add_argument('--console-python',type=Path,default=ROOT/'.venv/Scripts/python.exe')
    args=p.parse_args()
    import psutil
    output=args.output.resolve()
    if not output.is_relative_to(ROOT/'local-only'):raise ValueError('local_only_output_required')
    sys.path.insert(0,str(PROJECT))
    from src.board_control_gateway_client import BoardPreviewClient
    from src.sorting_known import bound_file
    from src.sorting_relocated_run import current_source_config
    from src.sorting_unified_run import BatchLease
    config=ROOT/'local-only/board-control-integration/active.json'
    board=BoardPreviewClient(config,ROOT/'local-only')
    health=board.health()
    if health['execution_enabled'] is not False or health['ownership']['status']!='idle':
        raise ValueError('board_preview_not_idle')
    pointer=ROOT/'local-only/station-grasp-integration/active.json'
    binding=bound_file(pointer.parent,json.loads(pointer.read_text(encoding='utf-8-sig')))
    record=json.loads(binding.read_text(encoding='utf-8-sig'))
    cp=bound_file(binding.parent,record['files']['config']);c=current_source_config(cp)
    fixture=dict(config=c,source=dict(pointer_sha256=sha(pointer)));require_idle(fixture)
    published=json.loads(config.read_text(encoding='utf-8'))
    policy_source=Path(published.get('policy_source_path',PROJECT/'config/yoloe_taco_recyclables.json')).resolve()
    trial_source=Path(published.get('trial_scope_source_path',ROOT/'local-only/yoloe-recyclables/calibration-20261030/deployment.json')).resolve()
    if not all(path.is_relative_to(ROOT) for path in (policy_source,trial_source)):
        raise ValueError('preview_source_outside_repository')
    if (published.get('policy_source_sha256')!=sha(policy_source) or
        published.get('timing_source_sha256')!=sha(PROJECT/'src/sorting_prompted_motion.py') or
        published.get('motion_options_source_sha256')!=sha(PROJECT/'config/prompted_motion_policy.json') or
        published.get('trial_scope_source_sha256')!=sha(trial_source) or
        published.get('station_pointer_sha256')!=sha(pointer)):
        raise ValueError('board_preview_requires_current_source_snapshot')
    ports={args.port}
    if args.retire_preview_port:ports.add(args.retire_preview_port)
    chosen=[]
    for process in psutil.process_iter(['pid','cmdline','cwd','create_time']):
        try:
            if selected_process(process.info,ports):chosen.append(process)
        except (psutil.Error,OSError):pass
    if args.check_only:
        print(json.dumps(dict(status='idle_switch_reviewed',replace_pids=[p.pid for p in chosen],port=args.port,
                             physical_commands=False,preview_backend='pynq_ps',motion_backend='existing_pc')))
        return
    output.mkdir(parents=True,exist_ok=False)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def status(port,path='/operation'):
        with opener.open('http://127.0.0.1:'+str(port)+path,timeout=2) as reply:return json.load(reply)
    def launch(module,label):
        argv=[str(args.console_python.resolve()),'-u','-m',module,'--port',str(args.port),'--upstream',args.upstream]
        if module=='src.board_prompted_console':argv+=['--board-controller-config',str(config)]
        if module=='src.board_prompted_console' and args.disable_pc_motion:argv+=['--disable-pc-motion']
        with (output/(label+'.stdout.log')).open('ab') as out,(output/(label+'.stderr.log')).open('ab') as err:
            return subprocess.Popen(argv,cwd=PROJECT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    report=dict(kind='board-console-preview-activation-v1',port=args.port,physical_commands=False,
                preview_backend='pynq_ps',motion_backend='existing_pc',replace_pids=[p.pid for p in chosen])
    with BatchLease((cp.parent/c['known']['device_lease']).resolve(),
                    [(cp.parent/name).resolve() for name in c['known']['stop_files']],time.monotonic()+40):
        if status(args.port).get('busy'):raise ValueError('console_became_busy')
        for process in chosen:
            try:
                if not selected_process(dict(cmdline=process.cmdline(),cwd=process.cwd()),ports):
                    raise ValueError('console_process_identity_changed')
                process.terminate()
            except psutil.NoSuchProcess:pass
        psutil.wait_procs(chosen,timeout=4)
        launched=launch('src.board_prompted_console','board-console')
        try:
            deadline=time.monotonic()+15
            while time.monotonic()<deadline:
                if launched.poll() is not None:raise RuntimeError('new_board_console_exited')
                try:
                    result=status(args.port,'/board-control')
                    if result.get('connected') and result.get('compute_location')=='pynq_ps':break
                except (OSError,ValueError):pass
                time.sleep(.1)
            else:raise RuntimeError('board_console_startup_timeout')
            report.update(status='connected',pid=launched.pid,board_status=result)
        except Exception:
            launched.terminate()
            report.update(status='rolled_back',rollback_pid=launch('src.sorting_prompted_console','rollback').pid)
            raise
        finally:
            (output/'activation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    report['lease_released']=not (cp.parent/c['known']['device_lease']).exists()
    (output/'activation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=report['status'],port=args.port,pid=report['pid'],lease_released=report['lease_released'])))


if __name__=='__main__':main()
