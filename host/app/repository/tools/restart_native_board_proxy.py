"""Operator-run update of the thin PC web process, leaving the PS host intact.

Check-only is the default. An explicit --restart updates only the PC proxy after
checking the native station is quiescent and preserves its existing operation.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]


def quiescent_state(state):
    if (state.get('compute_location')!='pynq_ps' or state.get('busy') or
            state.get('board_ownership',{}).get('status') not in ('idle','blocked')):
        raise ValueError('quiescent_native_station_required')
    return {k:state.get(k) for k in ('board_root','operation_id','state','reason','board_ownership')}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--restart',action='store_true')
    args=p.parse_args()
    import psutil
    config=ROOT/'local-only/board-control-integration/native-active.json'
    cfg=json.loads(config.read_text(encoding='utf8'))
    if cfg['mode']!='board_station':raise ValueError('native_board_configuration_required')
    def status():
        with urllib.request.urlopen('http://127.0.0.1:8810/operation',timeout=12) as reply:return json.load(reply)
    state=status()
    preserved=quiescent_state(state)
    script=(ROOT/'tools/board_station_proxy.py').resolve();found=[]
    for proc in psutil.process_iter(['cmdline','cwd']):
        try:
            command=proc.info['cmdline'] or []
            if (not proc.info['cwd'] or Path(proc.info['cwd']).resolve()!=ROOT or
                not any(Path(v).resolve()==script for v in command if v.endswith('.py'))):continue
            if '--config' not in command or Path(command[command.index('--config')+1]).resolve()!=config:continue
            if '--port' not in command or command[command.index('--port')+1]!='8810':continue
            if any(connection.status=='LISTEN' and connection.laddr.port==8810
                   for connection in proc.net_connections(kind='tcp')):found.append(proc)
        except (psutil.Error,OSError,ValueError):continue
    if len(found)!=1:raise ValueError('unique_exact_native_proxy_required')
    report=dict(status='native_proxy_restart_check_passed',pid=found[0].pid,
        board_host_untouched=True,mechanical_commands=False,current_blockers=state['blockers'])
    if not args.restart:print(json.dumps(report));return
    # Recheck just before changing only the reviewed PC display process.
    if quiescent_state(status())!=preserved:raise ValueError('station_state_changed_before_proxy_update')
    found[0].terminate();found[0].wait(10)
    folder=config.parent/'operator-proxy-update';folder.mkdir(exist_ok=True)
    with (folder/'stdout.log').open('ab') as stdout,(folder/'stderr.log').open('ab') as stderr:
        process=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'-u',str(script),
            '--config',str(config),'--port','8810'],cwd=ROOT,stdin=subprocess.DEVNULL,
            stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    deadline=time.monotonic()+35
    while time.monotonic()<deadline:
        if process.poll() is not None:raise RuntimeError('native_proxy_exited_pc_remains_fenced')
        try:
            if quiescent_state(status())!=preserved:raise RuntimeError('station_state_changed_during_proxy_update')
            with urllib.request.urlopen('http://127.0.0.1:8810/proxy-health',timeout=5) as reply:health=json.load(reply)
            if health.get('transport_contract')=='isolated-control-v1':
                print(json.dumps(dict(report,status='native_proxy_updated',launcher_pid=process.pid,
                                      stopped_state_preserved=True,transport_contract=health['transport_contract'])));return
        except OSError:pass
        time.sleep(.5)
    raise TimeoutError('native_proxy_update_deadline_no_fallback')


if __name__=='__main__':main()
