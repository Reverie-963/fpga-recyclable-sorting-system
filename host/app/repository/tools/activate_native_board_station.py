"""Fence the retired PC binding and publish the thin primary board proxy.

This hands off unresolved records unchanged. It never reviews, clears, opens a
UART, sends a motion, retries an operation, or restores the PC execution backend.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--activate',action='store_true')
    args=p.parse_args()
    stage=json.loads(args.stage.read_text(encoding='utf8'))
    config=json.loads(args.config.read_text(encoding='utf8'))
    output=args.stage.resolve().parent
    if (not output.is_relative_to(ROOT/'local-only') or config['mode']!='board_station' or
        config['board_root']!=stage['remote_root'] or config['state_root']!=stage['state_root'] or
        stage['board_verification']['files_verified']!=stage['files']):
        raise ValueError('verified_native_candidate_identity_required')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def get(base,path):
        with opener.open(base+path,timeout=12) as reply:return json.load(reply)
    control=get(config['review_url'],'/board-control')
    operation=get(config['review_url'],'/operation')
    workflow=get(config['review_url'],'/workflow')
    visual=get(config['review_url'],'/prompted/health')
    if (control['compute_location']!='pynq_ps' or control['ownership']['status']!='idle' or
        operation['busy'] or workflow['state_and_ledger_location']!='pynq_ps' or
        visual['ready'] is not True or operation['prompted_global_confidence']!=visual['settings']['confidence']):
        raise ValueError('native_station_identity_or_idle_check_failed')
    if set(operation['blockers'])-{'existing_physical_trial_unresolved','unified_batch_needs_review'}:
        raise ValueError('native_station_has_non_handoff_blockers:'+','.join(operation['blockers']))
    # An existing retained pending is transferred, never waived by activation.
    pointer=ROOT/'local-only/station-grasp-integration/active.json'
    original=pointer.read_bytes()
    if hashlib.sha256(original).hexdigest()!=stage['source_pointer_sha256']:
        raise ValueError('legacy_pointer_changed_since_stage')
    binding=json.loads(original.decode('utf-8-sig'))
    binding_path=(pointer.parent/binding['path']).resolve()
    if hashlib.sha256(binding_path.read_bytes()).hexdigest()!=binding['sha256']:
        raise ValueError('legacy_binding_changed')
    from serial.tools import list_ports
    if any((port.vid,port.pid)==(0x1a86,0x7523) for port in list_ports.comports()):
        raise ValueError('arm_usb_still_present_on_pc')
    with socket.socket() as probe:
        probe.settimeout(1)
        if probe.connect_ex(('127.0.0.1',8810))==0:raise ValueError('primary_port_already_in_use')
    record=dict(status='native_cutover_check_passed',board_root=config['board_root'],state_root=config['state_root'],
        ownership_db=config['ownership_db'],pending_preserved=True,physical_execution_admitted=False,
        current_blockers=operation['blockers'],physical_motion=False,model_hashes=visual['model_hashes'],
        model_execution=visual.get('execution'),confidence=visual['settings']['confidence'])
    if not args.activate:
        (output/'cutover-check.json').write_text(json.dumps(record,indent=2),encoding='utf8')
        print(json.dumps(record));return
    backup=output/'legacy-original-pointer.snapshot'
    backup.write_bytes(original)
    source_pointer=output/'legacy-source-pointer.json'
    source_pointer.write_text(json.dumps(dict(path=str(binding_path),sha256=binding['sha256']),indent=2),encoding='utf8')
    fence=dict(mode='board_station',pc_mechanical_execution_disabled=True,
        endpoint='http://127.0.0.1:8810',native_config=str(args.config.resolve()),
        legacy_pointer_snapshot=str(backup),legacy_pointer_sha256=hashlib.sha256(original).hexdigest(),
        archived_static_source_pointer=str(source_pointer),pending_preserved_on_board=True,
        automatic_pc_fallback=False)
    temporary=pointer.with_suffix('.native-handoff.tmp')
    temporary.write_text(json.dumps(fence,indent=2),encoding='utf8');temporary.replace(pointer)
    flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    with (output/'primary-proxy.stdout.log').open('ab') as stdout,(output/'primary-proxy.stderr.log').open('ab') as stderr:
        process=subprocess.Popen([sys.executable,str(ROOT/'tools/board_station_proxy.py'),
            '--config',str(args.config.resolve()),'--port','8810'],cwd=ROOT,stdin=subprocess.DEVNULL,
            stdout=stdout,stderr=stderr,creationflags=flags,close_fds=True)
    deadline=time.monotonic()+35
    try:
        while time.monotonic()<deadline:
            if process.poll() is not None:raise RuntimeError('native_primary_proxy_exited')
            try:
                current=get('http://127.0.0.1:8810','/operation')
                if current.get('compute_location')=='pynq_ps' and not current['busy']:break
            except OSError:pass
            time.sleep(.5)
        else:raise TimeoutError('native_primary_proxy_start_deadline')
    except BaseException:
        record['status']='primary_proxy_unavailable_pc_remains_fenced'
        (output/'cutover.json').write_text(json.dumps(record,indent=2),encoding='utf8')
        raise
    record.update(status='native_board_primary_active',primary_url='http://127.0.0.1:8810',
        proxy_launcher_pid=process.pid,pc_binding_fenced=True)
    (output/'cutover.json').write_text(json.dumps(record,indent=2),encoding='utf8')
    print(json.dumps(record))


if __name__=='__main__':main()
