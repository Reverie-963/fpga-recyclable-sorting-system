"""Build and transfer a complete PS control candidate; never start it or move.

Preserve original qualification/authorization values and historical ledger bytes.
Only runtime paths and the explicitly identified USB mapping become portable.
"""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path,PurePosixPath
import secrets
import shlex
import tarfile

from stage_board_control import ROOT,PROJECT,sha,dependency_files
from export_board_control import Exporter


class ControlExporter(Exporter):
    def __init__(self,output,arm_device,state_root=None):
        super().__init__(output);self.arm_device=arm_device;self.original_authorized=None
        self.state_root=PurePosixPath(str(state_root).replace('\\','/')) if state_root else None
    def config(self,base,record):
        source=self.read(base,record)
        original=json.loads(source.read_text(encoding='utf-8-sig'))
        result=super().config(base,record)
        value=json.loads((self.output/'objects'/result['path']).read_text(encoding='utf-8'))
        self.original_authorized=original['known']['experiment_authorized']
        # Retain the source authorization, never manufacture a qualification.
        value['known']['experiment_authorized']=self.original_authorized
        # Port is part of the historical qualification fingerprint. Preserve it
        # and map only the private process's transport copy, never re-sign proof.
        value['board_transport']=dict(qualified_port=original['arm']['port'],
            arm_device=self.arm_device,usb_identity=original['arm']['usb_identity'])
        value['camera_url']='http://127.0.0.1:8783/frame.jpg'
        value['statistics']['url']='http://127.0.0.1:8783'
        value['migration_execution_disabled']=False
        if self.state_root:
            value['journal_dir']=str(self.state_root/'ledger')
            value['known']['device_lease']=str(self.state_root/'device.lock')
            value['known']['stop_files']=[str(self.state_root/'STOP')]
        return dict(result,**self.json_object(value))
    def export(self,pointer):
        result=super().export(pointer)
        result.update(status='portable_control_candidate_not_started',source_authorization_preserved=True,
            requires_explicit_execution_action=True,
            caveat='Original authorization retained. No service or UART has started; ownership and scene checks remain mandatory.')
        (self.output/'bundle.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--connection-module',type=Path,required=True)
    p.add_argument('--deployment',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--remote-root',required=True)
    p.add_argument('--arm-device',required=True)
    p.add_argument('--voice-device',required=True)
    p.add_argument('--state-root',help='Existing PS state from a previously verified candidate; omit to copy original full history.')
    p.add_argument('--state-proof',type=Path,help='Verified first import used to identify ledger deltas.')
    p.add_argument('--handoff-paused',action='store_true',help='Preserve unresolved source pending during an explicitly paused handoff, never clear it.')
    p.add_argument('--source-pointer',type=Path,help='Explicit archived static source after PC control is fenced; never an automatic fallback.')
    p.add_argument('--control-token-source',type=Path,help='Keep current proxy authentication during a PS-only upgrade; never printed.')
    args=p.parse_args()
    output=args.output.resolve();deployment=args.deployment.resolve();remote=PurePosixPath(args.remote_root)
    if not output.is_relative_to(ROOT/'local-only') or not deployment.is_relative_to(ROOT/'local-only'):
        raise ValueError('local_only_paths_required')
    if remote.parent!=PurePosixPath('/home/xilinx') or not remote.name.startswith('km1-board-control-'):
        raise ValueError('new_board_control_directory_required')
    for device in (args.arm_device,args.voice_device):
        if not device.startswith('/dev/ttyUSB') or not device.removeprefix('/dev/ttyUSB').isdigit():
            raise ValueError('explicit_identified_board_usb_device_required')
    output.mkdir(parents=True,exist_ok=False)
    pointer=(args.source_pointer or ROOT/'local-only/station-grasp-integration/active.json').resolve()
    if not pointer.is_relative_to(ROOT/'local-only'):raise ValueError('local_archived_source_pointer_required')
    load=lambda path:json.loads(path.read_text(encoding='utf-8-sig'))
    active=load(pointer)
    if active.get('mode')=='board_station':raise ValueError('pc_control_fenced_use_explicit_archived_static_source')
    if args.source_pointer and (not args.state_root or args.state_proof):
        raise ValueError('archived_source_upgrade_must_keep_existing_ps_state_without_reimporting_pc_ledger')
    binding=(pointer.parent/active['path']).resolve();source_binding=load(binding)
    config_path=(binding.parent/source_binding['files']['config']['path']).resolve();config=load(config_path)
    ledger=(config_path.parent/config['journal_dir']).resolve()
    protected=[ledger/'controller.lock',
               (config_path.parent/config['known']['device_lease']).resolve()]
    protected += [(config_path.parent/v).resolve() for v in config['known']['stop_files']]
    if not args.handoff_paused:protected += [ledger/'pending.json',ledger/'batch-pending.json']
    if any(path.exists() for path in protected):raise ValueError('source_control_not_idle_and_reviewed')
    state_root=PurePosixPath(args.state_root) if args.state_root else None
    if state_root and (state_root.parent.parent!=PurePosixPath('/home/xilinx') or
        not state_root.parent.name.startswith('km1-board-control-') or state_root.name!='state'):
        raise ValueError('existing_native_board_state_directory_required')
    exporter=ControlExporter(output/'bundle',args.arm_device,state_root)
    bundle=exporter.export(pointer)
    if exporter.original_authorized is not True:raise ValueError('original_source_not_authorized_for_control')
    files={name:output/'bundle'/name for name in bundle['output_sha256']}
    for entry in ('src.board_station_host','src.board_voice_host','src.sorting_prompted_console'):
        for name,path in dependency_files(entry).items():
            target='pynq_km1_vision/'+name
            if target in files and sha(files[target])!=sha(path):raise ValueError('board_bound_source_changed')
            files[target]=path
    for path in (PROJECT/'src/monitor_assets').rglob('*'):
        if path.is_file() and path.suffix in ('.html','.css','.js','.svg','.png','.jpg'):
            files[path.relative_to(ROOT).as_posix()]=path
    files['pynq_km1_vision/config/prompted_motion_policy.json']=PROJECT/'config/prompted_motion_policy.json'
    for name in ('policy.json','settings.json','calibration/deployment.json','calibration/summary.json'):
        files['perception/'+name]=deployment/name
    receipt=load(deployment/'calibration/deployment.json')
    for name,digest in receipt.get('implementation_sha256',{}).items():
        source=ROOT/name
        if sha(source)!=digest:raise ValueError('current_trial_receipt_source_changed:'+name)
        files[name]=source
    # Preserve the complete original ledger, including images and recovery proof.
    if state_root is None:
        for path in ledger.rglob('*'):
            if path.is_file():
                if path.is_symlink() or not path.resolve().is_relative_to(ledger):raise ValueError('ledger_link_escape')
                files['state/ledger/'+path.relative_to(ledger).as_posix()]=path
    delta={};previous={}
    if state_root is not None and args.state_proof:
        proof=load(args.state_proof)
        if (proof.get('board_verification',{}).get('files_verified')!=proof['files'] or
            str(PurePosixPath(proof['remote_root'])/'state')!=str(state_root)):
            raise ValueError('verified_initial_state_proof_required')
        baseline=proof['remote_sha256']
        for path in ledger.rglob('*'):
            if not path.is_file():continue
            if path.is_symlink() or not path.resolve().is_relative_to(ledger):raise ValueError('ledger_link_escape')
            name=path.relative_to(ledger).as_posix();digest=sha(path)
            old=baseline.get('state/ledger/'+name)
            if old==digest:continue
            files['handoff-delta/'+name]=path;delta[name]=digest;previous[name]=old
    generated=output/'generated';generated.mkdir()
    native_pointer=dict(path='../../'+bundle['binding']['path'],sha256=bundle['binding']['sha256'])
    pointer_copy=generated/'active.json';pointer_copy.write_text(json.dumps(native_pointer,indent=2),encoding='utf-8')
    files['local-only/station-grasp-integration/active.json']=pointer_copy
    for name in ('vision-token.txt','control-token.txt'):
        value=secrets.token_hex(32)
        if name=='control-token.txt' and args.control_token_source:
            source=args.control_token_source.resolve()
            if not source.is_relative_to(ROOT/'local-only'):raise ValueError('existing_private_control_token_required')
            value=source.read_text().strip()
            if len(value)!=64 or any(v not in '0123456789abcdef' for v in value):raise ValueError('invalid_existing_control_token')
        path=generated/name;path.write_text(value,encoding='ascii');files[name]=path
    digests={name:sha(path) for name,path in files.items()}
    scope={name:digest for name,digest in digests.items()
           if not name.startswith('state/') and Path(name).suffix in ('.py','.json','.js','.css','.html')}
    scope_file=generated/'scope-frozen.json'
    scope_file.write_text(json.dumps(scope,indent=2),encoding='utf-8')
    files['scope-frozen.json']=scope_file;digests['scope-frozen.json']=sha(scope_file)
    source_hashes={str(path.relative_to(ROOT)):digests[name] for name,path in files.items()}
    source_hashes.update(bundle['original_sha256'])
    archive=output/'candidate.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for name,path in files.items():
            data=path.read_bytes()
            if hashlib.sha256(data).hexdigest()!=digests[name]:raise ValueError('source_changed_during_control_snapshot')
            info=tarfile.TarInfo(name);info.size=len(data);info.mtime=int(path.stat().st_mtime)
            info.mode=0o600 if name.endswith('token.txt') else 0o644
            tar.addfile(info,io.BytesIO(data))
    if any(sha(ROOT/name)!=digest for name,digest in source_hashes.items()) or any(p.exists() for p in protected):
        raise ValueError('source_changed_or_became_busy_during_control_snapshot')
    with archive.open('rb') as stream:archive_hash=hashlib.file_digest(stream,'sha256').hexdigest()
    manifest=dict(kind='board-native-control-stage-v1',status='staged_not_started',remote_root=str(remote),
        execution_started=False,physical_motion=False,files=len(files),bytes=sum(p.stat().st_size for p in files.values()),
        archive_sha256=archive_hash,source_sha256=source_hashes,
        remote_sha256={n:h for n,h in digests.items() if not n.endswith('token.txt')},
        original_authorization_preserved=True,qualification_values_preserved=True,
        original_ledger_bytes_preserved=True,arm_device=args.arm_device,voice_device=args.voice_device,
        binding=bundle['binding'],pc_ledger_source=str(ledger),source_pointer_sha256=sha(pointer))
    manifest['state_root']=str(state_root or remote/'state')
    manifest['ledger_delta_sha256']=delta;manifest['ledger_delta_previous_sha256']=previous
    manifest['source_pending_preserved']=args.handoff_paused
    (output/'stage.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    spec=importlib.util.spec_from_file_location('private_connection',args.connection_module.resolve())
    private=importlib.util.module_from_spec(spec);spec.loader.exec_module(private);client=private.connect()
    try:
        if state_root:
            check='import pathlib; p=pathlib.Path('+repr(str(state_root))+'); assert (p/"ledger").is_dir(),"existing_state_missing"'
            _,out,err=client.exec_command('python3 -c '+shlex.quote(check),timeout=10)
            if out.channel.recv_exit_status():raise RuntimeError(err.read().decode()[-500:])
        probe='import os; s=os.statvfs("/home/xilinx"); assert s.f_bavail*s.f_frsize>'+str(manifest['bytes']*2+200_000_000)+',"insufficient_board_disk"'
        _,out,err=client.exec_command('python3 -c '+shlex.quote(probe),timeout=10)
        if out.channel.recv_exit_status():raise RuntimeError(err.read().decode()[-500:])
        ftp=client.open_sftp()
        try:
            ftp.mkdir(str(remote),mode=0o700)
            ftp.put(str(archive),str(remote/'candidate.tar.gz'));ftp.chmod(str(remote/'candidate.tar.gz'),0o600)
        finally:ftp.close()
        script='''import pathlib,hashlib,tarfile,json,os
root=pathlib.Path('''+repr(str(remote))+''')
archive=root/'candidate.tar.gz'
h=hashlib.sha256()
with archive.open('rb') as stream:
 for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
assert h.hexdigest()=='''+repr(manifest['archive_sha256'])+'''
with tarfile.open(archive) as tar:
 for member in tar.getmembers():
  path=(root/member.name).resolve()
  assert member.isfile() and path.is_relative_to(root) and not pathlib.PurePosixPath(member.name).is_absolute()
 tar.extractall(root)
expected='''+repr({n:h for n,h in digests.items()})+'''
for name,digest in expected.items(): assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
state=pathlib.Path('''+repr(str(state_root or remote/'state'))+''')
changes='''+repr(delta)+'''
previous='''+repr(previous)+'''
for name,digest in changes.items():
 target=(state/'ledger'/name).resolve();assert target.is_relative_to((state/'ledger').resolve())
 if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest()==digest:continue
 old=previous[name]
 if old is None:assert not target.exists(),'new_ledger_target_already_exists:'+name
 else:
  assert target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest()==old,'ledger_changed_since_verified_import:'+name
for name,digest in changes.items():
 target=state/'ledger'/name
 if not (target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest()==digest):
  if target.exists():
   backup=state/'history-import-versions'/previous[name]/name
   backup.parent.mkdir(parents=True,exist_ok=True);backup.write_bytes(target.read_bytes())
  target.parent.mkdir(parents=True,exist_ok=True)
  data=(root/'handoff-delta'/name).read_bytes();assert hashlib.sha256(data).hexdigest()==digest
  target.write_bytes(data)
 source=root/'handoff-delta'/name
 os.utime(target,ns=(source.stat().st_atime_ns,source.stat().st_mtime_ns))
print(json.dumps({'files_verified':len(expected),'physical_motion':False}))
'''
        stdin,out,err=client.exec_command('timeout 240s nice -n 19 python3 -',timeout=260)
        stdin.write(script);stdin.flush();stdin.channel.shutdown_write()
        raw,error=out.read().decode(),err.read().decode()
        if out.channel.recv_exit_status():raise RuntimeError('board_native_stage_verification_failed:'+error[-800:])
        manifest['board_verification']=json.loads(raw)
        (output/'stage.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        print(json.dumps({k:manifest[k] for k in ('status','files','bytes','remote_root','board_verification')}),flush=True)
    finally:client.close()


if __name__=='__main__':main()
