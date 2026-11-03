"""Stage an isolated board-local vision candidate; never stops legacy services.

Default is file transfer only. --start refuses while a legacy camera/statistics
process owns the device. No actuator, ESP32 or wiring changes are supported.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path,PurePosixPath
import secrets
import shlex

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--connection-module',type=Path,required=True)
    parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--station-config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--remote-root',required=True)
    parser.add_argument('--start',action='store_true')
    parser.add_argument('--sudo-password-env',help='Optional existing account credential environment variable; never an inline password')
    args=parser.parse_args();build=args.build.resolve();out=args.output.resolve()
    if not out.is_relative_to(ROOT/'local-only'):raise ValueError('local_only_output_required')
    out.mkdir(parents=True,exist_ok=False)
    remote=PurePosixPath(args.remote_root)
    if remote.parent!=PurePosixPath('/home/xilinx') or not remote.name.startswith('km1-station-vision-'):
        raise ValueError('isolated_remote_candidate_directory_required')
    report=json.loads((build/'build-result.json').read_text(encoding='utf-8-sig'))
    assets=report['assets']
    for name,digest in assets.items():
        if name not in ('station_vision.bit','station_vision.hwh') or hashlib.sha256((build/'board'/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('build_assets_changed')
    config=json.loads(args.station_config.read_text(encoding='utf-8-sig'))
    roi=config['layout']['static_rois']
    if len(roi)<2:raise ValueError('two_reviewed_static_regions_required')
    regions=dict(source=config['source_roi'],static_a=roi[0],static_b=roi[1])
    for label in ('plastic','metal','paper'):regions['bin_'+label]=config['profiles'][label]['bin_roi']
    candidate=dict(schema_version='station-vision-v1',synthetic_fixture=False,backend='pl_board_local',
        bitfile='../board/station_vision.bit',assets=assets,token_env='KM1_STATION_VISION_TOKEN',
        image_size=config['image_size'],threshold=45,max_gap_s=.6,window_frames=3,regions=regions,
        purpose='diagnostic_candidate_not_mechanical_acceptance')
    (out/'station_vision.local.json').write_text(json.dumps(candidate,indent=2),encoding='utf-8')
    token=secrets.token_hex(32);(out/'token.txt').write_text(token,encoding='ascii')
    launcher='''from pathlib import Path
import os,sys
root=Path(__file__).resolve().parent
for entry in Path('/proc').iterdir():
 if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
 try:words=(entry/'cmdline').read_bytes().decode().split('\\0')
 except (OSError,UnicodeError):continue
 if any('src.sorting_statistics_server'==w or ('cv2.VideoCapture' in w and 'CAP_V4L2' in w) for w in words):
  raise RuntimeError('legacy_camera_or_statistics_still_active_no_overlay_changed')
os.chdir(root);sys.path.insert(0,str(root))
os.environ['KM1_STATION_VISION_TOKEN']=(root/'token.txt').read_text().strip()
sys.argv=['station_vision_server','--config',str(root/'config/station_vision.local.json'),'--load-overlay']
from src.station_vision_server import main
main()
'''
    (out/'launch.py').write_text(launcher,encoding='utf-8')
    files={str(remote/'launch.py'):out/'launch.py',str(remote/'token.txt'):out/'token.txt',
        str(remote/'config/station_vision.local.json'):out/'station_vision.local.json'}
    for name in ('station_vision.py','station_vision_server.py','station_vision_selftest.py'):
        files[str(remote/'src'/name)]=ROOT/'pynq_km1_vision/src'/name
    files[str(remote/'src/__init__.py')]=ROOT/'pynq_km1_vision/src/__init__.py'
    files[str(remote/'board/km1_station_vision.py')]=ROOT/'pynq_km1_vision/board/km1_station_vision.py'
    for name in assets:files[str(remote/'board'/name)]=build/'board'/name
    spec=importlib.util.spec_from_file_location('station_private_connection',args.connection_module.resolve())
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);client=module.connect()
    try:
        sftp=client.open_sftp()
        try:
            # A second staging operation cannot overwrite an existing candidate.
            sftp.mkdir(str(remote))
            for sub in ('src','board','config'):sftp.mkdir(str(remote/sub))
            for target,path in files.items():
                sftp.put(str(path),target)
                if target.endswith('token.txt'):sftp.chmod(target,0o600)
        finally:sftp.close()
        expected={target:hashlib.sha256(path.read_bytes()).hexdigest() for target,path in files.items()
            if not target.endswith('token.txt')}
        probe='import hashlib,json; paths='+repr(expected)+'; print(json.dumps([p for p,h in paths.items() if hashlib.sha256(open(p,"rb").read()).hexdigest()!=h]))'
        _,stdout,_=client.exec_command('python3 -c '+shlex.quote(probe),timeout=15)
        mismatches=json.loads(stdout.read().decode())
        if stdout.channel.recv_exit_status()!=0 or mismatches:raise RuntimeError('remote_staged_files_mismatch')
        manifest=dict(status='staged_not_started',remote_root=str(remote),assets=assets,
            source_sha256=expected,remote_hashes_verified=True,esp32_changed=False,mechanical_commands_sent=False,
            deployed_camera_changed=False,board_programmed=False)
        if args.start:
            cmd='. /etc/profile.d/xrt_setup.sh; nohup /usr/local/share/pynq-venv/bin/python3 -u '+shlex.quote(str(remote/'launch.py'))+' >'+shlex.quote(str(remote/'service.log'))+' 2>&1 < /dev/null & echo $!'
            sudo='sudo -S -p ""' if args.sudo_password_env else 'sudo -n'
            stdin,stdout,_=client.exec_command(sudo+' bash -c '+shlex.quote(cmd),timeout=10)
            if args.sudo_password_env:
                stdin.write(os.environ[args.sudo_password_env]+'\n');stdin.flush()
            pid=stdout.read().decode().strip()
            if stdout.channel.recv_exit_status()!=0:raise RuntimeError('privileged_board_start_failed')
            if not pid.isdigit():raise RuntimeError('candidate_start_outcome_unknown')
            manifest.update(status='start_requested_verify_health_required',pid=int(pid))
        (out/'stage-result.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        print(json.dumps(dict(status=manifest['status'],remote_root=str(remote),assets=assets)))
    finally:client.close()


if __name__=='__main__':main()
