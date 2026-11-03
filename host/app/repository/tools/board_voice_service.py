"""Check/start the frozen ASR bridge; never submit a robot operation.

Called by the local launcher after the native controller is available. The
package is provisioned separately; this tool neither uploads code nor installs
dependencies. An outdated bridge is replaced only while the station is idle.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]


def run(start=False):
    config=json.loads((ROOT/'local-only/board-control-integration/native-active.json').read_text())
    package=config.get('voice_library_package')
    if (not isinstance(package,dict) or not re.fullmatch(r'/opt/km1/voice/[a-zA-Z0-9_-]+',package.get('root','')) or
        not re.fullmatch('[a-f0-9]{64}',package.get('manifest_sha256',''))):raise ValueError('voice_package_not_provisioned')
    spec=importlib.util.spec_from_file_location('private_connection',config['connection_module'])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);client=module.connect()
    code=r'''import hashlib,json,os,pathlib,signal,subprocess,time,urllib.request
package=PACKAGE;start=START;root=pathlib.Path(package['root'])
manifest=(root/'manifest.json').read_bytes()
assert hashlib.sha256(manifest).hexdigest()==package['manifest_sha256'],'voice_manifest_changed'
for name,digest in json.loads(manifest).items():
 p=(root/name).resolve();assert p.is_relative_to(root) and hashlib.sha256(p.read_bytes()).hexdigest()==digest,'voice_source_changed'
def processes():
 rows=[]
 for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
  try:words=[s for s in p.read_bytes().decode().split('\0') if s]
  except (OSError,UnicodeError):continue
  rows.append((int(p.parent.name),words))
 return rows
native=[(pid,w) for pid,w in processes() if any(v.endswith('/runtime-launch.py') or v.endswith('/resume-control.py') for v in w) and '--control-token-file' in w and '--startup-preflight' not in w]
assert len(native)==1,'unique_native_controller_required'
_,words=native[0];args={words[i][2:]:words[i+1] for i in range(len(words)-1) if words[i].startswith('--')}
assert args['root'].startswith('/opt/km1/runtime/')
token=pathlib.Path(args['control-token-file']).read_text().strip()
def status():
 req=urllib.request.Request('http://127.0.0.1:18810/operation',headers={'Authorization':'Bearer '+token})
 with urllib.request.urlopen(req,timeout=5) as response:value=json.load(response)
 assert value['compute_location']=='pynq_ps' and value['board_root']==args['root']
 return value
before=status();voices=[(pid,w) for pid,w in processes() if any(v.endswith('/voice-launch.py') or v=='src.board_voice_host' for v in w)]
assert len(voices)<=1,'multiple_voice_bridges'
voice_ports=[]
for node in pathlib.Path('/sys/class/tty').glob('ttyUSB*'):
 device=(node/'device').resolve()
 usb=next((p for p in (device,*device.parents) if (p/'idVendor').is_file() and (p/'idProduct').is_file()),None)
 if usb is not None and ((usb/'idVendor').read_text().strip().lower(),(usb/'idProduct').read_text().strip().lower())==('1a86','7522'):
  voice_ports.append('/dev/'+node.name)
assert len(voice_ports)==1,'unique_asrpro_usb_identity_required'
voice_port=voice_ports[0]
current=bool(voices and str(root/'voice-launch.py') in voices[0][1] and '--root' in voices[0][1] and voices[0][1][voices[0][1].index('--root')+1]==args['root'] and '--voice-port' in voices[0][1] and voices[0][1][voices[0][1].index('--voice-port')+1]==voice_port)
if current:
 print(json.dumps(dict(status='voice_running',pid=voices[0][0],voice_service=before['voice_service'],physical_motion=False)))
elif not start:
 print(json.dumps(dict(status='voice_update_required' if voices else 'voice_stopped',physical_motion=False)))
else:
 assert not before['busy'] and before['board_ownership']['status']=='idle','voice_update_requires_idle_station'
 probe=subprocess.run(['/usr/local/share/pynq-venv/bin/python3','-B','-c','import serial'],capture_output=True)
 assert probe.returncode==0,'voice_dependency_unavailable'
 if voices:
  pid,w=voices[0];entry=next((v for v in w if v.endswith('/voice-launch.py')),None)
  allowed={str(root/'voice-launch.py'):json.loads(manifest)['voice-launch.py'],
   '/opt/km1/runtime/bin-zone-20261031-v1/voice-launch.py':'20fc048552d87d65cca1aa18ff1ce71f598da364f7c3f5a2edd968561b35904c'}
  assert entry in allowed and hashlib.sha256(pathlib.Path(entry).read_bytes()).hexdigest()==allowed[entry],'unrecognized_voice_bridge'
  latest=status();assert latest['operation_id']==before['operation_id'] and not latest['busy']
  os.kill(pid,signal.SIGTERM)
  deadline=time.monotonic()+8
  while pathlib.Path('/proc',str(pid)).exists() and time.monotonic()<deadline:time.sleep(.1)
  assert not pathlib.Path('/proc',str(pid)).exists(),'old_voice_still_running'
 state=pathlib.Path(args['state-root'])
 argv=['/usr/local/share/pynq-venv/bin/python3','-B','-u',str(root/'voice-launch.py'),'--root',args['root'],
  '--control-token-file',args['control-token-file'],'--state-root',str(state),'--voice-port',voice_port,'--controller-port','18810','--dimensions-port','18943']
 log_id='voice-library-'+str(time.time_ns())
 with (state/(log_id+'.stdout.log')).open('xb') as out,(state/(log_id+'.stderr.log')).open('xb') as err:
  proc=subprocess.Popen(argv,cwd=root,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
 deadline=time.monotonic()+12
 while time.monotonic()<deadline:
  assert proc.poll() is None,'voice_bridge_exited'
  after=status()
  if after['voice_service']['connected']:break
  time.sleep(.5)
 assert after['voice_service']['connected'],'voice_transport_not_connected'
 assert after['operation_id']==before['operation_id'] and after['state']==before['state'] and after['blockers']==before['blockers'] and not after['busy'],'station_changed_during_voice_update'
 print(json.dumps(dict(status='voice_started',pid=proc.pid,voice_service=after['voice_service'],physical_motion=False)))
'''.replace('PACKAGE',repr(package)).replace('START',repr(start))
    try:
        stdin,stdout,stderr=client.exec_command('/usr/local/share/pynq-venv/bin/python3 -B -',timeout=35)
        stdin.write(code);stdin.channel.shutdown_write();raw=stdout.read().decode();error=stderr.read().decode()
        if stdout.channel.recv_exit_status():raise RuntimeError('voice_service_check_failed:'+error[-800:])
        return json.loads(raw)
    finally:client.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--start',action='store_true');args=parser.parse_args()
    print(json.dumps(run(args.start)))
