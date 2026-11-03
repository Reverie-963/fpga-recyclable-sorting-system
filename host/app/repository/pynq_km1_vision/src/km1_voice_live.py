"""Persistent voice transport to the currently valid local station controller.

No planner imports, independent robot binding, COM9 access or startup motion.
Every newly recognized command is a new bounded request to the shared API.
"""
import argparse
import hashlib
import json
from pathlib import Path
import threading
import time

from .km1_voice import parse_request, reply_frame
from .km1_voice_run import Frames, VoiceSession
from .km1_voice_dashboard import DashboardVoiceClient, VoiceRecords


class LiveControl:
    def __init__(self, root, record=lambda *a, **k:None, candidates=None, client_factory=DashboardVoiceClient):
        self.root=Path(root).resolve()
        self.record,self.candidates,self.client_factory=record,candidates,client_factory
        self.lock=threading.Lock();self.client=None;self.gate=None
        self.last_operation_id=None

    def client_record(self,event,**values):
        if event=='interface_request':
            with self.lock:self.last_operation_id=values.get('operation_id')
        self.record(event,**values)

    def controller_candidates(self):
        if self.candidates is not None:return self.candidates()
        import psutil
        candidates=[]
        expected=(self.root/'pynq_km1_vision').resolve()
        for process in psutil.process_iter(['pid','cmdline','create_time','cwd']):
            try:
                d=process.info;args=d['cmdline'] or []
                if (not d['cwd'] or Path(d['cwd']).resolve()!=expected or
                    'src.sorting_monitor_server' not in args or '--voice-resident' in args or '--port' not in args):continue
                port=int(args[args.index('--port')+1])
                if 1024<=port<=65535:candidates.append((d['create_time'],port))
            except (psutil.Error,ValueError,IndexError,OSError):continue
        ports=[]
        for _,port in sorted(candidates,reverse=True):
            if port not in ports:ports.append(port)
        return ['http://127.0.0.1:'+str(port) for port in ports[:12]]

    def configuration(self):
        pointer=self.root/'local-only/station-grasp-integration/active.json'
        selected=json.loads(pointer.read_text(encoding='utf-8-sig'))
        binding=(pointer.parent/selected['path']).resolve();raw=binding.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=selected['sha256']:raise ValueError('active_voice_binding_changed')
        ref=json.loads(raw)['files']['config'];path=(binding.parent/ref['path']).resolve();raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=ref['sha256']:raise ValueError('voice_configuration_changed')
        return path,json.loads(raw)

    def persist_stop(self):
        paths=[self.root/'local-only/STOP_SORTING']
        try:
            path,c=self.configuration()
            extra=[(path.parent/p).resolve() for p in c['known']['stop_files']]
            if any(not p.is_relative_to(self.root/'local-only') for p in extra):
                raise ValueError('voice_stop_path_outside_local_state')
            paths+=extra
        finally:
            for path in set(paths):
                path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('a',encoding='utf-8') as out:
                    out.write('ASRPRO live bridge requested stop; inspect physical state before resuming.\n')

    def select(self):
        config,c=self.configuration()
        records=VoiceRecords((config.parent/c['journal_dir']).resolve())
        errors=[]
        for endpoint in self.controller_candidates():
            try:
                client=self.client_factory(endpoint,records,self.persist_stop,self.client_record)
                status=client._status()
                stale={'station_entry_unavailable','station_evidence_changed','station_binding_or_configuration_unavailable'}
                if not status.get('configured') or stale.intersection(status.get('blockers',[])):
                    raise ValueError('controller_binding_stale')
                self.record('controller_selected',endpoint=endpoint,blockers=status.get('blockers',[]))
                return client
            except Exception as exc:errors.append(type(exc).__name__)
        raise ValueError('no_current_local_controller:'+','.join(errors))

    def run(self,material,execute,ready):
        if execute is not True or ready is not True:raise ValueError('live_voice_mode_requires_explicit_spoken_request')
        gate=threading.Event()
        with self.lock:self.gate=gate
        try:
            client=self.select()
            with self.lock:
                if gate.is_set():return dict(voice_transport_reply='STOPPED')
                self.client=client
            return client.run(material,execute,ready)
        finally:
            with self.lock:self.client=None;self.gate=None

    def latest(self):return self.select().latest()

    def stop(self):
        with self.lock:
            if self.gate:self.gate.set()
            client=self.client
        self.persist_stop()
        if client is not None:client.stop()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',default='auto')
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--log',type=Path,required=True)
    parser.add_argument('--quiet-status',action='store_true',help='Disable automatic console-state announcements')
    args=parser.parse_args()
    import serial
    from serial.tools import list_ports
    args.log.parent.mkdir(parents=True,exist_ok=True)
    log_lock=threading.Lock()
    with args.log.open('a',encoding='utf-8') as log:
        def record(event,**values):
            row=dict(time=time.time(),event=event,**values)
            with log_lock:
                log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush()
        control=LiveControl(args.root,record)
        try:
            control.select()  # Read-only validation; absence never authorizes motion.
        except Exception as exc:
            record('controller_unavailable',error_type=type(exc).__name__)
        session=VoiceSession(control.run,control.latest,control.stop,execute=True,ready=True,repeat_requests=True)
        from .km1_voice_status import SpeechQueue,ConsoleMonitor
        speech=SpeechQueue();monitor=ConsoleMonitor(control,speech,record)
        def collect_replies(explicit=False):
            while not session.replies.empty():
                speech.offer(session.replies.get_nowait(),control.last_operation_id,source='voice_reply',explicit=explicit)
        try:
            while True:
                port=None
                try:
                    devices=[d for d in list_ports.comports() if (d.vid,d.pid)==(0x1a86,0x7522) and
                        (args.port=='auto' or d.device.casefold()==args.port.casefold())]
                    if len(devices)!=1:raise ValueError('unique_reviewed_voice_port_required')
                    port=serial.Serial(port=None,baudrate=115200,timeout=.1,write_timeout=1)
                    port.dtr=False;port.rts=False;port.port=devices[0].device;port.open()
                    time.sleep(.2);port.reset_input_buffer()
                    frames=Frames();last_packet=None;last_at=0;last_write=0
                    while not session.replies.empty():session.replies.get_nowait()
                    speech.discard()
                    if monitor.thread is None and not args.quiet_status:monitor.start()
                    record('voice_connected',port=port.port,resident=True,direct_commands=True,
                        page_confirmation_required=False,controller_selection='latest_valid_local_manual_controller',
                        buffered_input_discarded=True,no_session_timeout=True)
                    while True:
                        for packet in frames.feed(port.read(min(port.in_waiting or 1,4096))):
                            now=time.monotonic()
                            if packet==last_packet and now-last_at<2:
                                record('duplicate_ignored',request=parse_request(packet));continue
                            last_packet,last_at=packet,now
                            record('voice_request',request=parse_request(packet))
                            session.handle(packet)
                            collect_replies(explicit=True)
                        collect_replies()
                        while True:
                            announcement=speech.next()
                            if announcement is None:break
                            token=announcement['token'];payload=reply_frame(token)
                            time.sleep(max(0,.03-(time.monotonic()-last_write)))
                            if port.write(payload)!=len(payload):raise OSError('partial_voice_reply_no_retry')
                            port.flush();last_write=time.monotonic()
                            record('voice_reply_written',token=token,source=announcement['source'],
                                operation_id=announcement['operation_id'],audible_playback_not_verified=True)
                except (serial.SerialException,OSError,ValueError) as exc:
                    record('voice_disconnected',error_type=type(exc).__name__,detail=str(exc)[:180])
                    if session.busy:
                        try:control.stop()
                        except Exception:record('stop_request_failed')
                    time.sleep(2)
                finally:
                    if port is not None:port.close()
        except KeyboardInterrupt:
            record('service_exit_requested')
        finally:
            monitor.close()
            if session.busy:
                try:control.stop()
                except Exception:record('stop_request_failed')
            if session.worker:session.worker.join()


if __name__=='__main__':main()
