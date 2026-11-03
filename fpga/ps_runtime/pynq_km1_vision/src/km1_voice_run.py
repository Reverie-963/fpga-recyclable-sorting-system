"""ASRPRO USB request/result bridge. All audio stays on the arm's voice module.

Default mode only previews. Explicit execute+ready grants at most ONE attempt.
No ESP32 firmware changes and no second owner opens the motion UART.
"""
import argparse
import json
from pathlib import Path
import queue
import threading
import time

from .km1_voice import parse_request, reply_frame, result_reply


class VoiceSession:
    def __init__(self, run_sort, get_report, request_stop, execute=False, ready=False, repeat_requests=False):
        if execute and not ready:
            raise ValueError('explicit_ready_required_for_voice_motion')
        self.run_sort,self.get_report,self.request_stop=run_sort,get_report,request_stop
        self.execute,self.ready=execute,ready
        self.repeat_requests=repeat_requests
        self.attempt_available=bool(execute and ready)
        self.busy=False
        self.stopped=False
        self.replies=queue.Queue()
        self.worker=None

    def handle(self, frame):
        token=parse_request(frame)
        if token is None:
            return
        if token=='STOP':
            self.stopped=True
            self.attempt_available=bool(self.repeat_requests)
            # The existing motion guard checks this file even during waits.
            try:
                self.request_stop()
                self.replies.put('STOPPED')
            except Exception:
                self.replies.put('OFFLINE')
        elif token=='REPORT':
            try:
                self.replies.put('BUSY' if self.busy else result_reply(self.get_report()))
            except (OSError,ValueError,KeyError,TypeError):
                self.replies.put('NO_RESULT')
        elif self.busy:
            self.replies.put('BUSY')
        elif (self.stopped and not self.repeat_requests) or (self.execute and not self.attempt_available):
            self.replies.put('REJECTED')
        else:
            self.busy=True
            self.attempt_available=False
            def work():
                try:
                    report=self.run_sort(token,self.execute,self.ready)
                    if not self.execute and isinstance(report,dict):
                        report=dict(report,mode='preview')
                    # A preview cannot claim physical success.
                    transport_reply=report.get('voice_transport_reply') if isinstance(report,dict) else None
                    # Transport replies cannot select a physical-success clip.
                    if transport_reply in ('BUSY','OFFLINE','EMPTY','REJECTED','PREVIEW','STOPPED'):
                        self.replies.put(transport_reply)
                    else:
                        self.replies.put('PREVIEW' if not self.execute and isinstance(report,dict) and
                                         report.get('status')=='preview_only' else result_reply(report))
                except Exception:
                    self.replies.put('REJECTED')
                finally:
                    if self.repeat_requests:self.attempt_available=True
                    self.busy=False
            self.worker=threading.Thread(target=work,name='one-voice-sorting-attempt')
            self.worker.start()


class Frames:
    """Bounded framing: ignore logs; dispatch only exact whole @KSV1 packets."""
    def __init__(self):
        self.pending=bytearray()

    def feed(self, data):
        output=[]
        for c in data:
            if c==ord('@'):
                self.pending=bytearray(b'@')
            elif self.pending:
                self.pending.append(c)
                if len(self.pending)>32:
                    self.pending.clear()
                elif c==ord('!'):
                    packet=bytes(self.pending)
                    if parse_request(packet):output.append(packet)
                    self.pending.clear()
        return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',default='auto')
    p.add_argument('--binding',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--ready',action='store_true')
    p.add_argument('--listen-only',action='store_true',help='Observe actual voice frames without replies, preview or motion')
    p.add_argument('--seconds',type=int,default=300)
    p.add_argument('--log',type=Path,required=True)
    p.add_argument('--dashboard-url',help='Use the live shared material HTTP interfaces instead of the legacy direct entry')
    args=p.parse_args()
    if not 1<=args.seconds<=3600 or (args.listen_only and (args.execute or args.ready)):
        p.error('invalid bounded session options')
    from .sorting_relocated_run import load_binding
    from .km1_voice_station import run_voice_cycle
    from .monitor_tasks import TaskTelemetry
    _,c,base,*_=load_binding(args.binding.resolve())
    import serial
    from serial.tools import list_ports
    selected=[d for d in list_ports.comports() if (d.vid,d.pid)==(0x1a86,0x7522) and
              (args.port=='auto' or d.device.casefold()==args.port.casefold())]
    if len(selected)!=1:
        raise ValueError('reviewed_asrpro_ch340k_port_required')
    args.port=selected[0].device
    if args.port.casefold()==c['arm']['port'].casefold():
        raise ValueError('voice_port_must_not_be_motion_port')
    reader=TaskTelemetry((base/c['journal_dir']).resolve())
    def latest():
        d=reader.status({})
        return dict(mode='known_physical' if d.get('available') and d.get('record_kind')=='physical' else None,
                    status=d.get('raw_status'),choice={'label':d.get('label')})
    def stop():
        for name in c['known']['stop_files']:
            path=(base/name).resolve()
            path.parent.mkdir(parents=True,exist_ok=True)
            # Never delete/clear an existing stop or pending record.
            with path.open('a',encoding='utf-8') as out:out.write('ASRPRO requested stop; inspect physical state before resuming.\n')
    dashboard=None
    if args.dashboard_url and not args.listen_only:
        from .km1_voice_dashboard import DashboardVoiceClient,VoiceRecords
        dashboard=DashboardVoiceClient(args.dashboard_url,VoiceRecords((base/c['journal_dir']).resolve()),stop)
    session=VoiceSession(dashboard.run if dashboard else
                         lambda token,execute,ready:run_voice_cycle(args.binding.resolve(),token,execute,ready),
                         dashboard.latest if dashboard else latest,dashboard.stop if dashboard else stop,args.execute,args.ready)
    args.log.parent.mkdir(parents=True,exist_ok=True)
    port=serial.Serial(port=None,baudrate=115200,timeout=.1,write_timeout=1)
    port.dtr=False
    port.rts=False
    port.port=args.port
    frames=Frames()
    end=time.monotonic()+args.seconds
    last_packet=None
    last_at=0
    last_write=0
    with args.log.open('a',encoding='utf-8') as log:
        def record(event,**kwargs):
            value=dict(time=time.time(),event=event,**kwargs)
            log.write(json.dumps(value,ensure_ascii=False)+'\n');log.flush()
            print(json.dumps(value,ensure_ascii=False),flush=True)
        if dashboard is not None:dashboard.record=record
        try:
            port.open()
            # Do not treat UART bytes accumulated before this session as new
            # operator requests, especially when explicit one-shot execution is armed.
            time.sleep(.2)
            port.reset_input_buffer()
            record('voice_connected',port=args.port,listen_only=args.listen_only,execute=args.execute,
                   one_attempt_only=True,buffered_input_discarded=True,binding=str(args.binding.resolve()),
                   dashboard_url=args.dashboard_url,shared_material_interfaces=bool(dashboard))
            while time.monotonic()<end:
                for packet in frames.feed(port.read(min(port.in_waiting or 1,4096))):
                    now=time.monotonic()
                    if packet==last_packet and now-last_at<2:
                        record('duplicate_ignored',frame=packet.decode());continue
                    last_packet,last_at=packet,now
                    record('voice_request',frame=packet.decode())
                    if not args.listen_only:session.handle(packet)
                while not session.replies.empty():
                    token=session.replies.get_nowait()
                    time.sleep(max(0,.03-(time.monotonic()-last_write)))
                    payload=reply_frame(token)
                    if port.write(payload)!=len(payload):raise IOError('partial_voice_reply_no_retry')
                    port.flush();last_write=time.monotonic()
                    record('voice_reply_written',token=token,audible_playback_not_verified=True)
        except KeyboardInterrupt:
            record('session_interrupted')
        finally:
            if session.busy:
                try:
                    (dashboard.stop if dashboard else stop)()
                except Exception:
                    record('stop_request_failed',physical_state_requires_review=True)
                record('session_stopping',mechanical_emergency_stop_claimed=False)
            if session.worker is not None:session.worker.join()
            port.close()
            record('voice_disconnected')


if __name__=='__main__':main()
