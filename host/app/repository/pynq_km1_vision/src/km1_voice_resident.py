"""Persistent ASRPRO listener with explicit page-grant or spoken-command control."""
import json
from pathlib import Path
import queue
import threading
import time
import uuid

from .km1_voice import parse_request, reply_frame, result_reply
from .km1_voice_run import Frames
from .km1_voice_dashboard import VoiceRecords
from .monitor_tasks import TaskTelemetry


class ResidentVoice:
    def __init__(self, operations, folder, audit, port='auto', open_port=None, direct_commands=False):
        self.operations=operations
        self.direct_commands=direct_commands
        self.owner='voice-direct-'+uuid.uuid4().hex
        self.folder,self.audit=Path(folder),Path(audit)
        self.port_name=port
        self.open_port=open_port or self._open_serial
        self.records=VoiceRecords(self.folder)
        self.reader=TaskTelemetry(self.folder)
        self.done=threading.Event()
        self.replies=queue.Queue()
        self.pending=None
        self.thread=None
        self.record_lock=threading.Lock()
        self.operations.voice_connection(False)

    def _direct_request(self, material):
        """A fresh recognized utterance is the user's request for this batch.

        The operator explicitly selects this mode at service startup. Original
        station checks still run for every utterance; no page lease is issued.
        """
        with self.operations.lock:
            state=self.operations.status()
            if state['busy'] or (self.operations.worker is not None and self.operations.worker.is_alive()):
                return dict(reply='BUSY')
            if not self.operations.voice_connected or self.operations.closed.is_set():
                return dict(reply='OFFLINE')
            if state['blockers']:
                return dict(reply='REJECTED')
            request_id=uuid.uuid4().hex
            try:
                self.operations.command(dict(token=self.operations.token,owner=self.owner,
                    action='start',request_id=request_id,material=material,ready=True))
                # The worker cannot pass its first lock until this attribution
                # is set, so completion and disconnect handling keep their owner.
                self.operations.voice_operation=True
                self.record('spoken_command_accepted',operation_id=request_id,material=material,
                            page_confirmation_required=False)
                return dict(operation_id=request_id)
            except Exception:
                if self.operations.operation_id==request_id and self.operations.owner==self.owner:
                    self.operations.voice_operation=True
                    self.operations.voice_connection(False)
                return dict(reply='REJECTED')

    def heartbeat(self):
        if not self.direct_commands or not self.pending:return
        with self.operations.lock:
            if (self.operations.owner==self.owner and self.operations.operation_id==self.pending and
                    self.operations.state in self.operations.ACTIVE):
                self.operations.command(dict(token=self.operations.token,owner=self.owner,
                    action='heartbeat',operation_id=self.pending))

    def record(self, event, **values):
        with self.record_lock:
            self.audit.parent.mkdir(parents=True,exist_ok=True)
            with self.audit.open('a',encoding='utf-8') as out:
                out.write(json.dumps(dict(time=time.time(),event=event,**values),ensure_ascii=False)+'\n')

    def _open_serial(self):
        import serial
        from serial.tools import list_ports
        devices=[d for d in list_ports.comports() if (d.vid,d.pid)==(0x1a86,0x7522) and
                 (self.port_name=='auto' or d.device.casefold()==self.port_name.casefold())]
        if len(devices)!=1:
            raise ValueError('unique_reviewed_asrpro_port_required')
        port=serial.Serial(port=None,baudrate=115200,timeout=.1,write_timeout=1)
        port.dtr=False;port.rts=False;port.port=devices[0].device
        try:
            port.open()
            # Reconnects must not replay words buffered before a new page grant.
            if self.done.wait(.2):
                raise OSError('voice_service_closing')
            port.reset_input_buffer()
            return port
        except BaseException:
            port.close()
            raise

    def handle(self, packet):
        token=parse_request(packet)
        if token is None:return
        self.record('voice_request',request=token)
        if token=='STOP':
            try:
                self.operations.voice_stop()
                self.replies.put('STOPPED')
            except Exception:
                self.replies.put('OFFLINE')
        elif token=='REPORT':
            try:
                state=self.operations.status()
                if self.pending or (state['busy'] and state['state']!='voice_waiting'):
                    answer='BUSY'
                else:
                    report=self.reader.status({})
                    answer=result_reply(dict(mode='known_physical' if report.get('available') and
                        report.get('record_kind')=='physical' else None,
                        status=report.get('raw_status'),choice={'label':report.get('label')}))
                self.replies.put(answer)
            except Exception:
                self.replies.put('OFFLINE')
        elif self.pending:
            self.replies.put('BUSY')
        else:
            response=self._direct_request(token) if self.direct_commands else self.operations.voice_request(token)
            if response.get('operation_id'):
                self.pending=response['operation_id']
                self.record('voice_operation_bound',operation_id=self.pending,material=token)
            else:
                self.replies.put(response['reply'])

    def poll_result(self):
        if not self.pending:return
        completed=self.operations.voice_result(self.pending)
        if completed is None:return
        operation_id,self.pending=self.pending,None
        try:
            result=completed.get('result') or {}
            if completed['state']=='stopped':answer='STOPPED'
            elif completed['state']!='finished':answer='REJECTED'
            elif result.get('preview'):answer='REJECTED'
            elif result.get('status') in ('empty_scene','no_matching_material'):answer='EMPTY'
            elif result.get('status') in ('completed','selection_complete','completed_visual_evidence'):
                record=self.records.selected(result)
                answer=record.get('voice_transport_reply')
                if answer!='EMPTY':answer=result_reply(record)
            else:
                answer='REVIEW' if result.get('status')=='needs_review' else 'REJECTED'
        except Exception:
            answer='REVIEW'
        self.record('voice_operation_finished',operation_id=operation_id,reply=answer,
                    state=completed['state'])
        self.replies.put(answer)

    def start(self):
        if self.thread is not None:raise ValueError('voice_already_started')
        self.thread=threading.Thread(target=self._loop,daemon=True,name='resident-asrpro-listener')
        self.thread.start()

    def _loop(self):
        error_kind=None
        while not self.done.is_set():
            port=None
            try:
                port=self.open_port()
                self.operations.voice_connection(True)
                self.record('voice_connected',port=port.port,resident=True,armed=False,
                            direct_commands=self.direct_commands,page_confirmation_required=not self.direct_commands,
                            buffered_input_discarded=True,no_session_timeout=True)
                error_kind=None
                frames=Frames();last_packet=None;last_at=0;last_write=0
                # A reply from a disconnected device must not sound like the
                # result of a new utterance after reconnecting.
                while not self.replies.empty():self.replies.get_nowait()
                while not self.done.is_set():
                    self.heartbeat()
                    for packet in frames.feed(port.read(min(port.in_waiting or 1,4096))):
                        now=time.monotonic()
                        if packet==last_packet and now-last_at<2:
                            self.record('duplicate_ignored',request=parse_request(packet));continue
                        last_packet,last_at=packet,now
                        self.handle(packet)
                    self.poll_result()
                    while not self.replies.empty():
                        token=self.replies.get_nowait();payload=reply_frame(token)
                        if self.done.wait(max(0,.03-(time.monotonic()-last_write))):break
                        if port.write(payload)!=len(payload):raise OSError('partial_voice_reply_no_retry')
                        port.flush();last_write=time.monotonic()
                        self.record('voice_reply_written',token=token,audible_playback_not_verified=True)
            except Exception as exc:
                kind=type(exc).__name__+':'+str(exc)[:180]
                if kind!=error_kind:
                    self.record('voice_disconnected',reason=kind)
                    error_kind=kind
            finally:
                self.operations.voice_connection(False)
                if port is not None:port.close()
            if not self.done.is_set():self.done.wait(2)

    def close(self):
        self.done.set()
        self.operations.voice_connection(False)
        if self.thread is not None:self.thread.join(timeout=3)
