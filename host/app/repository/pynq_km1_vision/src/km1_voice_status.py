"""Read-only console-state announcements using the deployed ASRPRO clips."""
from collections import deque
import threading
import time

from .km1_voice import REPLIES, result_reply


class SpeechQueue:
    """One UART writer, bounded events, no stale progress after a final result."""
    def __init__(self, clock=time.monotonic, gap_s=2.5):
        self.clock,self.gap_s=clock,gap_s
        self.lock=threading.Lock();self.pending=[];self.seen=set();self.order=deque()
        self.last_token=None;self.last_operation=None;self.last_at=-1e30

    def offer(self, token, operation_id=None, source='console', explicit=False):
        if token not in REPLIES:return False
        now=self.clock();key=(operation_id,token)
        with self.lock:
            if not explicit and operation_id is not None and key in self.seen:return False
            if (token==self.last_token and operation_id==self.last_operation and
                    now-self.last_at<self.gap_s):return False
            if any(r['key']==key for r in self.pending):return False
            if key not in self.seen:
                self.seen.add(key);self.order.append(key)
                if len(self.order)>512:self.seen.discard(self.order.popleft())
            if token!='BUSY':
                self.pending=[r for r in self.pending if not(r['operation_id']==operation_id and r['token']=='BUSY')]
            self.pending.append(dict(token=token,operation_id=operation_id,key=key,at=now,source=source))
            if len(self.pending)>16:self.pending.pop(0)
            return True

    def next(self):
        now=self.clock()
        with self.lock:
            self.pending=[r for r in self.pending if now-r['at']<=8]
            if not self.pending:return None
            urgent=next((i for i,r in enumerate(self.pending) if r['token'] in ('STOPPED','REVIEW')),None)
            index=urgent if urgent is not None else 0
            if now-self.last_at<(0.05 if urgent is not None else self.gap_s):return None
            row=self.pending.pop(index)
            self.last_token,self.last_operation,self.last_at=row['token'],row['operation_id'],now
            return row

    def discard(self):
        with self.lock:self.pending.clear()


class ConsoleStatus:
    def __init__(self):self.controller=None;self.signature=None;self.offline=False

    def observe(self, client, state):
        controller=(client.endpoint,client.token)
        result=state.get('result') or {};operation=state.get('operation_id')
        signature=(operation,state.get('state'),result.get('status'),state.get('reason'),
                   tuple(sorted(state.get('blockers') or [])))
        first=controller!=self.controller
        changed=signature!=self.signature
        self.controller,self.signature,self.offline=controller,signature,False
        if not changed:return None
        # New connection/controller: do not announce an old saved completion.
        if first and not state.get('busy'):return None
        blockers=set(state.get('blockers') or [])
        if state.get('reason')=='stop_write_failed_inspect_station':return 'OFFLINE'
        if 'operator_stop_requested' in blockers:return 'STOPPED'
        if state.get('state')=='stopped':
            return None if state.get('reason') in ('standby_cancelled','standby_timeout') else 'STOPPED'
        if state.get('busy'):
            return 'BUSY' if state.get('state') in ('executing','checking_preview','stopping') else None
        if result.get('status')=='needs_review' or blockers.intersection(
                ('existing_physical_trial_unresolved','unified_batch_needs_review')):return 'REVIEW'
        if state.get('state')=='rejected' or result.get('status') in ('rejected','blocked'):return 'REJECTED'
        if result.get('status') in ('empty_scene','no_matching_material'):return 'EMPTY'
        if result.get('preview') or result.get('status')=='preview_only':
            return 'PREVIEW' if result.get('status')=='preview_only' else None
        if result.get('status') in ('completed','selection_complete','completed_visual_evidence'):
            try:
                report=client.records.selected(result)
                return report.get('voice_transport_reply') if report.get('voice_transport_reply')=='EMPTY' else result_reply(report)
            except Exception:return 'REVIEW'
        return None

    def lost(self):
        if self.offline:return None
        self.offline=True
        return 'OFFLINE'


class ConsoleMonitor:
    """Passive polling only. Never issues an operation or a heartbeat."""
    def __init__(self, control, speech, record, interval_s=1):
        self.control,self.speech,self.record=control,speech,record
        self.interval_s=interval_s;self.done=threading.Event();self.thread=None
        self.state=ConsoleStatus();self.client=None;self.selected_at=0

    def start(self):
        self.thread=threading.Thread(target=self._loop,daemon=True,name='console-status-speech')
        self.thread.start()

    def sample(self):
        with self.control.lock:active=self.control.client
        if active is not None:client=active
        else:
            if self.client is None or time.monotonic()-self.selected_at>10:
                self.client=self.control.select();self.selected_at=time.monotonic()
            client=self.client
        state=client._status()
        if self.done.is_set():return
        if set(state.get('blockers') or []).intersection(
                ('station_entry_unavailable','station_evidence_changed','station_binding_or_configuration_unavailable')):
            self.client=None;raise ValueError('console_controller_stale')
        token=self.state.observe(client,state)
        if token and self.speech.offer(token,state.get('operation_id'),source='console'):
            self.record('console_announcement_queued',token=token,operation_id=state.get('operation_id'),
                        controller=client.endpoint,state=state.get('state'),result_status=(state.get('result') or {}).get('status'))

    def _loop(self):
        while not self.done.is_set():
            try:self.sample()
            except Exception:
                if self.done.is_set():break
                self.client=None;token=self.state.lost()
                if token and self.speech.offer(token,source='console'):
                    self.record('console_announcement_queued',token=token,reason='control_connection_unavailable')
            self.done.wait(self.interval_s)

    def close(self):
        self.done.set()
        if self.thread:self.thread.join(timeout=2)
