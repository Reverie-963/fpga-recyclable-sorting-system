"""Local dashboard entry: one supervised attempt, optionally waiting for placement.

The calibrated executor remains the only owner of motion and physical journals.
Restart, disconnect, timeout and failures never re-arm an attempt.
"""
import hashlib
import json
from pathlib import Path
import secrets
import threading
import time

OPERATION_ROUTES={'/operation':None,'/operation/all':'AUTO','/operation/paper':'PAPER',
                  '/operation/metal':'METAL','/operation/plastic':'PLASTIC'}


def scoped_request(route,data):
    if route not in OPERATION_ROUTES or not isinstance(data,dict):raise ValueError('invalid_operation_route')
    fixed=OPERATION_ROUTES[route]
    if fixed is not None:
        if 'material' in data and data['material']!=fixed:raise ValueError('operation_material_conflicts_with_route')
        data=dict(data,material=fixed)
    return data


class StationBackend:
    def __init__(self, root):
        from .sorting_relocated_run import current_source_config
        from .sorting_known import bound_file
        self.root = Path(root)
        self.pointer = self.root/'local-only/station-grasp-integration/active.json'
        self.pointer_sha = hashlib.sha256(self.pointer.read_bytes()).hexdigest()
        record = json.loads(self.pointer.read_text(encoding='utf-8-sig'))
        self.binding = (self.pointer.parent/record['path']).resolve()
        self.sha = record['sha256']
        self._verify()
        binding = json.loads(self.binding.read_text(encoding='utf-8-sig'))
        self.unified = 'unified_policy' in binding['files']
        config = bound_file(self.binding.parent, binding['files']['config'])
        c, base = current_source_config(config), config.parent
        self.journal_dir = (base/c['journal_dir']).resolve()
        self.device_lease=(base/c['known']['device_lease']).resolve()
        self.stops = [(base/p).resolve() for p in c['known']['stop_files']]
        policy=json.loads(bound_file(self.binding.parent,binding['files']['unified_policy']).read_text(encoding='utf-8-sig')) if self.unified else {}
        self.decision_thresholds=policy.get('decision_thresholds',{})
        self.batch_verification_mode=policy.get('batch_verification_mode','per_item')
        self.empty_bin_max_changed_fraction=policy.get('empty_bin_max_changed_fraction',c['max_scene_changed_fraction'])
        self.min_fixed_markers=binding.get('motion_guard_policy',{}).get('min_fixed_markers',3)
        self.initial_marker_shift_px=binding.get('motion_guard_policy',{}).get('initial_max_displacement_px',.5)
        self.wait_time_scale=policy.get('wait_time_scale',1)
        self.skip_per_item_visual_verification=policy.get('skip_per_item_visual_verification',False)
        from .sorting_review_policy import review_factor
        self.review_tolerance_factor=review_factor(binding)
        self.bin_edge_overlap_limit=.05*self.review_tolerance_factor if policy.get('relax_bin_edge_overlap',False) else 0

    def _verify(self):
        if (hashlib.sha256(self.pointer.read_bytes()).hexdigest() != self.pointer_sha or
                hashlib.sha256(self.binding.read_bytes()).hexdigest() != self.sha):
            raise ValueError('station_binding_changed')

    def check(self):
        from .sorting_relocated_run import load_binding, integration_blockers
        self._verify()
        if self.unified:
            from .sorting_unified_run import run
            return run(self.binding,'preflight')['execution_blockers']
        b, c, base, *rest = load_binding(self.binding)
        return integration_blockers(b, c, base, rest[-1])

    def cycle(self, material, execute):
        from .km1_voice_station import run_voice_cycle
        self._verify()
        if self.unified:
            from .sorting_unified_run import run
            report=run(self.binding,'execute' if execute else 'preview',ready=execute,material=material)
            return report
        # Same class selection, marker/scene guards, leases, pending and receipts.
        return run_voice_cycle(self.binding, material, execute=execute, ready=execute,
                               transport='local_dashboard')

    def stop(self):
        # Stop paths are frozen at construction, so a changed binding cannot
        # prevent stopping an already running attempt. Never clear stop/pending.
        for path in self.stops:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a', encoding='utf-8') as out:
                out.write('Dashboard stop requested; inspect physical state before resuming.\n')

    def reset_requires_confirmation(self):
        return any(p.exists() for p in self.stops+[self.device_lease,self.journal_dir/'pending.json',
            self.journal_dir/'batch-pending.json',self.journal_dir/'controller.lock'])

    def reset(self,confirmed=False):
        self._verify()
        if self.unified:
            from .sorting_unified_run import load_unified
            bundle,_=load_unified(self.binding)
        else:
            from .sorting_relocated_run import load_binding
            bundle=load_binding(self.binding)
        if not any(Path(s['path']).name=='sorting_state_reset.py' for s in bundle[0]['implementation_sources']):
            raise ValueError('reset_implementation_not_bound')
        from .sorting_state_reset import reset_state
        return reset_state(self.journal_dir,self.device_lease,self.stops,confirmed,
            'Page operator confirms arm stopped, gripper empty, hands outside work area; reset does not infer jaw opening or a neutral pose.' if confirmed else
            'Idle UI reset; no protected station state existed.')

    def home(self):
        self._verify()
        from .sorting_neutral_return import return_neutral
        return return_neutral(self.binding,execute=True,confirmed=True)


class Operations:
    ACTIVE = ('checking', 'checking_preview', 'waiting_clear', 'standby', 'voice_waiting', 'executing', 'stopping')

    def __init__(self, backend=None, audit=None, poll_s=2, lease_s=12, standby_s=120, defer_preflight=False):
        self.backend, self.audit = backend, audit
        self.poll_s, self.lease_s, self.standby_s = poll_s, lease_s, standby_s
        self.defer_preflight = defer_preflight
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.RLock()
        self.state, self.result, self.reason = 'idle', None, None
        self.owner, self.operation_id = None, None
        self.seen = set()
        self.cancel = threading.Event()
        self.closed = threading.Event()
        self.worker = None
        self.last_heartbeat = 0
        self.deadline = 0
        self.physical = False
        self.material='AUTO'
        self.active_action=None
        self.voice_supported = False
        self.voice_connected = False
        self.voice_operation = False
        self.voice_completed = {}
        self.watchdog = threading.Thread(target=self._watch, daemon=True, name='dashboard-lease')
        self.watchdog.start()

    def _record(self, event, **values):
        if self.audit:
            self.audit.parent.mkdir(parents=True, exist_ok=True)
            with self.audit.open('a', encoding='utf-8') as out:
                out.write(json.dumps(dict(time=time.time(), event=event,
                    operation_id=self.operation_id, **values), ensure_ascii=False)+'\n')

    def status(self):
        with self.lock:
            blockers = []
            if self.backend is None:
                blockers = ['station_entry_unavailable']
            elif self.state not in self.ACTIVE:
                try:
                    blockers = self.backend.check()
                except Exception as exc:
                    blockers = ['station_evidence_changed' if str(exc).startswith('evidence_file_changed:')
                                else 'station_binding_or_configuration_unavailable']
            unified=getattr(self.backend,'unified',False)
            return dict(control_version=3 if unified else 1, batch_mode=unified,
                material_selection=self.material,operation_routes={k:v for k,v in OPERATION_ROUTES.items() if v},
                max_objects=6 if unified else 1, state=self.state, busy=self.state in self.ACTIVE,
                preflight_checks_deferred=self.defer_preflight,
                configured=self.backend is not None, blockers=blockers,
                decision_thresholds=getattr(self.backend,'decision_thresholds',{}),
                batch_verification_mode=getattr(self.backend,'batch_verification_mode','per_item'),
                skip_per_item_visual_verification=getattr(self.backend,'skip_per_item_visual_verification',False),
                wait_time_scale=getattr(self.backend,'wait_time_scale',1),
                review_tolerance_factor=getattr(self.backend,'review_tolerance_factor',1),
                bin_edge_overlap_limit=getattr(self.backend,'bin_edge_overlap_limit',0),
                reset_supported=callable(getattr(self.backend,'reset',None)),
                home_supported=callable(getattr(self.backend,'home',None)),
                empty_bin_max_changed_fraction=getattr(self.backend,'empty_bin_max_changed_fraction',None),
                min_fixed_markers=getattr(self.backend,'min_fixed_markers',3),
                initial_marker_shift_px=getattr(self.backend,'initial_marker_shift_px',.5),
                reset_requires_confirmation=(self.backend.reset_requires_confirmation()
                    if callable(getattr(self.backend,'reset_requires_confirmation',None)) else False),
                operation_id=self.operation_id, owner=self.owner, result=self.result,
                active_action=self.active_action,
                reason=self.reason, token=self.token, standby_seconds=self.standby_s,
                voice_service=dict(supported=self.voice_supported, connected=self.voice_connected,
                                   armed=self.state=='voice_waiting', resident=self.voice_supported),
                manual_review=(self.backend.review_status() if callable(getattr(self.backend,'review_status',None)) else dict(supported=False)),
                remaining_s=max(0, round(self.deadline-time.monotonic())) if self.state in
                    ('checking', 'waiting_clear', 'standby', 'voice_waiting') else None,
                single_attempt=True, hardware_emergency_stop=False)

    def command(self, data):
        with self.lock:
            if not isinstance(data, dict) or data.get('token') != self.token:
                raise ValueError('page_expired_reload')
            action = data.get('action')
            owner = data.get('owner')
            if not isinstance(owner, str) or not 8 <= len(owner) <= 80:
                raise ValueError('invalid_page_owner')
            if action == 'heartbeat':
                if owner != self.owner or data.get('operation_id') != self.operation_id:
                    raise ValueError('not_operation_owner')
                self.last_heartbeat = time.monotonic()
                return self.status()
            if action == 'stop':
                self._stop('operator_stop', global_stop=True)
                return self.status()
            if action == 'cancel':
                if owner != self.owner:
                    raise ValueError('not_operation_owner')
                if self.state not in ('checking', 'waiting_clear', 'standby', 'voice_waiting'):
                    raise ValueError('standby_not_active')
                self._stop('standby_cancelled')
                return self.status()
            if action not in ('preview', 'start', 'arm', 'voice_arm','reset','home','jaw_calibrate','review'):
                raise ValueError('invalid_action')
            request_id = data.get('request_id')
            if not isinstance(request_id, str) or not 8 <= len(request_id) <= 80:
                raise ValueError('invalid_request_id')
            if request_id in self.seen:
                raise ValueError('duplicate_request_no_retry')
            if (self.closed.is_set() or self.state in self.ACTIVE or
                    (self.worker is not None and self.worker.is_alive())):
                raise ValueError('operation_busy_or_closed')
            if len(self.seen) >= 1000:
                raise ValueError('session_request_limit_restart_required')
            if action=='review':
                if not callable(getattr(self.backend,'review',None)):raise ValueError('manual_review_not_supported')
                if data.get('expected_operation_id')!=self.operation_id:raise ValueError('manual_review_context_changed')
                if data.get('ready') is not True or data.get('review_confirmed') is not True:raise ValueError('manual_review_scene_confirmation_required')
                self._record('operator_manual_review_requested',review_epoch=data.get('review_epoch'),review_request_id=data.get('review_request_id'))
                self.seen.add(request_id)
                result=self.backend.review(epoch=data.get('review_epoch'),request_id=data.get('review_request_id'),confirmed=True)
                self.state,self.result,self.reason='idle',result,None
                self.active_action=None;self.owner,self.operation_id=None,None
                self.cancel=threading.Event();self.physical=False;self.voice_operation=False;self.deadline=0;self.last_heartbeat=0
                return self.status()
            if action=='reset':
                if not callable(getattr(self.backend,'reset',None)):raise ValueError('state_reset_not_supported')
                result=self.backend.reset(data.get('reset_confirmed') is True)
                self.seen.add(request_id)
                self._record('state_reset',reset=result,previous_operation_id=self.operation_id)
                self.state,self.result,self.reason='idle',result,None
                self.active_action=None
                self.owner,self.operation_id=None,None
                self.cancel=threading.Event();self.physical=False;self.voice_operation=False
                self.deadline=0;self.last_heartbeat=0
                return self.status()
            material = data.get('material', 'AUTO')
            if material not in ('AUTO', 'PAPER', 'METAL', 'PLASTIC'):
                raise ValueError('invalid_material')
            physical = action in ('start', 'arm', 'voice_arm','home','jaw_calibrate')
            if action=='jaw_calibrate' and not callable(getattr(self.backend,'jaw_calibrate',None)):
                raise ValueError('jaw_calibration_not_supported')
            if action=='home' and data.get('home_confirmed') is not True:
                raise ValueError('neutral_return_scene_confirmation_required')
            if action=='home' and not callable(getattr(self.backend,'home',None)):
                raise ValueError('neutral_return_not_supported')
            if physical and action!='home' and data.get('ready') is not True:
                raise ValueError('current_scene_ready_required')
            if self.backend is None:
                raise ValueError('station_entry_unavailable')
            blockers = [] if self.defer_preflight and action in ('start','preview','home','jaw_calibrate') else self.backend.check()
            if physical and blockers:
                raise ValueError('station_not_ready:'+','.join(blockers))
            if action == 'voice_arm' and not (self.voice_supported and self.voice_connected):
                raise ValueError('voice_service_not_connected')
            self.seen.add(request_id)
            self.owner, self.operation_id = owner, request_id
            self.result, self.reason, self.physical = None, None, physical and action!='voice_arm'
            self.voice_operation = False
            self.material=material
            self.active_action=action
            self.cancel = threading.Event()
            self.last_heartbeat = time.monotonic()
            self.deadline = self.last_heartbeat+self.standby_s
            if action == 'voice_arm':
                self.state = 'voice_waiting'
                self._record('voice_armed', material=material, physical=False, one_batch=True)
                return self.status()
            self.state = 'checking' if action == 'arm' else 'executing' if physical else 'checking_preview'
            self._record('accepted', action=action, material=material, physical=physical)
            self.worker = threading.Thread(target=self._run, args=(action, material),
                daemon=True, name='dashboard-one-attempt')
            self.worker.start()
            return self.status()

    def voice_connection(self, connected):
        with self.lock:
            self.voice_supported = True
            self.voice_connected = bool(connected)
            if not connected and (self.state=='voice_waiting' or
                                  (self.voice_operation and self.state in self.ACTIVE)):
                self._stop('voice_disconnected')

    def voice_request(self, material):
        """Consume exactly one live page grant; serial alone never grants Ready."""
        with self.lock:
            if material not in ('AUTO', 'PAPER', 'METAL', 'PLASTIC'):
                raise ValueError('invalid_voice_material')
            if self.state != 'voice_waiting':
                return dict(reply='BUSY' if self.state in self.ACTIVE else 'REJECTED')
            if (not self.voice_connected or time.monotonic()-self.last_heartbeat>self.lease_s or
                    time.monotonic()>=self.deadline or self.closed.is_set()):
                self._stop('voice_grant_expired')
                return dict(reply='REJECTED')
            try:
                blockers=self.backend.check()
                if blockers or (self.material!='AUTO' and material!=self.material):
                    raise ValueError('voice_material_or_station_not_ready')
            except Exception:
                self.state,self.reason='rejected','voice_conditions_rejected'
                self._record('voice_rejected', material=material, physical=False)
                return dict(reply='REJECTED')
            self.voice_operation=True
            self.physical=True
            self.material=material
            self.state='executing'
            self._record('voice_grant_consumed', material=material, one_batch=True)
            self.worker=threading.Thread(target=self._run,args=('start',material),
                                         daemon=True,name='one-voice-page-grant')
            self.worker.start()
            return dict(operation_id=self.operation_id)

    def voice_result(self, operation_id):
        with self.lock:
            return self.voice_completed.pop(operation_id, None)

    def voice_stop(self):
        with self.lock:
            self._stop('operator_voice_stop',global_stop=True)
            if self.reason in ('stop_write_failed_inspect_station','stop_audit_failed_inspect_station'):
                raise OSError('voice_stop_not_confirmed')

    def _stop(self, reason, global_stop=False):
        self.cancel.set()
        active = self.worker is not None and self.worker.is_alive()
        # Persist the stop before releasing the lock. Serial checks observe it
        # even if cancellation lands just before cycle(execute=True) starts.
        if self.backend is not None and (global_stop or (active and self.state == 'executing')):
            try:
                self.backend.stop()
            except Exception:
                self.state, self.reason = 'stopping', 'stop_write_failed_inspect_station'
                return
        self.state, self.reason = ('stopping' if active else 'stopped'), reason
        try:
            self._record('stop_requested', reason=reason, global_stop=global_stop)
        except OSError:
            self.reason = 'stop_audit_failed_inspect_station'

    def _watch(self):
        while not self.closed.wait(.25):
            with self.lock:
                if self.reason == 'stop_write_failed_inspect_station':
                    self._stop(self.reason, global_stop=True)
                    continue
                if self.state in self.ACTIVE and self.state != 'stopping':
                    if time.monotonic()-self.last_heartbeat > self.lease_s:
                        self._stop('page_disconnected')
                    elif self.state != 'executing' and time.monotonic() >= self.deadline:
                        self._stop('standby_timeout')

    def _run(self, action, material):
        try:
            if self.defer_preflight and action in ('start','preview','home','jaw_calibrate'):
                # Slow evidence/model/USB checks must never hold the lock used
                # by status, heartbeat and stop. Their result is still mandatory
                # before the existing fresh lease/cancellation checks below.
                with self.lock:
                    if self.cancel.is_set():return
                    self._record('preflight_started')
                started=time.monotonic()
                blockers=self.backend.check()
                with self.lock:
                    self._record('preflight_finished',elapsed_s=time.monotonic()-started,blockers=blockers)
                if self.physical and blockers:
                    raise ValueError('station_not_ready:'+','.join(blockers))
            if action == 'arm':
                cleared = False
                while not self.cancel.is_set():
                    report = self.backend.cycle(material, False)
                    with self.lock:
                        if self.cancel.is_set():
                            return
                        if report.get('execution_blockers'):
                            raise ValueError('station_preflight_blocked')
                        if report.get('status') in ('empty_scene','no_matching_material'):
                            cleared = True
                        elif report.get('status') != 'preview_only':
                            raise ValueError('standby_observation_rejected')
                        elif cleared:
                            break
                        self.state = 'standby' if cleared else 'waiting_clear'
                    if self.cancel.wait(self.poll_s):
                        return
            with self.lock:
                if self.cancel.is_set():
                    return
                # Recheck synchronously as well as in the watchdog: a slow
                # observation must not start motion after its page lease expires.
                if time.monotonic()-self.last_heartbeat > self.lease_s:
                    self._stop('page_disconnected')
                    return
                if (action == 'arm' or self.voice_operation) and time.monotonic() >= self.deadline:
                    self._stop('standby_timeout')
                    return
                self.state = 'executing' if self.physical else 'checking_preview'
                self._record('attempt_started', physical=self.physical)
            report = (self.backend.jaw_calibrate() if action=='jaw_calibrate' else
                      self.backend.home() if action=='home' else self.backend.cycle(material, self.physical))
            with self.lock:
                self.result = {k: report.get(k) for k in ('task_id', 'batch_id', 'status', 'reason',
                    'physical_motion_attempted','board_port_open_attempted','board_uart_write_attempted',
                    'completed_count','motion_completed_count','per_item_visual_failures','slot_checks',
                    'jaw_calibration','start_pwm','target_pwm')}
                if not self.result['slot_checks'] and report.get('items'):
                    self.result['slot_checks']=report['items'][-1].get('slot_checks',[])
                if report.get('final_verification'):
                    self.result['final_verification']={k:report['final_verification'].get(k) for k in ('status','reason')}
                self.result['remaining_count']=len(report.get('remaining',[]))
                self.result['material_selection']=material
                self.result['preview'] = not self.physical
                self.result['label'] = report.get('choice', {}).get('label')
                self.result['candidate_order']=[{k:choice.get(k) for k in
                    ('track_id','label','official_label','height_item_id','object_height_mm','grasp_z_mm','height_source','selection')}
                    for choice in report.get('candidate_order',[])]
                self._record('attempt_finished', result=self.result)
                if not self.cancel.is_set():
                    self.state = 'finished'
        except Exception as exc:
            with self.lock:
                self.reason = self.reason or 'attempt_rejected_check_task_record'
                self.state = 'rejected' if not self.cancel.is_set() else 'stopping'
                self.result = dict(status='rejected',reason=type(exc).__name__+':'+str(exc)[:1000],
                    preview=not self.physical,material_selection=material)
                self._record('attempt_error', reason=self.reason,error_type=type(exc).__name__,detail=str(exc)[:1000])
        finally:
            with self.lock:
                if self.cancel.is_set() and self.reason != 'stop_write_failed_inspect_station':
                    self.state = 'stopped'
                if self.voice_operation:
                    self.voice_completed[self.operation_id]=dict(state=self.state,result=self.result,reason=self.reason)
                    while len(self.voice_completed)>16:
                        self.voice_completed.pop(next(iter(self.voice_completed)))
                    self.voice_operation=False

    def close(self):
        with self.lock:
            if self.state=='voice_waiting' or (self.worker is not None and self.worker.is_alive()):
                self._stop('monitor_shutdown')
            self.closed.set()
        self.watchdog.join(timeout=2)
        if self.worker is not None:
            self.worker.join(timeout=5)


def default_operations(root):
    try:
        backend = StationBackend(root)
    except Exception:
        backend = None
    scale=getattr(backend,'wait_time_scale',1)
    return Operations(backend, Path(root)/'local-only/monitor-operations/events.jsonl',
                      poll_s=2*scale,lease_s=12*scale,standby_s=120*scale)
