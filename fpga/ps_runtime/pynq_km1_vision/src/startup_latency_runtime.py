"""Explicit board-host composition and truthful startup timing.

Does not grant Ready, clear a stop, change limits, bypass a source check, load PL
or send commands. A successful original send supplies the original write stamp.
"""
import copy
import threading
import time

_local = threading.local()
_lock = threading.RLock()
_last = None
_installed = False


def _publish(trace):
    global _last
    with _lock:
        _last = copy.deepcopy({k:v for k,v in trace.items() if k != '_started'})


def wrap_send(original):
    def send(self, data, duration, guard=None):
        stamp = original(self,data,duration,guard)
        trace = getattr(_local,'trace',None)
        if trace is not None and trace.get('first_uart_command_ms') is None:
            trace['first_uart_command_ms'] = (stamp-trace['_started'])*1000
            trace['first_uart_command_confirmed_written'] = True
            _publish(trace)
        return stamp
    return send


def install_runtime(expected_legacy_sha256, native_library, native_sha256, preserved_state=None):
    global _installed
    if _installed:
        raise RuntimeError('startup_latency_runtime_already_installed')
    from . import empty_reference_warp as warp
    from . import monitor_operation,monitor_prompted_operation,sorting_prompted_motion,arm_transport
    from .sorting_relocated_geometry import RelocatedGeometry
    warp.install(expected_legacy_sha256,native_library,native_sha256)
    fast_warp = RelocatedGeometry.warp_empty_reference
    def measured_warp(geometry,image):
        result = fast_warp(geometry,image)
        trace = getattr(_local,'trace',None)
        if trace is not None:
            trace['empty_reference_ms'] = geometry.last_empty_reference_timing['elapsed_ms']
            trace['empty_reference'] = dict(geometry.last_empty_reference_timing)
            _publish(trace)
        return result
    RelocatedGeometry.warp_empty_reference = measured_warp

    original_init = monitor_operation.Operations.__init__
    def initialize(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        self._startup_submitted = {}
        if preserved_state is not None:
            if preserved_state.get('busy') or preserved_state.get('state') not in ('idle','stopped','rejected','finished'):
                raise ValueError('quiescent_original_state_required')
            with self.lock:
                for key in ('state','reason','operation_id','owner','result','physical','material','active_action'):
                    if key in preserved_state:
                        setattr(self,key,copy.deepcopy(preserved_state[key]))
                if self.operation_id:
                    self.seen.add(self.operation_id)
                if self.state == 'stopped':
                    self.cancel.set()
    monitor_operation.Operations.__init__ = initialize

    original_command = monitor_operation.Operations.command
    def command(self,data):
        request = data.get('request_id') if isinstance(data,dict) else None
        selected = isinstance(data,dict) and data.get('action') in ('start','preview','home','arm','voice_arm')
        if selected and isinstance(request,str):
            self._startup_submitted[request] = time.monotonic()
        try:
            return original_command(self,data)
        except BaseException:
            if selected:
                self._startup_submitted.pop(request,None)
            raise
    monitor_operation.Operations.command = command

    original_worker = monitor_operation.Operations._run
    def worker(self,*args,**kwargs):
        request = self.operation_id
        trace = dict(_started=self._startup_submitted.pop(request,time.monotonic()),
            request_id=request,first_uart_command_ms=None,first_uart_command_confirmed_written=False,
            timing_origin='board_command_received',physical_motion_measured=False)
        _local.trace = trace
        try:
            return original_worker(self,*args,**kwargs)
        finally:
            trace['worker_elapsed_ms'] = (time.monotonic()-trace['_started'])*1000
            _publish(trace)
            _local.trace = None
    monitor_operation.Operations._run = worker

    original_cycle = sorting_prompted_motion.PromptedMotionController.cycle
    def cycle(controller,*args,**kwargs):
        trace = getattr(_local,'trace',None)
        if trace is not None:
            trace['preparation_ms'] = (time.monotonic()-trace['_started'])*1000
            _publish(trace)
        return original_cycle(controller,*args,**kwargs)
    sorting_prompted_motion.PromptedMotionController.cycle = cycle

    original_run = sorting_prompted_motion.run
    def run(*args,**kwargs):
        trace = getattr(_local,'trace',None)
        own_trace = trace is None
        if own_trace:
            trace = dict(_started=time.monotonic(),first_uart_command_ms=None,
                first_uart_command_confirmed_written=False,timing_origin='runner_entered',physical_motion_measured=False)
            _local.trace = trace
        try:
            result = original_run(*args,**kwargs)
            result['startup_timing'] = copy.deepcopy({k:v for k,v in trace.items() if k != '_started'})
            result['empty_reference_backend'] = warp.BACKEND
            return result
        finally:
            _publish(trace)
            if own_trace:
                _local.trace = None
    sorting_prompted_motion.run = run
    monitor_prompted_operation.run = run
    arm_transport.ArmConnection.send = wrap_send(arm_transport.ArmConnection.send)

    original_status = monitor_prompted_operation.PromptedOperations.status
    def status(self):
        result = original_status(self)
        with _lock:
            timing = copy.deepcopy(_last)
        result.update(empty_reference_backend=warp.BACKEND,
            empty_reference_projector='cpu_native_forward_projection',
            startup_native_library_sha256=native_sha256,last_startup_timing=timing)
        return result
    monitor_prompted_operation.PromptedOperations.status = status
    _installed = True
