"""Display-only operation timing. Hooks retain original guards and return values."""
import copy
from functools import wraps
import threading
import time


class Timeline:
    def __init__(self, operation_id, action, clock=time.monotonic):
        self.clock, self.started = clock, clock()
        self.lock = threading.RLock()
        self.value = dict(schema='operation-progress-v1', operation_id=operation_id,
                          action=action, events=[], completed_commands=0,
                          total_commands=0, sound_confirmed=False)
        self.ended = None
        self.mark('submitted')

    def mark(self, phase, **values):
        with self.lock:
            if self.ended is not None:
                return
            elapsed = max(0, self.clock()-self.started)
            if self.value.get('phase') != phase:
                self.value['events'].append(dict(phase=phase, elapsed_s=elapsed))
            self.value.update(phase=phase, **values)

    def finish(self, state, result):
        with self.lock:
            self.mark('ended', outcome=(result or {}).get('status') or state)
            self.ended = self.clock()

    def snapshot(self):
        with self.lock:
            result = copy.deepcopy(self.value)
            result.update(elapsed_s=max(0, (self.ended if self.ended is not None else self.clock())-self.started),
                          terminal=self.ended is not None)
            return result


def install_runtime():
    """Install once, before serving. This module never submits an operation."""
    from .monitor_operation import Operations
    from .sorting_io import Motion
    from .arm_transport import ArmConnection
    from .sorting_unified import UnifiedController
    from . import sorting_batch_verification
    if getattr(Operations, '_progress_installed', False):
        return
    Operations._progress_installed = True
    context = threading.local()

    def current():
        return getattr(context, 'timeline', None)

    record, status, run = Operations._record, Operations.status, Operations._run

    @wraps(record)
    def recorded(self, event, **values):
        result = record(self, event, **values)
        if event in ('accepted', 'voice_armed'):
            self._display_timeline = Timeline(self.operation_id, self.active_action)
        timeline = getattr(self, '_display_timeline', None)
        if timeline and event == 'preflight_started':
            timeline.mark('checking')
        return result

    @wraps(status)
    def displayed(self):
        result = status(self)
        timeline = getattr(self, '_display_timeline', None)
        if timeline and timeline.value['operation_id'] == result.get('operation_id'):
            result['progress'] = timeline.snapshot()
        return result

    @wraps(run)
    def running(self, *args, **kwargs):
        timeline = getattr(self, '_display_timeline', None)
        context.timeline = timeline
        context.route_count=0
        context.command_total=0
        if timeline:
            timeline.mark('checking')
        try:
            return run(self, *args, **kwargs)
        finally:
            if timeline:
                timeline.finish(self.state, self.result)
            context.timeline = None

    Operations._record, Operations.status, Operations._run = recorded, displayed, running
    opened, sent, waited = Motion.open, ArmConnection.send, Motion.wait
    verifying = UnifiedController.verify_within_window
    batch_verifying=sorting_batch_verification.verify_batch_bin_occupied

    @wraps(opened)
    def opening(self, plan):
        timeline = current()
        if timeline:
            context.steps = [name for name, _, _ in plan]
            context.index = 0
            context.command_offset=getattr(context,'command_total',0)
            context.command_total=context.command_offset+len(plan)
            context.route_count=getattr(context,'route_count',0)+1
            timeline.mark('connecting' if context.route_count==1 else 'moving',total_commands=context.command_total)
        result = opened(self, plan)
        if timeline:
            if context.route_count==1:timeline.mark('ready')
        return result

    @wraps(sent)
    def sending(self, data, duration, guard=None):
        def checked():
            if guard:
                guard()
            timeline = current()
            if timeline:
                index = getattr(context, 'index', 0)
                steps = getattr(context, 'steps', [])
                timeline.mark('moving', command_index=getattr(context,'command_offset',0)+index+1,
                              command_name=steps[index] if index < len(steps) else '')
        return sent(self, data, duration, guard=checked)

    @wraps(waited)
    def waiting(self, seconds):
        result = waited(self, seconds)
        timeline = current()
        if timeline:
            context.index = getattr(context, 'index', 0)+1
            timeline.mark('moving', completed_commands=getattr(context,'command_offset',0)+context.index)
        return result

    @wraps(verifying)
    def verify(self, *args, **kwargs):
        timeline = current()
        if timeline:
            timeline.mark('verifying')
        return verifying(self, *args, **kwargs)

    Motion.open, ArmConnection.send, Motion.wait = opening, sending, waiting
    UnifiedController.verify_within_window = verify
    @wraps(batch_verifying)
    def verify_batch(*args,**kwargs):
        timeline=current()
        if timeline:timeline.mark('verifying')
        return batch_verifying(*args,**kwargs)
    sorting_batch_verification.verify_batch_bin_occupied=verify_batch
