"""Bounded per-thread startup timing; timestamps use one monotonic clock."""
from contextlib import contextmanager
import copy
import threading
import time

_local = threading.local()
_lock = threading.RLock()
_last = None


def current():
    return getattr(_local, 'trace', None)


def publish(trace):
    global _last
    with _lock:
        _last = copy.deepcopy({k:v for k,v in trace.items() if not k.startswith('_')})


def status():
    with _lock:
        return copy.deepcopy(_last)


@contextmanager
def transaction(request_id=None, started=None):
    existing = current()
    if existing is not None:
        yield existing
        return
    trace = dict(request_id=request_id, origin='board_request_received' if started is not None else 'runner_entered',
                 stages={}, preparation_ms=None, first_uart_written_ms=None, first_uart_confirmed=False,
                 physical_motion_measured=False, _started=time.monotonic() if started is None else started, _stack=[])
    _local.trace = trace
    try:
        yield trace
    finally:
        trace['elapsed_ms'] = (time.monotonic()-trace['_started'])*1000
        publish(trace)
        _local.trace = None


@contextmanager
def span(name):
    trace = current()
    if trace is None:
        yield
        return
    started = time.perf_counter_ns()
    trace['_stack'].append([name, 0.])
    try:
        yield
    finally:
        elapsed = (time.perf_counter_ns()-started)/1e6
        _, children = trace['_stack'].pop()
        row = trace['stages'].setdefault(name, dict(calls=0, inclusive_ms=0., exclusive_ms=0.))
        row['calls'] += 1
        row['inclusive_ms'] += elapsed
        row['exclusive_ms'] += elapsed-children
        if trace['_stack']:
            trace['_stack'][-1][1] += elapsed
        publish(trace)


def boundary():
    trace = current()
    if trace is not None:
        trace['preparation_ms'] = (time.monotonic()-trace['_started'])*1000
        publish(trace)


def uart_written(stamp):
    trace = current()
    if trace is not None and trace['first_uart_written_ms'] is None:
        trace['first_uart_written_ms'] = (stamp-trace['_started'])*1000
        trace['first_uart_confirmed'] = True
        publish(trace)
