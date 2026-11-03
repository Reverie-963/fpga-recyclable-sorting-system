"""Pinned route-compiler composition; no motion authorization is added here."""
import copy
import hashlib
import http.client
import inspect
import json
import math
from pathlib import Path
import secrets
import sys
import threading
import time

from .motion_path import checked_segments
from .motion_path_service import encoded, input_digest

_installed = False
_local = threading.local()
_last = None
_startup_proof = None
_lock = threading.Lock()
# These exact sources supplied the nominal math and compiler-equivalence proof.
AUDITED_PLAN_SHA256 = 'b23d8afa5c2689feeb577507941131ea636fa0f0f9c1427f703f0fa557eff340'
AUDITED_MODEL_SHA256 = '70d5c958aa8e19750a5feb2528306dc62dc2fb7e58438df83d7eeb5a2d5657c9'


class MotionBackendFault(RuntimeError):
    """Must escape the compiler's ordinary ValueError/unreachable-chain path."""


class LocalMotionClient:
    backend = 'pl_motion_path_v1'

    def __init__(self, token_file, assets, port=18840, timeout_s=3):
        if type(port) is not int or not 1024 <= port <= 65535 or not 0 < timeout_s <= 10:
            raise ValueError('invalid_local_motion_endpoint')
        self.port, self.timeout = port, timeout_s
        self.token = Path(token_file).read_text().strip()
        if len(self.token) < 32:
            raise ValueError('motion_service_token_too_short')
        self.assets = dict(assets)
        self.faulted = False
        self.lock = threading.Lock()

    def request(self, path, value=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=self.timeout)
        try:
            body = None if value is None else encoded(value)
            connection.request('GET' if body is None else 'POST', path, body,
                {'Authorization': 'Bearer '+self.token, 'Content-Type': 'application/json'})
            response = connection.getresponse()
            size = int(response.getheader('Content-Length', '0'))
            if response.status != 200 or not 1 <= size <= 40000:
                raise MotionBackendFault('motion_service_unavailable_or_faulted')
            data = response.read(size+1)
            if len(data) != size:
                raise MotionBackendFault('motion_service_truncated_response')
            return json.loads(data)
        finally:
            connection.close()

    def readiness(self):
        result = self.request('/health')
        if (result.get('ready') is not True or result.get('faulted') is not False or
                result.get('assets') != self.assets or result.get('backend') != self.backend):
            raise MotionBackendFault('motion_service_identity_or_readiness_changed')
        return result

    def evaluate(self, segments):
        segments = checked_segments(segments)
        with self.lock:
            if self.faulted:
                raise MotionBackendFault('motion_client_fault_latched')
            started = time.perf_counter_ns()
            nonce, digest = secrets.token_hex(16), input_digest(segments)
            try:
                result = self.request('/evaluate', dict(segments=segments, nonce=nonce,
                    input_sha256=digest, assets=self.assets))
                if (result.get('nonce') != nonce or result.get('input_sha256') != digest or
                        result.get('assets') != self.assets or result.get('backend') != self.backend or
                        result.get('physical_motion') is not False or len(result['segments']) != len(segments)):
                    raise ValueError('motion_service_response_identity_changed')
                for row, (start, end) in zip(result['segments'], segments):
                    lower, upper = row['minimum_z_lower_mm'], row['minimum_z_upper_mm']
                    if (type(lower) not in (int, float) or type(upper) not in (int, float) or
                            not math.isfinite(lower) or not math.isfinite(upper) or
                            not -248.01 <= lower <= upper <= 448.01 or not .0039 <= upper-lower <= .004 or
                            row['samples'] != 101 or type(row['approximate_minimum_index']) is not int or
                            not 0 <= row['approximate_minimum_index'] <= 100 or
                            row['pwm_l1_cost'] != sum(abs(a-b) for a, b in zip(start, end))):
                        raise ValueError('motion_service_invalid_result')
                result['timing']['client_total_ms'] = (time.perf_counter_ns()-started)/1e6
                return result
            except Exception as exc:
                self.faulted = True
                raise MotionBackendFault('motion_batch_failed_no_cpu_fallback') from exc


def audit_route(initial_pwm, steps, floor_mm, backend, tip_rz):
    started = time.perf_counter_ns()
    state, segments = list(initial_pwm), []
    for step in steps:
        target, text = list(state), step['command']
        for offset in range(1, len(text)-1, 15):
            word = text[offset:offset+15]
            target[int(word[1:4])] = int(word[5:9])
        segments.append([state, target])
        state = target
    try:
        report = backend.evaluate(segments)
        if len(report['segments']) != len(segments):
            raise ValueError('motion_batch_wrong_result_count')
    except Exception as exc:
        raise MotionBackendFault('route_height_hardware_failed') from exc
    audit, boundary, decisions = [], [], []
    for i, (step, segment, row) in enumerate(zip(steps, segments, report['segments'])):
        lower, upper = row['minimum_z_lower_mm'], row['minimum_z_upper_mm']
        nominal = lower
        if lower >= floor_mm:
            passed = True
        elif upper < floor_mm:
            passed = False
        else:
            boundary.append(i)
            a, b = segment
            nominal = min(tip_rz([x+(y-x)*j/100 for x, y in zip(a, b)])[1] for j in range(101))
            passed = nominal >= floor_mm
        decisions.append(passed)
        audit.append(dict(name=step['name'], start_pwm=segment[0], target_pwm=segment[1],
            nominal_z_min_mm=nominal,
            nominal_z_min_kind='cpu_exact_101_points' if i in boundary else 'conservative_pl_lower_bound',
            minimum_z_lower_mm=lower, minimum_z_upper_mm=upper))
    measured = dict(backend=report['backend'], segments=len(segments), boundary_cpu_segments=boundary,
        timing=report['timing'], audit_elapsed_ms=(time.perf_counter_ns()-started)/1e6,
        floor_mm=floor_mm, segment_pass=decisions, physical_motion=False)
    _local.report = measured
    global _last
    with _lock:
        _last = copy.deepcopy(measured)
    if not all(decisions):
        raise ValueError('unified_nominal_height_floor')
    return state, audit


def equivalent_route(reference, actual):
    """Retain every route field except explicitly bounded height diagnostics."""
    actual = copy.deepcopy(actual)
    actual.pop('motion_path_backend')
    actual.pop('motion_path_last_candidate')
    for old, new in zip(reference['nominal_interpolation'], actual['nominal_interpolation']):
        if not new['minimum_z_lower_mm'] <= old['nominal_z_min_mm'] <= new['minimum_z_upper_mm']:
            raise MotionBackendFault('original_minimum_not_in_pl_interval')
        for key in ('nominal_z_min_kind', 'minimum_z_lower_mm', 'minimum_z_upper_mm'):
            new.pop(key)
        new['nominal_z_min_mm'] = old['nominal_z_min_mm']
    if actual != reference:
        raise MotionBackendFault('accelerated_route_semantics_changed')


def install_runtime(expected_plan_sha256, expected_model_sha256, startup_fixture=None, **client_options):
    global _installed, _startup_proof
    if _installed:
        raise RuntimeError('motion_runtime_already_installed')
    if expected_plan_sha256 != AUDITED_PLAN_SHA256 or expected_model_sha256 != AUDITED_MODEL_SHA256:
        raise RuntimeError('motion_compiler_or_model_not_in_audited_revision')
    from . import sorting_unified_plan as plan
    original = plan.compile_route
    for obj, expected in ((plan, expected_plan_sha256), (inspect.getmodule(plan.tip_rz), expected_model_sha256)):
        if hashlib.sha256(Path(obj.__file__).read_bytes()).hexdigest() != expected:
            raise RuntimeError('motion_compiler_or_nominal_model_source_changed')
    client = LocalMotionClient(**client_options)
    client.readiness()
    source = inspect.getsource(original)
    begin = "            state, audit = list(c['known']['initial_pwm']), []\n"
    end = "            need(state == c['known']['initial_pwm'], 'unified_route_must_end_neutral_open')"
    if source.count(begin) != 1 or source.count(end) != 1:
        raise RuntimeError('motion_audit_source_layout_changed')
    start, finish = source.index(begin), source.index(end)
    old_block = source[start:finish]
    if 'for i in range(101)' not in old_block or 'unified_nominal_height_floor' not in old_block:
        raise RuntimeError('motion_audit_source_semantics_changed')
    source = source[:start]+"            state, audit = _motion_audit(c['known']['initial_pwm'], steps, p['grasp_z_mm']-.75)\n"+source[finish:]
    namespace = dict(original.__globals__)
    namespace['_motion_audit'] = lambda initial, steps, floor: audit_route(initial, steps, floor, client, plan.tip_rz)
    exec(compile(source, str(Path(plan.__file__).with_name('motion_path_runtime.py'))+'::pinned_compile_route', 'exec'), namespace)
    accelerated = namespace['compile_route']

    def compile_route(*args, **kwargs):
        _local.report = None
        result = accelerated(*args, **kwargs)
        result['motion_path_backend'] = client.backend
        # Report all candidate calls separately through the runtime status; the
        # selected route carries interval provenance in each audit row.
        result['motion_path_last_candidate'] = copy.deepcopy(_local.report)
        return result

    # Replace already imported aliases as well as future imports.
    aliases = []
    for module in tuple(sys.modules.values()):
        if module is not None and getattr(module, '__name__', '').startswith('src.'):
            if getattr(module, 'compile_route', None) is original:
                module.compile_route = compile_route
                aliases.append(module.__name__)
    plan.compile_route = compile_route
    from . import sorting_unified as consumer
    if consumer.compile_route is not compile_route:
        raise RuntimeError('production_route_consumer_alias_not_bound')
    if startup_fixture is not None:
        fixture = json.loads(Path(startup_fixture).read_text())
        if not 1 <= len(fixture['cases']) <= 4:
            raise RuntimeError('bounded_motion_boot_selftest_required')
        cases = []
        for case in fixture['cases']:
            label = case['label']
            values = (fixture['config'], label, case['card_xy'], case['axis_deg'], fixture['policy'],
                fixture['policy']['bin_slots'][label][case['slot_index']])
            baseline = original(*values)
            accelerated_result = consumer.compile_route(*values)
            equivalent_route(baseline, accelerated_result)
            cases.append(dict(case=case['id'], segments=len(baseline['steps']),
                command_sha256=hashlib.sha256(encoded(baseline['steps'])).hexdigest()))
        _startup_proof = dict(passed=True, cases=cases,
            fixture_sha256=hashlib.sha256(Path(startup_fixture).read_bytes()).hexdigest(),
            original_vs_actual_host_compiler=True, consumer='src.sorting_unified.compile_route',
            bound_aliases=sorted(aliases), physical_motion=False)
    from . import monitor_prompted_operation
    original_status = monitor_prompted_operation.PromptedOperations.status

    def status(self):
        result = original_status(self)
        with _lock:
            last = copy.deepcopy(_last)
        result.update(motion_path_backend=client.backend, motion_path_faulted=client.faulted,
                      last_motion_path_audit=last, motion_path_assets=client.assets,
                      motion_path_startup_selftest=copy.deepcopy(_startup_proof))
        return result

    monitor_prompted_operation.PromptedOperations.status = status
    _installed = True
    return client
