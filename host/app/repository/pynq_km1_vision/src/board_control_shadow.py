"""Board PS migration worker: frozen route planning and offline control checks.

No camera, detector, PL programming, UART or production binding is opened.
Input is a locally staged immutable fixture, never a live execution request.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import time


def signature(route):
    """Compare actuator bytes and complete wait policy across CPU architectures."""
    return hashlib.sha256(json.dumps(dict(
        steps=[dict(name=s['name'], command=s['command'], duration_ms=s['duration_ms'])
               for s in route['steps']],
        waits=route['_waits']), sort_keys=True, allow_nan=False).encode()).hexdigest()


def plan_case(config, policy, case):
    from .sorting_unified_plan import compile_route
    from .sorting_unified_motion import movement_wait
    route = compile_route(config, case['label'], case['card_xy'], case['axis_deg'],
                          policy, policy['bin_slots'][case['label']][case['slot_index']])
    route['_waits'] = [movement_wait(policy, s['name'], s['duration_ms']) for s in route['steps']]
    return signature(route), len(route['steps'])


def ledger_checks(root):
    from .sorting_loop import Journal
    with tempfile.TemporaryDirectory(prefix='ledger-check-', dir=root) as directory:
        with Journal(directory) as ledger:
            ledger.save('pending.json', dict(task_id='shadow-only', status='needs_review'))
            exclusive = False
            try:
                with Journal(directory):
                    pass
            except FileExistsError:
                exclusive = True
        with Journal(directory) as ledger:
            preserved = ledger.pending() == dict(task_id='shadow-only', status='needs_review')
        return dict(exclusive=exclusive, pending_survives_restart=preserved,
                    passed=exclusive and preserved, production_ledger_used=False)


def run(fixture, scratch):
    if fixture.get('kind') != 'board-control-shadow-fixture-v1':
        raise ValueError('invalid_shadow_fixture')
    cases = fixture['cases']
    if not isinstance(cases, list) or not 1 <= len(cases) <= 120:
        raise ValueError('invalid_shadow_case_count')
    # Low priority, one BLAS/OpenCV thread; no persistent polling service.
    if hasattr(os, 'nice'):
        os.nice(19)
    import cv2
    cv2.setNumThreads(1)
    started = time.monotonic()
    rows = []
    for case in cases:
        before = time.monotonic()
        cpu_before = time.process_time()
        try:
            actual, count = plan_case(fixture['config'], fixture['policy'], case)
            row = dict(passed=actual == case['expected_sha256'], actual_sha256=actual, steps=count)
        except ValueError as exc:
            row = dict(passed=False, error=str(exc))
        wall = time.monotonic()-before
        cpu = time.process_time()-cpu_before
        rows.append(dict(case=case['id'], elapsed_ms=wall*1000, cpu_ms=cpu*1000, **row))
        # At most roughly 20% of one CPU core for the route loop. nice also
        # yields to the existing camera/PL process; no latency claim is implied.
        time.sleep(max(.05, 5*cpu-wall))
    from .standby_offline import run_suite
    standby = run_suite()
    ledger = ledger_checks(scratch)
    passed = all(r['passed'] for r in rows) and standby['status'] == 'passed' and ledger['passed']
    return dict(kind='board-control-shadow-result-v1', status='passed' if passed else 'failed',
                machine=platform.machine(), python=platform.python_version(),
                location='pynq_ps' if platform.machine().startswith('arm') else 'host_reference',
                fixture_source=fixture['source'], routes=rows, route_count=len(rows),
                standby=standby, ledger=ledger, elapsed_s=time.monotonic()-started,
                execution_enabled=False, physical_motion=False, camera_opened=False,
                overlay_changed=False, production_state_changed=False,
                remaining=['live_detector_transport', 'cross_host_device_ownership',
                           'arm_and_voice_serial_attachment', 'bound_runtime_cutover',
                           'physical_acceptance'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Reserve evidence before doing work. Existing reports cannot be replaced.
    with args.output.open('x', encoding='utf-8') as out:
        result = run(json.loads(args.fixture.read_text(encoding='utf-8')), args.output.parent)
        json.dump(result, out, indent=2, allow_nan=False)
    print(json.dumps(dict(status=result['status'], routes=result['route_count'],
                         standby_passed=result['standby']['passed'], elapsed_s=result['elapsed_s'])))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
