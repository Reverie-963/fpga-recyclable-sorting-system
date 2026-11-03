"""Exercise the real standby state machine using scripted scenes, without devices.

This module never constructs StationBackend, opens a camera, starts a network
service, loads a local station binding, or sends serial/robot commands.
"""
import argparse
from collections import deque
import json
from pathlib import Path
import threading
import time

from .monitor_operation import Operations


MATERIALS = ('AUTO', 'PAPER', 'METAL', 'PLASTIC')


class ScriptedBackend:
    unified = True

    def __init__(self, scenes=(), *, outcome='simulated_visual_pass', blockers=()):
        self.scenes = deque(scenes)
        self.current = []
        self.outcome = outcome
        self.blockers = list(blockers)
        self.calls = []
        self.stop_requests = 0
        self.observation_entered = threading.Event()
        self.observation_release = threading.Event()
        self.pause_observation = False

    def check(self):
        return list(self.blockers)

    def cycle(self, material, execute):
        call = dict(material=material, simulated_attempt=bool(execute), physical_motion=False)
        self.calls.append(call)
        if execute:
            call['status'] = self.outcome
            return dict(status=self.outcome, task_id='offline-simulated-attempt',
                        physical_motion_attempted=False, completed_count=int(self.outcome == 'simulated_visual_pass'))
        self.observation_entered.set()
        if self.pause_observation and not self.observation_release.wait(3):
            raise TimeoutError('offline_observation_release_timeout')
        if self.scenes:
            self.current = self.scenes.popleft()
        if isinstance(self.current, dict):
            report = dict(self.current)
        else:
            matching = [v for v in self.current if material == 'AUTO' or v == material]
            report = dict(status='preview_only' if matching else
                          'no_matching_material' if self.current else 'empty_scene')
            if matching:
                report['choice'] = dict(label=matching[0].lower())
        call['status'] = report['status']
        call['scene'] = self.current
        return report

    def stop(self):
        # Count a simulated stop only. Never touch the real global stop files.
        self.stop_requests += 1


def command(session, action, *, request_id='offline-request-one', owner='offline-owner', **values):
    return session.command(dict(token=session.token, action=action, owner=owner,
                                request_id=request_id, ready=True, **values))


def until(predicate, timeout=3):
    deadline = time.monotonic()+timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError('offline_scenario_timeout')
        threading.Event().wait(.002)


def run_case(name, backend, *, material='AUTO', intervention=None, lease_s=12, standby_s=120,
             expected_state='finished', expected_attempts=1, expected_reason=None,
             expected_stop_requests=0):
    session = Operations(backend, poll_s=.005, lease_s=lease_s, standby_s=standby_s)
    started = time.monotonic()
    validation_error = None
    try:
        if intervention == 'blocked':
            try:
                command(session, 'arm', material=material)
            except ValueError as exc:
                validation_error = str(exc)
        else:
            command(session, 'arm', material=material)
            if intervention in ('cancel', 'stop', 'busy', 'foreign_owner'):
                if not backend.observation_entered.wait(2):
                    raise TimeoutError('offline_observation_not_started')
                if intervention == 'busy':
                    try:
                        command(session, 'start', request_id='offline-request-two')
                    except ValueError as exc:
                        validation_error = str(exc)
                    command(session, 'cancel')
                elif intervention == 'foreign_owner':
                    try:
                        command(session, 'cancel', owner='offline-foreign-owner')
                    except ValueError as exc:
                        validation_error = str(exc)
                    command(session, 'cancel')
                else:
                    command(session, intervention)
                backend.observation_release.set()
            until(lambda: session.state not in session.ACTIVE)
            if session.worker:
                session.worker.join(timeout=2)
        attempts = sum(v['simulated_attempt'] for v in backend.calls)
        passed = (session.state == expected_state and attempts == expected_attempts and
                  backend.stop_requests == expected_stop_requests and
                  (expected_reason is None or session.reason == expected_reason))
        if intervention == 'blocked':
            passed = passed and bool(validation_error and validation_error.startswith('station_not_ready:'))
        elif intervention == 'busy':
            passed = passed and bool(validation_error and 'busy' in validation_error)
        elif intervention == 'foreign_owner':
            passed = passed and validation_error == 'not_operation_owner'
        # The completed worker never re-arms. Duplicate IDs remain rejected.
        duplicate_rejected = None
        if session.state == 'finished':
            try:
                command(session, 'arm', material=material)
            except ValueError as exc:
                duplicate_rejected = str(exc) == 'duplicate_request_no_retry'
            passed = passed and duplicate_rejected is True
        return dict(name=name, passed=bool(passed), material=material, state=session.state,
                    reason=session.reason, simulated_attempts=attempts, result=session.result,
                    simulated_stop_requests=backend.stop_requests, calls=list(backend.calls),
                    request_rejection=validation_error, duplicate_request_rejected=duplicate_rejected,
                    elapsed_s=round(time.monotonic()-started, 4), physical_motion=False)
    finally:
        backend.observation_release.set()
        session.close()


def run_suite():
    cases = []
    for material in MATERIALS:
        selected = 'PAPER' if material == 'AUTO' else material
        cases.append(run_case('clear_then_new_'+material.lower(),
            ScriptedBackend([[selected], [], [selected]]), material=material))
    for material in MATERIALS[1:]:
        other = next(v for v in MATERIALS[1:] if v != material)
        cases.append(run_case('other_material_remains_'+material.lower(),
            ScriptedBackend([[other], [other, material]]), material=material))
    for action in ('cancel', 'stop', 'busy', 'foreign_owner'):
        backend = ScriptedBackend([[]]);backend.pause_observation = True
        cases.append(run_case(action+'_during_observation', backend, intervention=action,
                              expected_state='stopped', expected_attempts=0,
                              expected_stop_requests=int(action == 'stop')))
    cases.append(run_case('old_target_never_clears', ScriptedBackend([['PAPER']]), standby_s=.03,
                          expected_state='stopped', expected_attempts=0, expected_reason='standby_timeout'))
    cases.append(run_case('target_never_arrives', ScriptedBackend([[]]), standby_s=.03,
                          expected_state='stopped', expected_attempts=0, expected_reason='standby_timeout'))
    cases.append(run_case('page_heartbeat_lost', ScriptedBackend([[]]), lease_s=.03,
                          expected_state='stopped', expected_attempts=0, expected_reason='page_disconnected'))
    cases.append(run_case('observation_rejected', ScriptedBackend([dict(status='rejected')]),
                          expected_state='rejected', expected_attempts=0))
    cases.append(run_case('observation_blocked', ScriptedBackend([
        dict(status='preview_only', execution_blockers=['offline_occupied_slot'])]),
        expected_state='rejected', expected_attempts=0))
    cases.append(run_case('existing_unresolved_task', ScriptedBackend(blockers=['offline_pending']),
                          intervention='blocked', expected_state='idle', expected_attempts=0))
    cases.append(run_case('failed_attempt_not_retried', ScriptedBackend([[], ['PAPER']], outcome='needs_review')))
    return dict(kind='offline-standby-rehearsal-v1', simulation_only=True, physical_motion=False,
                real_devices_used=False, live_configuration_changed=False,
                status='passed' if all(v['passed'] for v in cases) else 'failed',
                passed=sum(v['passed'] for v in cases), total=len(cases), cases=cases,
                limits=['Scripted scene reports exercise standby/control behavior only.',
                        'No image recognition, calibration, grasp trajectory, servo speed or physical success verification.',
                        'Shortened offline test timers are not production timeout settings.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New JSON report; existing files are never overwritten')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new report path')
    report = run_suite()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as out:
        json.dump(report, out, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(status=report['status'], passed=report['passed'], total=report['total'],
                          physical_motion=False, report=str(args.output.resolve()))))
    raise SystemExit(0 if report['status'] == 'passed' else 1)


if __name__ == '__main__':
    main()
