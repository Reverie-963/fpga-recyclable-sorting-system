"""Supervised, frozen-path physical trial alongside the offline sorting simulator.

This runner does not consume simulated success, infer robot coordinates, classify
objects, or claim visual verification. Every stage stops for a human observation.
Machine-specific paths and joint values belong in a separately reviewed plan.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time


def stamp():
    return datetime.datetime.now().astimezone().isoformat()


def save(path, value, exclusive=False):
    with path.open('x' if exclusive else 'w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def load_plan(path, expected_sha):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError('Reviewed plan hash differs')
    plan = json.loads(raw)
    if plan.get('kind') != 'supervised_fixed_path_trial_v1':
        raise ValueError('Expected an explicit supervised physical plan')
    wait_policy(plan)
    if not isinstance(plan.get('task_id'), str) or not plan['task_id'].strip():
        raise ValueError('Missing task identity')
    stages = plan['stages']
    if not stages or len({s['name'] for s in stages}) != len(stages):
        raise ValueError('Stage names must be unique')
    for stage in stages:
        if not re.fullmatch(r'[a-z_]+', stage['name']) or not stage.get('observe'):
            raise ValueError('Invalid stage or missing observation condition')
        if not stage['steps']:
            raise ValueError('Empty motion stage')
        for step in stage['steps']:
            frame, duration = step['command'], step['duration_ms']
            tokens = re.findall(r'#(\d{3})P(\d{4})T(\d{4})!', frame)
            if not tokens or '{' + ''.join('#%sP%sT%s!' % t for t in tokens) + '}' != frame:
                raise ValueError('Malformed protocol frame')
            ids = [int(t[0]) for t in tokens]
            if len(set(ids)) != len(ids) or any(i not in range(6) for i in ids):
                raise ValueError('Invalid or duplicate servo')
            if type(duration) is not int or not 1500 <= duration <= 5000:
                raise ValueError('Unexpected supervised-trial duration')
            settle = step.get('extra_settle_s', 0)
            if type(settle) not in (int, float) or not math.isfinite(settle) or not 0 <= settle <= 10:
                raise ValueError('Invalid supervised-trial settle time')
            if any(not 500 <= int(p) <= 2500 or int(t) != duration for _, p, t in tokens):
                raise ValueError('Invalid pulse or duration')
    return plan


def wait_policy(plan):
    policy = plan.get('wait_policy', 'motion_plus_echo_grace')
    if policy not in ('motion_plus_echo_grace', 'motion_and_echo'):
        raise ValueError('Unknown supervised wait policy')
    return policy


def prerequisites(root, plan, stage_name, digest):
    names = [s['name'] for s in plan['stages']]
    index = names.index(stage_name)
    for stage in plan['stages'][:index]:
        prior = json.loads((root / (stage['name'] + '.json')).read_text(encoding='utf-8'))
        observation = json.loads((root / (stage['name'] + '.observation.json')).read_text(encoding='utf-8'))
        if (prior['plan_sha256'] != digest or prior['task_id'] != plan['task_id'] or
                prior['status'] != 'sent_pending_observation' or
                observation['plan_sha256'] != digest or observation['task_id'] != plan['task_id'] or
                observation['execution_sha256'] != hashlib.sha256((root / (stage['name'] + '.json')).read_bytes()).hexdigest() or
                observation['ok'] is not True):
            raise ValueError('Prior stage is unverified or stopped')
    if (root / (stage_name + '.json')).exists():
        raise ValueError('Stage already attempted; never retry automatically')
    return plan['stages'][index]


def execute(root, plan, stage, digest, connection, clock=time):
    policy = wait_policy(plan)
    path = root / (stage['name'] + '.json')
    report = dict(task_id=plan['task_id'], stage=stage['name'], plan_sha256=digest,
                  started_at=stamp(), status='attempt_consumed_before_serial_open',
                  wait_policy=policy,
                  physical_result='pending_operator_observation', retry=False,
                  serial_opened=False, steps=[])
    # Durable reservation before opening the device. A crash or uncertain command
    # never becomes permission to repeat it in a new process.
    save(path, report, exclusive=True)
    try:
        connection.open()
        report['serial_opened'] = True
        save(path, report)
        clock.sleep(2)
        connection.reset_input_buffer()
        for step in stage['steps']:
            entry = dict(step, attempted=True, echo_matched=False)
            report['steps'].append(entry)
            report['status'] = 'command_may_be_in_flight'
            save(path, report)
            payload = step['command'].encode('ascii')
            entry['bytes_written'] = connection.write(payload)
            save(path, report)
            if entry['bytes_written'] != len(payload):
                raise IOError('Partial write; no subsequent command')
            chunks = []
            started = clock.monotonic()
            motion_end = started + step['duration_ms'] / 1000 + step.get('extra_settle_s', 0)
            deadline = started + step['duration_ms'] / 1000 + max(.5, step.get('extra_settle_s', 0))
            while clock.monotonic() < deadline:
                chunks.append(connection.read(4096))
                # Complete echo may arrive DURING the movement. An opt-in
                # candidate can overlap that wait, never the movement itself.
                if (policy == 'motion_and_echo' and clock.monotonic() >= motion_end
                        and payload in b''.join(chunks)):
                    break
            received = b''.join(chunks)
            entry['received_hex'] = received.hex()
            entry['echo_matched'] = payload in received
            save(path, report)
            if not entry['echo_matched']:
                raise IOError('Missing complete echo; physical result unknown')
        report['status'] = 'sent_pending_observation'
    except Exception as exc:
        report['status'] = 'stopped_needs_review'
        report['error'] = str(exc)
    finally:
        connection.close()
        report.update(serial_closed=True, finished_at=stamp())
        save(path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--journal', required=True, type=Path)
    parser.add_argument('--stage', required=True)
    parser.add_argument('--port')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--execute', action='store_true')
    group.add_argument('--observe')
    parser.add_argument('--ok', action='store_true')
    args = parser.parse_args()
    plan = load_plan(args.plan, args.plan_sha256)
    root = args.journal
    root.mkdir(parents=True, exist_ok=True)
    # One lock for the whole trial journal, including observation updates.
    lock = root / 'active.lock'
    fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    try:
        if args.observe is not None:
            if args.stage not in [s['name'] for s in plan['stages']] or not args.observe.strip():
                raise ValueError('Unknown stage or empty observation')
            execution = root / (args.stage + '.json')
            report = json.loads(execution.read_text(encoding='utf-8'))
            if report['task_id'] != plan['task_id'] or report['plan_sha256'] != args.plan_sha256:
                raise ValueError('Observation identity differs')
            if args.ok and report['status'] != 'sent_pending_observation':
                raise ValueError('Cannot confirm an uncertain execution for continuation')
            observation = dict(task_id=plan['task_id'], plan_sha256=args.plan_sha256,
                               execution_sha256=hashlib.sha256(execution.read_bytes()).hexdigest(),
                               ok=args.ok, text=args.observe, observed_at=stamp())
            save(root / (args.stage + '.observation.json'), observation, exclusive=True)
            print(json.dumps(observation, ensure_ascii=False))
            return
        stage = prerequisites(root, plan, args.stage, args.plan_sha256)
        if not args.execute:
            print(json.dumps(dict(status='preview_no_serial_open', stage=stage), ensure_ascii=False))
            return
        import serial
        from serial.tools import list_ports
        matches = [p for p in list_ports.comports() if (p.vid, p.pid) == (0x1a86, 0x7523)]
        if len(matches) != 1 or matches[0].device != args.port:
            raise ValueError('Expected unique explicitly selected CH340')
        connection = serial.Serial(port=None, baudrate=115200, timeout=.1, write_timeout=1)
        connection.dtr = False
        connection.rts = False
        connection.port = args.port
        report = execute(root, plan, stage, args.plan_sha256, connection)
        print(json.dumps(report, ensure_ascii=False))
        if report['status'] != 'sent_pending_observation':
            raise SystemExit(1)
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
