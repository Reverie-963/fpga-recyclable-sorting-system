"""Offline shortest waypoint subsequence under explicit model constraints.

No serial, camera or execution entry point. This is not collision checking.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from .arm import packet
from .static_supervised_trial import load_plan, wait_policy
from openmv_gripper.km1_arm_bridge import validate_packet


LIMITS = ((500, 2500),) * 6  # Protocol bounds, not measured mechanical limits.


def tip_rz(pwm):
    shoulder = math.radians(90 + (pwm[1]-1500)*270/2000)
    elbow = math.radians((pwm[2]-1500)*270/2000)
    wrist = math.radians((pwm[3]-1500)*270/2000)
    angles = shoulder, shoulder-elbow, shoulder-elbow+wrist
    return (sum(n*math.cos(a) for n, a in zip((105, 88, 155), angles)),
            100+sum(n*math.sin(a) for n, a in zip((105, 88, 155), angles)))


def sample(start, end):
    return [tip_rz([a+(b-a)*i/100 for a, b in zip(start, end)]) for i in range(101)]


def distance(point, a, b):
    vector = [y-x for x, y in zip(a, b)]
    norm = sum(v*v for v in vector)
    fraction = max(0, min(1, sum((p-x)*v for p, x, v in zip(point, a, vector))/norm)) if norm else 0
    return math.dist(point, [x+fraction*v for x, v in zip(a, vector)])


def prepare(plan):
    state = plan.get('start_pwm')
    if (not isinstance(state, list) or len(state) != 6 or
            any(type(p) is not int or not 500 <= p <= 2500 for p in state)):
        raise ValueError('Six explicit starting PWM values required')
    stages = []
    for stage in plan['stages']:
        rows = []
        for index, step in enumerate(stage['steps']):
            ms = validate_packet(step['command'], LIMITS)
            if type(step['duration_ms']) is not int or ms != step['duration_ms']:
                raise ValueError('Duration mismatch')
            settle = step.get('extra_settle_s', 0)
            if type(settle) not in (int, float) or not math.isfinite(settle) or not 0 <= settle <= 10:
                raise ValueError('Invalid settle')
            target, ids = list(state), []
            for offset in range(1, len(step['command'])-1, 15):
                word = step['command'][offset:offset+15]
                sid = int(word[1:4])
                target[sid] = int(word[5:9])
                ids.append(sid)
            rows.append(dict(index=index, source=step, start=list(state), end=target,
                             ids=ids, points=sample(state, target)))
            state = target
        stages.append(rows)
    return stages


def budget(steps, policy='motion_plus_echo_grace'):
    # Same formula as the current supervised executor, excluding connection,
    # camera checks, fsync, communication and read timeout overshoot.
    return sum(s['duration_ms']/1000 + (max(.5, s.get('extra_settle_s', 0))
        if policy == 'motion_plus_echo_grace' else s.get('extra_settle_s', 0)) for s in steps)


def make_preview(plan, policy):
    deviation = policy.get('max_tip_deviation_mm', 2)
    rate = policy.get('max_arm_pwm_per_s', 200)
    floor = policy.get('min_move_ms', 1500)
    candidate_wait = wait_policy(policy)
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in (deviation, rate)):
        raise ValueError('Positive finite model tolerance and PWM rate required')
    if type(floor) is not int or not 1500 <= floor <= 5000:
        raise ValueError('Supervised preview minimum must be 1500..5000 ms')
    prepared = prepare(plan)
    required = policy.get('keep_after', {})
    if not isinstance(required, dict) or any(k not in [s['name'] for s in plan['stages']] for k in required):
        raise ValueError('Unknown stage in keep_after')
    output, original_count, total_before = [], 0, 0
    for stage, rows in zip(plan['stages'], prepared):
        mandatory = required.get(stage['name'], [])
        if (not isinstance(mandatory, list) or
                any(type(i) is not int or not 0 <= i < len(rows) for i in mandatory)):
            raise ValueError('Invalid protected waypoint index')
        original_count += len(rows)
        total_before += budget(stage['steps'], wait_policy(plan))
        cache = {}

        def edge(first, last):
            key = first, last
            if key in cache:
                return cache[key]
            group = rows[first:last+1]
            start, end = group[0]['start'], group[-1]['end']
            # Every wrist/gripper command and every explicit settling point is
            # a barrier. Keep all stage boundaries and requested clearance poses.
            if last > first and (any(i in mandatory for i in range(first, last)) or
                    any(any(sid >= 4 for sid in r['ids']) or r['source'].get('extra_settle_s', 0) > 0 for r in group)):
                cache[key] = None
                return None
            # Do not combine a substantial base turn with reaching/lowering.
            base_values = [start[0]]+[r['end'][0] for r in group]
            other_change = any(r['start'][1:4] != r['end'][1:4] for r in group)
            if last > first and max(base_values)-min(base_values) > 32 and other_change:
                cache[key] = None
                return None
            points = sample(start, end)
            original = [p for r in group for p in r['points']]
            error = 0.
            if last > first:
                # Symmetric sampled polyline distance: shortcuts cannot erase
                # a detour simply because their endpoints are on the old path.
                for a, b in ((points, original), (original, points)):
                    for point in a:
                        error = max(error, min(distance(point, x, y) for x, y in zip(b, b[1:])))
                        if error > deviation:
                            cache[key] = None
                            return None
            original_z = min(p[1] for p in original)
            if min(p[1] for p in points) < original_z-.5:
                cache[key] = None
                return None
            ids = sorted({sid for r in group for sid in r['ids']})
            if any(sid >= 4 for sid in ids):
                duration = group[0]['source']['duration_ms']  # Preserve grip timing.
            else:
                duration = max(floor, math.ceil(max(abs(end[i]-start[i]) for i in ids)*1000/rate))
                if duration > 5000:
                    cache[key] = None
                    return None
            settle = group[-1]['source'].get('extra_settle_s', 0)
            result = dict(step=group[-1]['source'].get('step', group[-1]['source'].get('timing_step', 'move')),
                command=packet([(i, end[i]) for i in ids], duration).decode(),
                duration_ms=duration, extra_settle_s=settle,
                source_indices=list(range(first, last+1)), start_pwm=start, target_pwm=end,
                max_sampled_tip_deviation_mm=error,
                nominal_model_z_range_mm=[min(p[1] for p in points), max(p[1] for p in points)],
                simultaneous_base_and_arm=abs(end[0]-start[0]) > 0 and start[1:4] != end[1:4])
            cache[key] = result
            return result

        # Lexicographic shortest path: fewest commands, then shortest budget.
        best = {0: ((0, 0.), [])}
        for end in range(1, len(rows)+1):
            for first in range(end):
                if first not in best:
                    continue
                candidate = edge(first, end-1)
                if candidate is None:
                    continue
                old_cost, old_path = best[first]
                cost = old_cost[0]+1, old_cost[1]+budget([candidate], candidate_wait)
                if end not in best or cost < best[end][0]:
                    best[end] = cost, old_path+[candidate]
        if len(rows) not in best:
            raise ValueError('No route under supplied timing and waypoint constraints')
        output.append(dict(name=stage['name'], observe=stage['observe'], steps=best[len(rows)][1]))
    all_steps = [s for stage in output for s in stage['steps']]
    return dict(kind='arm_route_preview_v1', preview_only=True, automatic_motion_allowed=False,
        physical_speed_verified=False, collision_free_verified=False,
        source_task_id=plan['task_id'], start_pwm=plan['start_pwm'], policy=policy,
        source_step_count=original_count, candidate_step_count=len(all_steps),
        source_executor_budget_s=total_before, candidate_executor_budget_s=budget(all_steps, candidate_wait),
        candidate_legacy_wait_budget_s=budget(all_steps), wait_policy=candidate_wait,
        minimum_scope='Fewest commands among original waypoint subsequences satisfying these sampled model constraints',
        stages=output, limitations=[
            'Nominal R/Z is not measured table or bin clearance; no calibrated base yaw or 3D swept volume',
            'Linear commanded PWM interpolation and 101 samples per segment; not actual controller dynamics',
            'No payload, cable or link collision model; skipped waypoints require new physical path verification',
            'PWM rate and model deviation are candidate settings, not measured hardware limits',
            'Budget assumes timely complete echo; excludes startup, checks, logging and I/O'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    plan = load_plan(args.plan, args.plan_sha256)
    policy_raw = args.policy.read_bytes()
    result = make_preview(plan, json.loads(policy_raw))
    result.update(source_plan_sha256=args.plan_sha256, policy_sha256=hashlib.sha256(policy_raw).hexdigest())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir/'preview.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('preview_only', 'source_step_count', 'candidate_step_count',
        'source_executor_budget_s', 'candidate_executor_budget_s')}))


if __name__ == '__main__':
    main()
