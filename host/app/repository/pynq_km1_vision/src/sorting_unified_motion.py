"""Explicit supervised timing trials; keep every compiled joint target and guard."""
import copy
import math

from .arm import packet
from .sorting_loop import finite, need
from openmv_gripper.km1_arm_bridge import validate_packet


def timing_policy(policy):
    value = policy.get('motion_timing')
    if value is None:
        return None
    need(isinstance(value, dict) and value.get('kind') == 'supervised-distance-timing-v1'
         and value.get('operator_statement') and value.get('physical_speed_verified') is False,
         'unified_speed_requires_explicit_trial_policy')
    rates = value.get('max_pwm_per_s')
    need(isinstance(rates, list) and len(rates) == 6 and
         all(finite(v) and 100 <= v <= 1000 for v in rates), 'invalid_unified_joint_rate')
    for key in ('min_move_ms', 'home_min_ms', 'vertical_min_ms', 'close_ms', 'release_ms'):
        need(type(value.get(key)) is int and 300 <= value[key] <= 5000,
             'invalid_unified_motion_duration:'+key)
    need(value['home_min_ms'] >= value['min_move_ms'] and
         value['vertical_min_ms'] >= value['min_move_ms'], 'invalid_unified_motion_floors')
    for key in ('close_settle_s', 'release_settle_s'):
        need(finite(value.get(key)) and .1 <= value[key] <= 2,
             'invalid_unified_grip_settle:'+key)
    need(finite(value.get('target_cycle_s')) and 10 <= value['target_cycle_s'] <= 180,
         'invalid_unified_cycle_target')
    return value


def movement_wait(policy, name, duration):
    timing = timing_policy(policy)
    extra = (1 if name == 'release' else .5) if timing is None else {
        'close': timing['close_settle_s'], 'release': timing['release_settle_s']}.get(name, 0)
    return duration/1000 + extra


def retime_route(route, policy):
    """Retiming changes T fields only, retaining even zero-distance waypoints."""
    timing = timing_policy(policy)
    if timing is None:
        return route
    result = copy.deepcopy(route)
    baseline = sum(movement_wait({}, s['name'], s['duration_ms']) for s in route['steps'])
    for step, audit in zip(result['steps'], result['nominal_interpolation']):
        name, command = step['name'], step['command']
        ids = [int(command[o+1:o+4]) for o in range(1, len(command)-1, 15)]
        floor = timing['min_move_ms']
        if name == 'home':
            floor = timing['home_min_ms']
        elif name in ('close', 'release'):
            floor = timing[name+'_ms']
        elif name.startswith('source_'):
            floor = timing['vertical_min_ms']
        ms = max(floor, math.ceil(max(abs(audit['target_pwm'][i]-audit['start_pwm'][i])
                 *1000/timing['max_pwm_per_s'][i] for i in ids)))
        need(ms <= 5000, 'unified_motion_exceeds_timing_bound')
        data = packet([(i, audit['target_pwm'][i]) for i in ids], ms)
        need(validate_packet(data, [[500, 2500]]*6) == ms, 'unified_retimed_packet_mismatch')
        step.update(command=data.decode('ascii'), duration_ms=ms,
                    settle_s=movement_wait(policy, name, ms)-ms/1000)
    result['motion_timing'] = dict(policy=copy.deepcopy(timing),
        baseline_motion_wait_s=baseline,
        candidate_motion_wait_s=sum(movement_wait(policy, s['name'], s['duration_ms'])
                                    for s in result['steps']),
        all_joint_targets_and_waypoints_preserved=True, physical_speed_verified=False,
        excludes='serial startup, localization, inference, frame guards, logging and final verification')
    return result
