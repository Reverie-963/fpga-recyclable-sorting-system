"""Bounded, supervised motion candidates derived from the qualified full route.

The old specimen geometry remains the input; neither shortened paths nor faster
timing inherit its physical acceptance. No serial or configuration writes here.
"""
import math

from .arm import packet
from .arm_route_preview import make_preview
from .arm_timing import settle_s
from openmv_gripper.km1_arm_bridge import validate_packet


PROFILES = ('qualified', 'compact')
POLICY = dict(max_tip_deviation_mm=2, max_arm_pwm_per_s=300, min_move_ms=1500,
              wait_policy='motion_and_echo')


def motion_wait(arm, name, duration, profile='qualified'):
    if profile not in PROFILES:
        raise ValueError('Unknown known-item motion profile')
    extra = settle_s(arm, name) if profile == 'qualified' else {
        'close': .3, 'release': .5}.get(name, 0)
    return duration/1000 + extra


def compact_plan(config, plan):
    """Keep approach, grip, lift clearance, base turns and release boundaries.

    Search the remaining waypoint subsequences using the same bidirectional
    sampled model checks as the offline route tool, then retime every command.
    """
    names = [name for name, _, _ in plan]
    protected = [names.index('source_hover')]
    original = dict(task_id='known-item-motion-candidate',
        start_pwm=list(config['known']['initial_pwm']), wait_policy='motion_and_echo',
        stages=[dict(name='cycle', observe='post-return three-frame verification', steps=[
            dict(step=name, command=data.decode('ascii'), duration_ms=ms, extra_settle_s=0)
            for name, data, ms in plan])])
    policy = dict(POLICY, keep_after={'cycle': protected})
    preview = make_preview(original, policy)
    result, details = [], []
    for row in preview['stages'][0]['steps']:
        name = row['step']
        start, target = row['start_pwm'], row['target_pwm']
        ids = sorted({int(row['command'][offset+1:offset+4])
                      for offset in range(1, len(row['command'])-1, 15)})
        # Home includes already-set wrist/open values. The explicit known start
        # lets it use the same distance budget instead of always taking 5 s.
        ms = 1000 if name in ('close', 'release') else max(POLICY['min_move_ms'],
            math.ceil(max(abs(target[i]-start[i]) for i in ids)*1000/POLICY['max_arm_pwm_per_s']))
        if ms > 5000:
            raise ValueError('Compact route exceeds bounded joint rate')
        data = packet([(i, target[i]) for i in ids], ms)
        if validate_packet(data, [[500, 2500]]*6) != ms:
            raise ValueError('Compact motion protocol mismatch')
        result.append((name, data, ms))
        details.append(dict(step=name, merged_steps=[names[i] for i in row['source_indices']],
            duration_ms=ms, wait_s=motion_wait(config['arm'], name, ms, 'compact'),
            max_sampled_tip_deviation_mm=row['max_sampled_tip_deviation_mm'],
            start_pwm=start, target_pwm=target))
    baseline = sum(motion_wait(config['arm'], name, ms) for name, _, ms in plan)
    candidate = sum(row['wait_s'] for row in details)
    return result, dict(profile='compact', supervised_motion_trial=True,
        baseline_geometry_qualification_only=True, physical_speed_verified=False,
        collision_free_verified=False, policy=policy, baseline_steps=len(plan),
        candidate_steps=len(result), baseline_motion_wait_s=baseline,
        candidate_motion_wait_s=candidate, motion_wait_reduction_fraction=1-candidate/baseline,
        excludes='serial startup, frame guards, inference, verification and durable logging', steps=details)
