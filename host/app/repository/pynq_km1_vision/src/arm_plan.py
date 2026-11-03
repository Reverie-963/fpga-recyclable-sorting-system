"""Inspect teammate planner steps without importing serial or touching devices."""
import math

from .arm_timing import validate_timing, settle_s

from openmv_gripper.km1_arm_bridge import checked_limits, validate_packet


def inspect_plan(plan, cfg, dynamic=False):
    """Keep the existing (name, packet_bytes, duration_ms) contract.

    This checks software limits, NOT physical readiness or collision clearance.
    Dynamic OpenMV execution stays blocked until its timing design is commissioned.
    """
    arm = cfg['arm']
    mode = arm.get('transport', 'direct')
    blockers = []
    if mode not in ('direct', 'openmv_v3'):
        blockers.append('Unknown arm transport')
    if arm.get('commissioned') is not True:
        blockers.append('Arm not commissioned')
    if dynamic and mode == 'openmv_v3':
        blockers.append('Dynamic OpenMV bridge timing is unverified')
    limits = None
    try:
        limits = checked_limits(arm.get('joint_limits_pwm')) if mode == 'openmv_v3' else ((500, 2500),) * 6
    except ValueError as exc:
        blockers.append(str(exc))
    try:
        validate_timing(arm)
    except ValueError as exc:
        blockers.append(str(exc))
        return {'transport': mode, 'steps': [], 'execution_blockers': blockers,
                'software_checks_passed': False, 'nominal_wait_s': 0,
                'physical_success_verified': False}
    steps = []
    total = 0
    for step in plan:
        if not isinstance(step, (list, tuple)) or len(step) != 3:
            blockers.append('Malformed motion step')
            continue
        name, data, duration = step
        try:
            if not isinstance(name, str) or not name or not isinstance(data, bytes):
                raise ValueError('Named bytes motion step required')
            if type(duration) is not int or not 300 <= duration <= 5000:
                raise ValueError('Invalid step duration')
            if limits is not None and validate_packet(data, limits) != duration:
                raise ValueError('Plan duration differs from frame')
            steps.append({'step': name, 'command': data.decode('ascii'), 'wait_ms': duration,
                          'settle_s': settle_s(arm, name)})
            total += duration / 1000 + settle_s(arm, name)
        except (ValueError, UnicodeError) as exc:
            blockers.append(str(name) + ': ' + str(exc))
    if not steps:
        blockers.append('No valid motion steps')
    return {'transport': mode, 'steps': steps, 'execution_blockers': blockers,
            'software_checks_passed': not blockers, 'nominal_wait_s': total,
            'nominal_wait_note': 'Excludes USB/UART transfer, guard, planning and logging delays',
            'physical_success_verified': False}


def require_plan(plan, cfg, dynamic=False):
    report = inspect_plan(plan, cfg, dynamic)
    if report['execution_blockers']:
        raise ValueError('Plan blocked before motion: ' + '; '.join(report['execution_blockers']))
    return report
