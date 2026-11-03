"""Static pick timing; never shorten a command's required movement wait."""
import math

PICK_STEPS = ('home', 'wrist', 'open', 'above_target', 'descend', 'close',
              'lift', 'above_bin', 'lower_bin', 'release', 'leave_bin', 'return')


def validate_timing(arm):
    duration = arm.get('move_ms', 1500)
    if type(duration) is not int or not 300 <= duration <= 5000:
        raise ValueError('Invalid move_ms: expected integer 300..5000')
    settle = arm.get('settle_s', 0)
    if type(settle) not in (int, float) or not math.isfinite(settle) or not 0 <= settle <= 10:
        raise ValueError('Invalid settle time')
    for key, integer, lo, hi in (('step_move_ms', True, 300, 5000),
                                 ('step_settle_s', False, 0, 10)):
        mapping = arm.get(key, {})
        if not isinstance(mapping, dict) or any(name not in PICK_STEPS for name in mapping):
            raise ValueError('Invalid ' + key + ' step names')
        for value in mapping.values():
            valid_type = type(value) is int if integer else type(value) in (int, float)
            if not valid_type or not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError('Invalid ' + key + ' value')


def move_ms(arm, name):
    return arm.get('step_move_ms', {}).get(name, arm['move_ms'])


def settle_s(arm, name):
    return arm.get('step_settle_s', {}).get(name, arm.get('settle_s', 0))
