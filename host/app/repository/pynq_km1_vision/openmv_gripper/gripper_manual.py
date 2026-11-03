"""Explicit gripper-only calls using the teammate's existing driver and parameters.

Importing this file sends nothing. No sensor, model or HerbSort is loaded.
"""


def describe(action):
    import arm_parameters as cfg
    if action == 'open':
        pwm = cfg.OPEN_PWM
    elif action == 'close':
        pwm = cfg.CLOSE_PWM
    else:
        raise ValueError('Only open or close is supported')
    duration = cfg.MOVE_MS
    if type(pwm) is not int or not 500 <= pwm <= 2500:
        raise ValueError('Invalid configured gripper PWM')
    if type(duration) is not int or not 300 <= duration <= 5000:
        raise ValueError('Invalid configured move duration')
    # Identical format to HerbSort._send_servo, ID5 only.
    if cfg.GRIPPER_ID != 5:
        raise ValueError('This commissioning module is limited to gripper ID5')
    return '{{#{0:0>3d}P{1:0>4d}T{2:0>4d}!}}'.format(cfg.GRIPPER_ID, pwm, duration)


def execute_once(action):
    command = describe(action)
    import kinematics
    driver = kinematics.Kinematics()
    written = driver.send_str(command)
    if written != len(command):
        raise OSError('Incomplete UART write; do not automatically replay')
    return {'command': command, 'written': written, 'status': 'sent_unverified'}
