"""Explicit one-check motion sequence; transport and stop guards stay live."""


def scene_check_mode(options):
    mode=options.get('motion_scene_check_mode','per_command')
    if mode not in ('per_command','before_sequence'):
        raise ValueError('invalid_motion_scene_check_mode')
    statement=options.get('operator_statement')
    if mode=='before_sequence' and (not isinstance(statement,str) or not statement.strip()):
        raise ValueError('sequence_motion_requires_operator_statement')
    return mode


def send_sequence_step(motion,data,duration,guard,send):
    """Use the controller's fresh post-USB first-command guard exactly once.

    `send` is the unchanged Motion.send transport path. It checks stops again
    immediately before UART; its original echo and movement wait are retained.
    No image or source-file scan is performed between committed route steps.
    """
    if not getattr(motion,'sequence_visual_checked',False) and not callable(guard):
        raise ValueError('sequence_initial_visual_guard_required')
    def checked():
        motion.check_stop()
        if guard:guard()
        motion.check_stop()
        if not getattr(motion,'sequence_visual_checked',False):
            motion.sequence_visual_checked=True
            motion.sequence_scene_checks=1
    return send(motion,data,duration,checked)
