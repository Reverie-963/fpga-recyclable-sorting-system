"""Measure each joint before replacing None with (minimum_pwm, maximum_pwm).

None locks that joint. The jaw interval comes from teammate configuration;
only its OPEN endpoint has physical confirmation. Not a collision model.
"""
JOINT_LIMITS = (None, None, None, None, None, (1000, 1700))
