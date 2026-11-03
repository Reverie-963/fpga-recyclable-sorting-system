"""V3 entry: deploy AS main.py only for a separately arranged commissioning."""
from km1_gripper_bridge import main
from km1_arm_bridge import ArmBridge
from arm_limits import JOINT_LIMITS


def configured_bridge(uart, open_pwm, close_pwm, move_ms):
    return ArmBridge(uart, open_pwm, close_pwm, move_ms, limits=JOINT_LIMITS)


main(bridge_factory=configured_bridge, frame_limit=128)
