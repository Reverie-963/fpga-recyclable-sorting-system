"""KM1 机械臂逆运动学。

参数和舵机协议与厂家 V3.1 源码一致；复制到独立新增目录，未改动厂家文件。
"""

import math
from pyb import UART


class Kinematics:
    # 四个关节长度，按厂家程序放大 10 倍。
    L0 = 1000
    L1 = 1050
    L2 = 880
    L3 = 1550
    pi = 3.1415926
    time = 1500

    uart = UART(3, 115200)
    uart.init(115200, bits=8, parity=None, stop=1)

    def send_str(self, cmd):
        return self.uart.write(cmd)

    def kinematics_analysis(self, x, y, z, alpha):
        """把末端坐标和夹爪俯角换算为 0~3 号总线舵机指令。"""
        x = x * 10
        y = y * 10
        z = z * 10

        l0 = float(self.L0)
        l1 = float(self.L1)
        l2 = float(self.L2)
        l3 = float(self.L3)

        if x == 0:
            theta6 = 0.0
        else:
            theta6 = math.atan(x / y) * 270.0 / self.pi

        y = math.sqrt(x * x + y * y)
        y = y - l3 * math.cos(alpha * self.pi / 180.0)
        z = z - l0 - l3 * math.sin(alpha * self.pi / 180.0)
        if z < -l0:
            return 1
        if math.sqrt(y * y + z * z) > (l1 + l2):
            return 2

        ccc = math.acos(y / math.sqrt(y * y + z * z))
        bbb = (y * y + z * z + l1 * l1 - l2 * l2) / (2 * l1 * math.sqrt(y * y + z * z))
        if bbb > 1 or bbb < -1:
            return 5
        zf_flag = -1 if z < 0 else 1

        theta5 = ccc * zf_flag + math.acos(bbb)
        theta5 = theta5 * 180.0 / self.pi
        if theta5 > 180.0 or theta5 < 0.0:
            return 6

        aaa = -(y * y + z * z - l1 * l1 - l2 * l2) / (2 * l1 * l2)
        if aaa > 1 or aaa < -1:
            return 3

        theta4 = math.acos(aaa)
        theta4 = 180.0 - theta4 * 180.0 / self.pi
        if theta4 > 135.0 or theta4 < -135.0:
            return 4

        theta3 = alpha - theta5 + theta4
        if theta3 > 90.0 or theta3 < -90.0:
            return 7

        servo_pwm0 = int(1500 - 2000.0 * theta6 / 270.0)
        servo_pwm1 = int(1500 + 2000.0 * (theta5 - 90) / 270.0)
        servo_pwm2 = int(1500 + 2000.0 * theta4 / 270.0)
        servo_pwm3 = int(1500 + 2000.0 * theta3 / 270.0)

        return ("{{#000P{0:0>4d}T{4:0>4d}!#001P{1:0>4d}T{4:0>4d}!"
                "#002P{2:0>4d}T{4:0>4d}!#003P{3:0>4d}T{4:0>4d}!}}".format(
                    servo_pwm0, servo_pwm1, servo_pwm2, servo_pwm3, self.time))

    def kinematics_move(self, x, y, z, move_time, alpha=None):
        """移动末端；alpha 为空时沿用厂家自动选角，给定时保持固定俯角。"""
        if y < 0:
            return 0

        self.time = move_time
        if alpha is not None:
            arm_str = self.kinematics_analysis(x, y, z, alpha)
            if isinstance(arm_str, str):
                return self.uart.write(arm_str)
            return 0

        best_angle = 0
        found = False
        for angle in range(0, -136, -1):
            if isinstance(self.kinematics_analysis(x, y, z, angle), str):
                if angle < best_angle:
                    best_angle = angle
                found = True

        if found:
            return self.uart.write(self.kinematics_analysis(x, y, z, best_angle))
        return 0

    def is_reachable(self, x, y, z, alpha=None):
        """只检查坐标是否可达，不向舵机发送指令。"""
        if y < 0:
            return False
        if alpha is not None:
            return isinstance(self.kinematics_analysis(x, y, z, alpha), str)
        for angle in range(0, -136, -1):
            if isinstance(self.kinematics_analysis(x, y, z, angle), str):
                return True
        return False
