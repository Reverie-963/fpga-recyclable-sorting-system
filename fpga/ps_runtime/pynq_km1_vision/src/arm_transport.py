"""Static arm transport. V3 is opt-in; no motion retry or V2 fallback."""
import time

from openmv_gripper.km1_arm_bridge import checked_limits, limits_line, validate_packet


class ArmConnection:
    def __init__(self, arm):
        self.arm = arm
        self.mode = arm.get('transport', 'direct')
        if self.mode not in ('direct', 'openmv_v3'):
            raise ValueError('Unknown arm transport')
        self.limits = checked_limits(arm['joint_limits_pwm']) if self.mode == 'openmv_v3' else None
        self.port = None
        self.last_id = 0
        self.failed = False

    def validate(self, data, duration):
        if self.mode == 'openmv_v3' and validate_packet(data, self.limits) != duration:
            raise ValueError('Plan duration differs from frame')

    def open(self):
        import serial
        selected = self.arm.get('port', '')
        if self.mode == 'openmv_v3':
            from serial.tools import list_ports
            devices = [p for p in list_ports.comports() if (p.vid, p.pid) == (0x37c5, 0x124a)
                       and (not selected or p.device == selected)]
            if len(devices) != 1:
                raise RuntimeError('Exactly one matching OpenMV required')
            selected = devices[0].device
        read_timeout=self.arm.get('read_timeout_s',.25)
        write_timeout=self.arm.get('write_timeout_s',1)
        if not (.05<=read_timeout<=.25 and .1<=write_timeout<=1):
            raise ValueError('Invalid arm IO timeout')
        self.port = serial.Serial(port=None, baudrate=115200, timeout=read_timeout, write_timeout=write_timeout)
        self.port.dtr = self.mode == 'openmv_v3'
        self.port.rts = False
        self.port.port = selected
        try:
            self.port.open()
            if self.mode == 'direct':
                time.sleep(2)
                return
            if self.request('PING') != 'READY KM1_ARM_V3 NO_VISION':
                raise RuntimeError('V3 firmware required; V2 cannot control the whole arm')
            if self.request('LIMITS') != limits_line(self.limits):
                raise RuntimeError('Host and OpenMV joint limits differ')
            if self.request('DISARM') != 'DISARMED':
                raise RuntimeError('Cannot verify idle state')
            words = self.request('STATUS').split()
            if (len(words) != 3 or words[0] != 'STATUS' or words[2] != 'ARMED0'
                    or not words[1].startswith('LAST') or not words[1][4:].isdigit()):
                raise RuntimeError('Invalid bridge state')
            self.last_id = int(words[1][4:])
            if not 0 <= self.last_id < 2147483647:
                raise RuntimeError('Request IDs exhausted; manual restart required')
        except Exception:
            self.close()
            raise

    def request(self, line):
        data = (line + '\n').encode('ascii')
        if self.port.write(data) != len(data):
            raise IOError('Partial USB write; never retry')
        self.port.flush()
        response = self.port.read_until(b'\n', size=160)
        if not response.endswith(b'\n'):
            raise TimeoutError('Reply lost; inspect physical state before any new motion')
        return response.decode('ascii').strip()

    def send(self, data, duration, guard=None):
        if self.failed:
            raise RuntimeError('Connection faulted; no further motions allowed')
        self.validate(data, duration)
        try:
            if self.mode == 'direct':
                if guard:
                    guard()
                stamp = time.monotonic()
                if self.port.write(data) != len(data):
                    raise IOError('Partial UART write; never retry')
                self.port.flush()
                return stamp
            if self.request('ARM') != 'ARMED ONE_COMMAND':
                raise RuntimeError('One-shot permission missing')
            # Recheck target AFTER the ARM exchange, immediately before MOVE.
            if guard:
                guard()
            self.last_id += 1
            stamp = time.monotonic()
            reply = self.request('MOVE %d %s' % (self.last_id, data.decode('ascii')))
            if reply != 'SENT_UNVERIFIED %d' % self.last_id:
                raise RuntimeError('Motion uncertain/rejected; no replay: ' + reply)
            return stamp
        except Exception:
            self.failed = True
            # Close drops USB DTR and pending permission; no STOP/rollback claim.
            self.close()
            raise

    def close(self):
        if self.port is not None:
            self.port.close()
            self.port = None
