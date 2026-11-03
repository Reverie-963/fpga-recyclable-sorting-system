"""OpenMV USB-to-UART3 gripper commissioning bridge. No vision or startup motion."""


class Bridge:
    def __init__(self, uart, open_pwm=1000, close_pwm=1700, move_ms=800):
        self.uart = uart
        self.armed = False
        self.last_id = 0
        self.open_pwm = open_pwm
        self.close_pwm = close_pwm
        self.move_ms = move_ms
        if not all(type(v) is int for v in (open_pwm, close_pwm, move_ms)):
            raise ValueError('Invalid parameter types')
        if not (1000 <= open_pwm <= 1700 and 1000 <= close_pwm <= 1700 and 300 <= move_ms <= 5000):
            raise ValueError('Parameters outside commissioning bounds')

    def disarm(self):
        self.armed = False

    def handle(self, line):
        words = line.split()
        if words == ['PING']:
            return 'READY KM1_GRIPPER_V2 NO_VISION'
        if words == ['CONFIG']:
            return 'CONFIG ID5 OPEN%d CLOSE%d MOVE%d' % (self.open_pwm, self.close_pwm, self.move_ms)
        if words == ['STATUS']:
            return 'STATUS LAST%d ARMED%d' % (self.last_id, int(self.armed))
        if words == ['ARM']:
            self.armed = True
            return 'ARMED ONE_COMMAND'
        if words == ['DISARM']:
            self.disarm()
            return 'DISARMED'
        armed = self.armed
        self.disarm()  # Every attempt consumes permission, including malformed commands.
        if len(words) == 2 and words[0] in ('OPEN', 'CLOSE'):
            pwm = self.open_pwm if words[0] == 'OPEN' else self.close_pwm
            words = ['GRIP', words[1], str(pwm), str(self.move_ms)]
        if len(words) != 4 or words[0] != 'GRIP':
            return 'ERROR COMMAND'
        if not armed:
            return 'ERROR NOT_ARMED'
        try:
            request_id, pwm, duration = (int(word) for word in words[1:])
        except ValueError:
            return 'ERROR INTEGER'
        if request_id <= self.last_id or request_id > 2147483647:
            return 'ERROR DUPLICATE_OR_OLD'
        if not 1000 <= pwm <= 1700 or not 300 <= duration <= 5000:
            return 'ERROR RANGE'
        self.last_id = request_id  # Consume before UART write. Never retry partial sends.
        command = ('{#005P%04dT%04d!}' % (pwm, duration)).encode('ascii')
        try:
            written = self.uart.write(command)
        except Exception:
            return 'ERROR UART_WRITE_NO_RETRY'
        if written != len(command):
            return 'ERROR UART_PARTIAL_NO_RETRY'
        return 'SENT_UNVERIFIED %d' % request_id


class VendorTransport:
    """Use the actual teammate driver that passed the physical OPEN test."""
    def __init__(self):
        from kinematics import Kinematics
        self.driver = Kinematics()

    def write(self, data):
        return self.driver.send_str(data.decode('ascii'))

    def any(self):
        return self.driver.uart.any()

    def read(self, count):
        return self.driver.uart.read(count)


def main(bridge_factory=Bridge, frame_limit=63):
    import pyb
    import time
    import arm_parameters as cfg
    usb = pyb.USB_VCP()
    if cfg.GRIPPER_ID != 5:
        raise ValueError('Only ID5 is allowed')
    uart = VendorTransport()
    bridge = bridge_factory(uart, cfg.OPEN_PWM, cfg.CLOSE_PWM, cfg.MOVE_MS)
    pending = bytearray()
    dropping = False
    last_rx = time.ticks_ms()
    last_arm = last_rx
    while True:
        now = time.ticks_ms()
        if not usb.isconnected():
            bridge.disarm()
            pending = bytearray()
            dropping = False
        if bridge.armed and time.ticks_diff(now, last_arm) > 5000:
            bridge.disarm()
        if (pending or dropping) and time.ticks_diff(now, last_rx) > 1000:
            pending = bytearray()
            dropping = False
            bridge.disarm()
        data = usb.read(64) if usb.any() else None
        if data:
            last_rx = now
            for value in data:
                if value == 10:
                    if dropping:
                        response = 'ERROR TOO_LONG'
                        bridge.disarm()
                    else:
                        try:
                            response = bridge.handle(bytes(pending).decode('ascii').strip())
                        except (UnicodeError, ValueError):
                            bridge.disarm()
                            response = 'ERROR ASCII'
                    pending = bytearray()
                    dropping = False
                    if response == 'ARMED ONE_COMMAND':
                        last_arm = now
                    usb.send((response + '\n').encode('ascii'), timeout=100)
                elif value != 13:
                    if len(pending) >= frame_limit or not 32 <= value <= 126:
                        dropping = True
                        bridge.disarm()
                    elif not dropping:
                        pending.append(value)
        # Drain bounded UART input; never echo it into the USB command parser.
        if uart.any():
            uart.read(min(uart.any(), 128))
        time.sleep_ms(5)
