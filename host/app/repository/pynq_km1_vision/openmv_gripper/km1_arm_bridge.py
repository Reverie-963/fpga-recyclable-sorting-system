"""Pure MicroPython/CPython protocol core. Never moves on import or startup."""


def checked_limits(limits):
    if not isinstance(limits, (list, tuple)) or len(limits) != 6:
        raise ValueError('Six joint limits required')
    result = []
    for interval in limits:
        if interval is None:
            result.append(None)
        elif (isinstance(interval, (list, tuple)) and len(interval) == 2
              and all(type(v) is int for v in interval)
              and 500 <= interval[0] <= interval[1] <= 2500):
            result.append(tuple(interval))
        else:
            raise ValueError('Invalid measured joint limits')
    return tuple(result)


def limits_line(limits):
    limits = checked_limits(limits)
    return 'LIMITS ' + ','.join('%d:-' % i if value is None else '%d:%d:%d' % (i, value[0], value[1])
                                for i, value in enumerate(limits))


def validate_packet(command, limits):
    """Reject the complete frame before any UART byte; no arbitrary passthrough."""
    limits = checked_limits(limits)
    if isinstance(command, bytes):
        command = command.decode('ascii')
    if not isinstance(command, str) or not command.startswith('{') or not command.endswith('}'):
        raise ValueError('Wrapped servo frame required')
    body = command[1:-1]
    if not body or len(body) > 90 or len(body) % 15:
        raise ValueError('One to six servo entries required')
    seen = set()
    duration = None
    for offset in range(0, len(body), 15):
        word = body[offset:offset + 15]
        if word[0] != '#' or word[4] != 'P' or word[9] != 'T' or word[14] != '!':
            raise ValueError('Malformed servo entry')
        numbers = word[1:4] + word[5:9] + word[10:14]
        if not all('0' <= char <= '9' for char in numbers):
            raise ValueError('ASCII decimal fields required')
        sid, pwm, ms = int(word[1:4]), int(word[5:9]), int(word[10:14])
        if sid not in range(6) or sid in seen:
            raise ValueError('Unknown or duplicate joint')
        bounds = limits[sid]
        if bounds is None or not bounds[0] <= pwm <= bounds[1]:
            raise ValueError('Joint locked or PWM outside measured limits')
        if not 300 <= ms <= 5000 or (duration is not None and duration != ms):
            raise ValueError('Invalid or mixed durations')
        seen.add(sid)
        duration = ms
    return duration


class ArmBridge:
    def __init__(self, uart, open_pwm=1000, close_pwm=1700, move_ms=800, limits=None):
        self.uart = uart
        self.limits = checked_limits(limits if limits is not None else (None,) * 6)
        self.armed = False
        self.last_id = 0

    def disarm(self):
        self.armed = False

    def handle(self, line):
        words = line.split()
        if words == ['PING']:
            return 'READY KM1_ARM_V3 NO_VISION'
        if words == ['LIMITS']:
            return limits_line(self.limits)
        if words == ['STATUS']:
            return 'STATUS LAST%d ARMED%d' % (self.last_id, int(self.armed))
        if words == ['DISARM']:
            self.disarm()
            return 'DISARMED'
        if words == ['ARM']:
            self.armed = True
            return 'ARMED ONE_COMMAND'
        armed = self.armed
        self.disarm()
        if len(words) != 3 or words[0] != 'MOVE':
            return 'ERROR COMMAND'
        if not armed:
            return 'ERROR NOT_ARMED'
        try:
            if not words[1] or not all('0' <= c <= '9' for c in words[1]):
                raise ValueError('Invalid ID')
            request_id = int(words[1])
            if not self.last_id < request_id <= 2147483647:
                return 'ERROR DUPLICATE_OR_OLD'
            validate_packet(words[2], self.limits)
        except (ValueError, UnicodeError):
            return 'ERROR FRAME_OR_LIMITS'
        self.last_id = request_id
        data = words[2].encode('ascii')
        try:
            written = self.uart.write(data)
        except Exception:
            return 'ERROR UART_WRITE_NO_RETRY'
        if written != len(data):
            return 'ERROR UART_PARTIAL_NO_RETRY'
        return 'SENT_UNVERIFIED %d' % request_id
