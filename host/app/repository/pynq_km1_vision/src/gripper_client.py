"""PC/PYNQ client for the OpenMV V2 gripper bridge. Motions require --execute."""
import argparse
import json

from openmv_gripper import arm_parameters as cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', choices=('status', 'open', 'close'), default='status')
    parser.add_argument('--port', help='OpenMV serial path; auto-detects its USB identity if omitted')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.action != 'status' and not args.execute:
        # Flat MicroPython imports are intentionally not used for host preview.
        pwm = cfg.OPEN_PWM if args.action == 'open' else cfg.CLOSE_PWM
        print(json.dumps({'status': 'preview_only', 'action': args.action,
                          'command': '{#005P%04dT%04d!}' % (pwm, cfg.MOVE_MS),
                          'serial_opened': False, 'physical_execution': False}, indent=2))
        return

    import serial
    from serial.tools import list_ports
    devices = [p for p in list_ports.comports() if (p.vid, p.pid) == (0x37C5, 0x124A)]
    if args.port:
        devices = [p for p in devices if p.device == args.port]
    if len(devices) != 1:
        raise RuntimeError('Select exactly one OpenMV H7 Plus port; CH340 is not this protocol')
    port = serial.Serial(port=None, baudrate=115200, timeout=2, write_timeout=1)
    port.dtr = True  # Native OpenMV USB CDC connection, not ESP32 boot control.
    port.rts = False
    port.port = devices[0].device

    def request(command):
        data = (command + '\n').encode('ascii')
        if port.write(data) != len(data):
            raise IOError('Partial USB write; no automatic retry')
        port.flush()
        raw = port.read_until(b'\n', size=160)
        if not raw.endswith(b'\n'):
            raise TimeoutError('Bridge reply missing/incomplete; do not replay motion')
        return raw.decode('ascii').strip()

    result = {'port': port.port, 'action': args.action, 'automatic_retry': False}
    try:
        port.open()
        result['ping'] = request('PING')
        if result['ping'] != 'READY KM1_GRIPPER_V2 NO_VISION':
            raise RuntimeError('Unexpected bridge firmware; no motion attempted')
        result['config'] = request('CONFIG')
        expected = 'CONFIG ID5 OPEN%d CLOSE%d MOVE%d' % (cfg.OPEN_PWM, cfg.CLOSE_PWM, cfg.MOVE_MS)
        if result['config'] != expected:
            raise RuntimeError('Host/device parameters differ; no motion attempted')
        if request('DISARM') != 'DISARMED':
            raise RuntimeError('Cannot verify idle permission state')
        result['bridge_status'] = request('STATUS')
        words = result['bridge_status'].split()
        if len(words) != 3 or words[0] != 'STATUS' or not words[1].startswith('LAST') or words[2] != 'ARMED0':
            raise RuntimeError('Unexpected status; no motion attempted')
        last_id = int(words[1][4:])
        if args.action == 'status':
            result['status'] = 'ready_no_motion'
        else:
            if request('ARM') != 'ARMED ONE_COMMAND':
                raise RuntimeError('Permission not confirmed; no motion attempted')
            request_id = last_id + 1
            # Exactly one motion request. Never automatically retry an uncertain send.
            response = request('%s %d' % (args.action.upper(), request_id))
            result['reply'] = response
            if response != 'SENT_UNVERIFIED %d' % request_id:
                raise RuntimeError('Motion not confirmed; inspect before trying again: ' + response)
            result['status'] = 'uart_sent_unverified'
    finally:
        port.close()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
