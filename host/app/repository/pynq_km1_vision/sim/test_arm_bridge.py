import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from openmv_gripper.km1_arm_bridge import ArmBridge, limits_line, validate_packet
from src.arm import execute, packet
from src.arm_transport import ArmConnection


LIMITS = [(700, 2300)] * 5 + [(1000, 1700)]  # Simulation fixture, NOT physical calibration.


class FakeUSB:
    def __init__(self, limits=LIMITS, lose_move=False):
        self.uart = Mock()
        self.uart.write.side_effect = len
        self.bridge = ArmBridge(self.uart, limits=limits)
        self.lines = []
        self.pending = b''
        self.lose_move = lose_move
        self.closed = False

    def open(self):
        pass

    def write(self, data):
        line = data.decode().strip()
        self.lines.append(line)
        answer = self.bridge.handle(line)
        self.pending = b'' if self.lose_move and line.startswith('MOVE ') else (answer + '\n').encode()
        return len(data)

    def flush(self):
        pass

    def read_until(self, *args, **kwargs):
        return self.pending

    def close(self):
        self.closed = True
        self.bridge.disarm()


class ArmBridgeTests(unittest.TestCase):
    def test_whole_frame_validated_before_any_uart_byte(self):
        uart = Mock()
        bridge = ArmBridge(uart, limits=[None] * 5 + [(1000, 1700)])
        for frame in (packet([(5, 1000), (0, 1500)], 800), b'{#005P1000T0800!#005P1500T0800!}',
                      b'{#005P1000T0800!#004P1500T0801!}', b'#005P1000T0800!', b'{#005P+100T0800!}'):
            bridge.handle('ARM')
            self.assertTrue(bridge.handle('MOVE 1 ' + frame.decode()).startswith('ERROR'))
            self.assertFalse(bridge.armed)
        uart.write.assert_not_called()

    def test_multi_joint_packet_uses_one_write_and_consumes_permission(self):
        uart = Mock()
        uart.write.side_effect = len
        bridge = ArmBridge(uart, limits=LIMITS)
        frame = packet([(i, 1500) for i in range(6)], 1500)
        self.assertEqual(validate_packet(frame, LIMITS), 1500)
        bridge.handle('ARM')
        self.assertEqual(bridge.handle('MOVE 1 ' + frame.decode()), 'SENT_UNVERIFIED 1')
        self.assertEqual(bridge.handle('MOVE 2 ' + frame.decode()), 'ERROR NOT_ARMED')
        bridge.handle('ARM')
        self.assertEqual(bridge.handle('MOVE 1 ' + frame.decode()), 'ERROR DUPLICATE_OR_OLD')
        uart.write.assert_called_once_with(frame)

    def test_partial_send_consumes_id_without_retry(self):
        uart = Mock()
        uart.write.return_value = 3
        bridge = ArmBridge(uart, limits=LIMITS)
        bridge.handle('ARM')
        self.assertEqual(bridge.handle('MOVE 7 {#005P1000T0800!}'), 'ERROR UART_PARTIAL_NO_RETRY')
        bridge.handle('ARM')
        self.assertEqual(bridge.handle('MOVE 7 {#005P1000T0800!}'), 'ERROR DUPLICATE_OR_OLD')
        uart.write.assert_called_once()

    def connect(self, fake, limits=LIMITS):
        device = SimpleNamespace(device='TEST', vid=0x37c5, pid=0x124a)
        return patch('serial.Serial', return_value=fake), patch('serial.tools.list_ports.comports', return_value=[device])

    def test_host_device_roundtrip_and_full_sequence(self):
        fake = FakeUSB()
        cfg = {'arm': {'transport': 'openmv_v3', 'port': '', 'joint_limits_pwm': LIMITS,
                       'commissioned': True, 'settle_s': 0}}
        a, b = self.connect(fake)
        plan = [('pose', packet([(0, 1500), (1, 1600)], 800), 800),
                ('open', packet([(5, 1000)], 800), 800)]
        with a, b, patch('src.arm.time.sleep'):
            result = execute(plan, cfg, True)
        self.assertEqual(result['status'], 'sequence_sent_unverified')
        self.assertEqual(fake.uart.write.call_count, 2)
        self.assertEqual(sum(line == 'ARM' for line in fake.lines), 2)
        self.assertTrue(fake.closed)
        self.assertTrue(fake.dtr)

    def test_unmeasured_later_waypoint_blocks_before_usb(self):
        cfg = {'arm': {'transport': 'openmv_v3', 'joint_limits_pwm': [None] * 5 + [(1000, 1700)],
                       'commissioned': True}}
        with patch('serial.Serial') as serial:
            with self.assertRaises(ValueError):
                execute([('open', packet([(5, 1000)], 800), 800),
                         ('later', packet([(0, 1500)], 800), 800)], cfg, True)
            serial.assert_not_called()

    def test_limits_mismatch_never_arms(self):
        fake = FakeUSB(limits=[None] * 6)
        a, b = self.connect(fake)
        with a, b:
            connection = ArmConnection({'transport': 'openmv_v3', 'joint_limits_pwm': LIMITS})
            with self.assertRaises(RuntimeError):
                connection.open()
        self.assertNotIn('ARM', fake.lines)
        fake.uart.write.assert_not_called()

    def test_lost_ack_faults_connection_and_no_next_motion(self):
        fake = FakeUSB(lose_move=True)
        a, b = self.connect(fake)
        with a, b:
            connection = ArmConnection({'transport': 'openmv_v3', 'joint_limits_pwm': LIMITS})
            connection.open()
            with self.assertRaises(TimeoutError):
                connection.send(packet([(5, 1000)], 800), 800)
            with self.assertRaises(RuntimeError):
                connection.send(packet([(5, 1700)], 800), 800)
        self.assertEqual(sum(line.startswith('MOVE ') for line in fake.lines), 1)
        self.assertEqual(fake.uart.write.call_count, 1)

    def test_guard_rechecked_after_arm_before_move(self):
        fake = FakeUSB()
        a, b = self.connect(fake)
        with a, b:
            connection = ArmConnection({'transport': 'openmv_v3', 'joint_limits_pwm': LIMITS})
            connection.open()
            def changed():
                self.assertEqual(fake.lines[-1], 'ARM')
                raise RuntimeError('Target changed')
            with self.assertRaises(RuntimeError):
                connection.send(packet([(5, 1000)], 800), 800, guard=changed)
        self.assertFalse(any(line.startswith('MOVE ') for line in fake.lines))
        self.assertTrue(fake.closed)

    def test_preview_has_no_serial_and_reports_locked_joints(self):
        from src.arm_pick import main
        with patch('serial.Serial', side_effect=AssertionError('Unexpected hardware access')), \
                patch('sys.argv', ['arm_pick', '--xy', '0', '170']), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            main()
        self.assertIn('preview_only', out.getvalue())
        self.assertIn('Joint locked', out.getvalue())
