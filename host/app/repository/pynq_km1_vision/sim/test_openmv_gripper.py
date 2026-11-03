import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock

spec = importlib.util.spec_from_file_location('km1_gripper_bridge', Path(__file__).resolve().parents[1] / 'openmv_gripper/km1_gripper_bridge.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class GripperBridgeTests(unittest.TestCase):
    def setUp(self):
        self.uart = Mock()
        self.uart.write.side_effect = len
        self.bridge = module.Bridge(self.uart)

    def test_startup_ping_and_unarmed_request_never_write(self):
        self.assertIn('READY', self.bridge.handle('PING'))
        self.assertEqual(self.bridge.handle('GRIP 1 1450 2000'), 'ERROR NOT_ARMED')
        self.uart.write.assert_not_called()

    def test_one_shot_and_duplicate_protection(self):
        self.bridge.handle('ARM')
        self.assertEqual(self.bridge.handle('GRIP 1 1450 2000'), 'SENT_UNVERIFIED 1')
        self.uart.write.assert_called_once_with(b'{#005P1450T2000!}')
        self.assertEqual(self.bridge.handle('GRIP 2 1500 2000'), 'ERROR NOT_ARMED')
        self.bridge.handle('ARM')
        self.assertEqual(self.bridge.handle('GRIP 1 1450 2000'), 'ERROR DUPLICATE_OR_OLD')
        self.assertEqual(self.uart.write.call_count, 1)

    def test_bad_commands_consume_arm_without_motion(self):
        for command in ['GRIP 1 500 2000', 'GRIP 1 1450 299', 'GRIP 1 nope 2000', '#000P1500T1000!', 'POSE 0 170 30', 'GRIP 1 1450 2000 extra']:
            self.bridge.handle('ARM')
            self.assertTrue(self.bridge.handle(command).startswith('ERROR'))
            self.assertFalse(self.bridge.armed)
        self.uart.write.assert_not_called()

    def test_partial_write_consumes_id_and_disarms(self):
        self.uart.write.side_effect = lambda data: 2
        self.bridge.handle('ARM')
        self.assertEqual(self.bridge.handle('GRIP 1 1450 2000'), 'ERROR UART_PARTIAL_NO_RETRY')
        self.bridge.handle('ARM')
        self.assertEqual(self.bridge.handle('GRIP 1 1450 2000'), 'ERROR DUPLICATE_OR_OLD')
        self.assertEqual(self.uart.write.call_count, 1)

    def test_disconnect_disarm_and_uart_exception(self):
        self.bridge.handle('ARM')
        self.bridge.disarm()
        self.assertEqual(self.bridge.handle('GRIP 1 1450 2000'), 'ERROR NOT_ARMED')
        self.uart.write.side_effect = OSError('disconnected')
        self.bridge.handle('ARM')
        self.assertEqual(self.bridge.handle('GRIP 1 1450 2000'), 'ERROR UART_WRITE_NO_RETRY')
        self.assertFalse(self.bridge.armed)

    def test_presets_match_teammate_and_each_need_arm(self):
        self.assertEqual(self.bridge.handle('OPEN 1'), 'ERROR NOT_ARMED')
        self.bridge.handle('ARM')
        self.assertEqual(self.bridge.handle('OPEN 1'), 'SENT_UNVERIFIED 1')
        self.uart.write.assert_called_with(b'{#005P1000T0800!}')
        self.assertEqual(self.bridge.handle('CLOSE 2'), 'ERROR NOT_ARMED')
        self.bridge.handle('ARM')
        self.assertEqual(self.bridge.handle('CLOSE 2'), 'SENT_UNVERIFIED 2')
        self.uart.write.assert_called_with(b'{#005P1700T0800!}')

    def test_status_config_and_disarm_never_send(self):
        self.assertEqual(self.bridge.handle('STATUS'), 'STATUS LAST0 ARMED0')
        self.assertEqual(self.bridge.handle('CONFIG'), 'CONFIG ID5 OPEN1000 CLOSE1700 MOVE800')
        self.bridge.handle('ARM')
        self.assertEqual(self.bridge.handle('DISARM'), 'DISARMED')
        self.uart.write.assert_not_called()

    def test_invalid_configuration_rejected_before_writes(self):
        for values in [(0, 1700, 800), (1000, 2500, 800), (1000, 1700, 0), (True, 1700, 800)]:
            with self.assertRaises(ValueError):
                module.Bridge(self.uart, *values)
        self.uart.write.assert_not_called()
