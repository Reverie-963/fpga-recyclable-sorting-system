import contextlib
import io
import unittest
from unittest.mock import patch

from src import gripper_client


class FakeSerial:
    def __init__(self, bad_config=False, lose_motion_reply=False):
        self.port = None
        self.writes = []
        self.bad_config = bad_config
        self.lose_motion_reply = lose_motion_reply
        self.closed = False

    def open(self):
        pass

    def close(self):
        self.closed = True

    def flush(self):
        pass

    def write(self, data):
        self.writes.append(data.decode().strip())
        return len(data)

    def read_until(self, *args, **kwargs):
        command = self.writes[-1]
        responses = {'PING': 'READY KM1_GRIPPER_V2 NO_VISION',
                     'CONFIG': 'WRONG' if self.bad_config else 'CONFIG ID5 OPEN1000 CLOSE1700 MOVE800',
                     'DISARM': 'DISARMED', 'STATUS': 'STATUS LAST7 ARMED0', 'ARM': 'ARMED ONE_COMMAND'}
        if command in responses:
            return (responses[command] + '\n').encode()
        return b'' if self.lose_motion_reply else b'SENT_UNVERIFIED 8\n'


class GripperClientTests(unittest.TestCase):
    def invoke(self, args, fake):
        from types import SimpleNamespace
        device = SimpleNamespace(device='COM5', vid=0x37C5, pid=0x124A)
        with patch('serial.Serial', return_value=fake), patch('serial.tools.list_ports.comports', return_value=[device]), patch('sys.argv', ['client', *args]), contextlib.redirect_stdout(io.StringIO()):
            gripper_client.main()

    def test_preview_does_not_open_serial(self):
        with patch('serial.Serial', side_effect=AssertionError('Unexpected UART access')), patch('sys.argv', ['client', 'open']), contextlib.redirect_stdout(io.StringIO()) as out:
            gripper_client.main()
        self.assertIn('preview_only', out.getvalue())

    def test_status_has_no_motion_permission_or_request(self):
        fake = FakeSerial()
        self.invoke(['status'], fake)
        self.assertEqual(fake.writes, ['PING', 'CONFIG', 'DISARM', 'STATUS'])
        self.assertTrue(fake.closed)

    def test_config_mismatch_blocks_motion(self):
        fake = FakeSerial(bad_config=True)
        with self.assertRaisesRegex(RuntimeError, 'parameters differ'):
            self.invoke(['open', '--execute'], fake)
        self.assertNotIn('ARM', fake.writes)
        self.assertTrue(fake.closed)

    def test_uncertain_motion_response_is_not_replayed(self):
        fake = FakeSerial(lose_motion_reply=True)
        with self.assertRaises(TimeoutError):
            self.invoke(['close', '--execute'], fake)
        self.assertEqual([w for w in fake.writes if w.startswith('CLOSE')], ['CLOSE 8'])
        self.assertTrue(fake.closed)
