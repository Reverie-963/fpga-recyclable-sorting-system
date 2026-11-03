import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
from src.arm_transport import ArmConnection


class ArmIOTimeoutTests(unittest.TestCase):
    def test_explicit_half_io_timeouts_preserve_boot_stability_wait(self):
        for arm,read,write in [({},.25,1),({'read_timeout_s':.125,'write_timeout_s':.5},.125,.5)]:
            serial=Mock();serial.Serial.return_value=Mock()
            with patch.dict(sys.modules,{'serial':serial}),patch('src.arm_transport.time.sleep') as sleep:
                connection=ArmConnection(dict(arm,port='offline'));connection.open()
            serial.Serial.assert_called_once_with(port=None,baudrate=115200,timeout=read,write_timeout=write)
            sleep.assert_called_once_with(2)
            connection.close()

    def test_invalid_io_timeout_does_not_open_port(self):
        serial=Mock()
        with patch.dict(sys.modules,{'serial':serial}):
            with self.assertRaisesRegex(ValueError,'IO timeout'):
                ArmConnection(dict(read_timeout_s=0,port='offline')).open()
        serial.Serial.assert_not_called()
