from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.board_serial_inventory import identify_arm, usb_serial_inventory


class SerialIdentificationTests(unittest.TestCase):
    def device(self, **changes):
        return dict(dict(device='/dev/ttyUSB0', vid=0x1a86, pid=0x7523,
                         serial_number=None, character_device=True,
                         read_write_access=True, by_id=[]), **changes)

    def identify(self, devices, serial_number=None):
        return identify_arm(devices, dict(vid=0x1a86, pid=0x7523, serial_number=serial_number))

    def test_voice_device_with_same_vendor_is_not_arm(self):
        result = self.identify([self.device(pid=0x7522), self.device()])
        self.assertEqual(result['match_count'], 1)
        self.assertEqual(result['status'], 'arm_usb_identified_not_opened')
        self.assertFalse(result['serial_opened'])
        self.assertFalse(result['cutover_ready'])
        self.assertFalse(result['execution_enabled'])

    def test_missing_or_duplicate_devices_cannot_be_selected(self):
        self.assertEqual(self.identify([])['status'], 'arm_usb_not_found')
        r = self.identify([self.device(), self.device(device='/dev/ttyUSB1')])
        self.assertEqual(r['status'], 'arm_usb_ambiguous')
        self.assertIsNone(r['selected'])

    def test_required_serial_number_is_not_optional(self):
        self.assertEqual(self.identify([self.device()], 'unique')['match_count'], 0)
        r = self.identify([self.device(serial_number='unique'), self.device()], 'unique')
        self.assertEqual(r['match_count'], 1)

    def test_wrong_node_or_missing_permissions_cannot_claim_readiness(self):
        for changes, status in ((dict(character_device=False), 'arm_usb_node_not_character_device'),
                                (dict(read_write_access=False), 'arm_usb_access_unavailable')):
            r = self.identify([self.device(**changes)])
            self.assertEqual(r['status'], status)
            self.assertFalse(r['cutover_ready'])

    def test_invalid_identity_rejected(self):
        for value in (dict(vid=True, pid=1), dict(vid=1, pid=65536),
                      dict(vid=1, pid=1, serial_number=123)):
            with self.assertRaises(ValueError):
                identify_arm([], value)

    def test_sysfs_inventory_reads_attributes_without_opening_device(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            attributes = base/'sys/ttyUSB0/device'
            attributes.mkdir(parents=True)
            (attributes/'idVendor').write_text('1a86\n')
            (attributes/'idProduct').write_text('7523\n')
            (base/'dev').mkdir()
            node = base/'dev/ttyUSB0'
            node.write_bytes(b'do not open')
            real_open = Path.open
            def guarded_open(path, *args, **kwargs):
                if path == node:
                    raise AssertionError('serial device must not be opened')
                return real_open(path, *args, **kwargs)
            with patch.object(Path, 'open', guarded_open):
                rows = usb_serial_inventory(base/'sys', base/'dev')
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['pid'], 0x7523)
            self.assertFalse(rows[0]['character_device'])


if __name__ == '__main__':
    unittest.main()
