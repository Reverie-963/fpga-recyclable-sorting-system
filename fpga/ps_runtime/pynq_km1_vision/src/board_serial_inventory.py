"""Read Linux USB serial identity without opening a serial device.

The result is identification evidence only. No DTR/RTS changes, writes, driver
configuration, ownership claims or execution permission are performed.
"""
import argparse
import json
import os
from pathlib import Path
import stat


def usb_serial_inventory(sys_tty=Path('/sys/class/tty'), dev_root=Path('/dev')):
    result = []
    for entry in sorted(Path(sys_tty).glob('tty*')):
        if not entry.name.startswith(('ttyUSB', 'ttyACM')):
            continue
        device = (entry/'device').resolve()
        usb = next((p for p in (device, *device.parents)
                    if (p/'idVendor').is_file() and (p/'idProduct').is_file()), None)
        if usb is None:
            continue
        try:
            vid = int((usb/'idVendor').read_text().strip(), 16)
            pid = int((usb/'idProduct').read_text().strip(), 16)
            serial_path = usb/'serial'
            serial_number = serial_path.read_text().strip() if serial_path.is_file() else None
            node = Path(dev_root)/entry.name
            if not node.resolve().is_relative_to(Path(dev_root).resolve()):
                continue
            character_device = stat.S_ISCHR(node.stat().st_mode)
            aliases = [str(p) for p in sorted((Path(dev_root)/'serial/by-id').glob('*'))
                       if p.is_symlink() and p.resolve() == node.resolve()]
        except (OSError, ValueError):
            continue
        result.append(dict(device=str(node), vid=vid, pid=pid,
                           serial_number=serial_number, by_id=aliases,
                           character_device=character_device,
                           read_write_access=os.access(node, os.R_OK | os.W_OK)))
    return result


def identify_arm(devices, expected):
    if (not isinstance(expected, dict) or
            any(type(expected.get(k)) is not int or not 0 <= expected[k] <= 65535
                for k in ('vid', 'pid')) or
            (expected.get('serial_number') is not None and
             not isinstance(expected['serial_number'], str))):
        raise ValueError('invalid_expected_usb_identity')
    matches = [p for p in devices if
               (p.get('vid'), p.get('pid')) == (expected['vid'], expected['pid']) and
               (expected.get('serial_number') is None or
                p.get('serial_number') == expected['serial_number'])]
    selected = matches[0] if len(matches) == 1 else None
    if not matches:
        status = 'arm_usb_not_found'
    elif len(matches) != 1:
        status = 'arm_usb_ambiguous'
    elif not selected.get('character_device'):
        status = 'arm_usb_node_not_character_device'
    elif not selected.get('read_write_access'):
        status = 'arm_usb_access_unavailable'
    else:
        status = 'arm_usb_identified_not_opened'
    return dict(kind='board-arm-usb-identification-v1', status=status,
                expected_identity=expected, match_count=len(matches), selected=selected,
                devices=devices, serial_opened=False, physical_motion=False,
                execution_enabled=False, cutover_ready=False,
                caveat='USB identity alone does not prove actuator readiness or resolve pending tasks.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vid', required=True, type=lambda v: int(v, 0))
    parser.add_argument('--pid', required=True, type=lambda v: int(v, 0))
    parser.add_argument('--serial-number')
    args = parser.parse_args()
    result = identify_arm(usb_serial_inventory(),
                          dict(vid=args.vid, pid=args.pid, serial_number=args.serial_number))
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__':
    main()
