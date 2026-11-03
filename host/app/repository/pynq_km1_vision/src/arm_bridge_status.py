"""Query V3 identity, joint limits and idle state. Never ARM or MOVE."""
import argparse
import json
from pathlib import Path
from .arm_transport import ArmConnection, limits_line


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/arm_openmv.example.json')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    if cfg['arm'].get('transport') != 'openmv_v3':
        raise ValueError('V3 status requires openmv_v3 transport')
    connection = ArmConnection(cfg['arm'])
    try:
        connection.open()
        print(json.dumps({'status': 'ready_no_motion', 'port': connection.port.port,
                          'limits': limits_line(connection.limits), 'last_id': connection.last_id,
                          'physical_execution': False}, indent=2))
    finally:
        connection.close()


if __name__ == '__main__':
    main()
