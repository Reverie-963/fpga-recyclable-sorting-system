"""Offline fixed-target preview using the existing KM1 planner; never opens UART."""
import argparse
import json
from pathlib import Path

from .arm import execute, plan_pick


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/arm_pc.example.json')
    parser.add_argument('--xy', nargs=2, type=float, required=True, metavar=('X_MM', 'Y_MM'))
    parser.add_argument('--label', default='cap_deformed')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    plan = plan_pick(*args.xy, args.label, cfg)
    print(json.dumps({
        'status': 'offline_preview_only',
        'physical_execution': False,
        'coordinate_unit': 'mm',
        'target': {'xy': args.xy, 'label': args.label},
        'note': 'Example coordinates are not measured. IK reachability is not collision clearance.',
        'plan': execute(plan, cfg, enabled=False),
    }, indent=2))


if __name__ == '__main__':
    main()
