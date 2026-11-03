"""Fixed-target commissioning only; inputs are measured arm coordinates in mm.

No camera/model needed. Not the continuous conveyor controller.
"""
import argparse
import json
from pathlib import Path
from .arm import plan_pick, execute
from .arm_plan import inspect_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/arm_openmv.example.json')
    parser.add_argument('--xy', type=float, nargs=2, required=True, metavar=('X_MM', 'Y_MM'))
    parser.add_argument('--label', default='cap_deformed')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    plan = plan_pick(*args.xy, args.label, cfg)
    # Report blockers even in dry-run without opening serial or sending ARM.
    report = inspect_plan(plan, cfg)
    blockers = report['execution_blockers']
    if args.execute and blockers:
        raise ValueError('Execution blocked before USB: ' + '; '.join(blockers))
    result = execute(plan, cfg, enabled=args.execute)
    print(json.dumps({'status': 'sequence_sent_unverified' if args.execute else 'preview_only',
                      'coordinate_unit': 'mm', 'target_xy': args.xy, 'label': args.label,
                      'execution_blockers': blockers, 'plan_validation': report, 'result': result}, indent=2))


if __name__ == '__main__':
    main()
