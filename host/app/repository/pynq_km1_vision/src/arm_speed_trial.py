"""Generate a faster static-arm trial from calibrated config; preview only."""
import argparse
import copy
import json
from pathlib import Path
from .arm import plan_pick
from .arm_plan import inspect_plan
from .arm_timing import PICK_STEPS, validate_timing

# Candidates for supervised experiments, NOT measured physical limits.
PROFILES = {
    'brisk': (1500,600,600,1000,900,1000,1000,1300,900,900,1000,1300),
    'candidate': (1500,500,500,800,800,800,900,1200,800,700,800,1200),
}


def make_trial(config, profile, xy, label):
    if profile not in PROFILES:
        raise ValueError('Unknown trial profile')
    original = plan_pick(*xy, label, config)
    original_report = inspect_plan(original, config)
    trial = copy.deepcopy(config)
    arm = trial['arm']
    arm['commissioned'] = False
    arm['step_move_ms'] = dict(zip(PICK_STEPS, PROFILES[profile]))
    arm['step_settle_s'] = dict.fromkeys(PICK_STEPS, 0)
    arm['step_settle_s'].update(descend=.1, close=.2, release=.2)
    validate_timing(arm)
    plan = plan_pick(*xy, label, trial)
    report = inspect_plan(plan, trial)
    baseline = {n:d for n,_,d in original}
    comparison = [{'step':s['step'], 'before_ms':baseline[s['step']],
                   'candidate_ms':s['wait_ms'], 'extra_settle_s':s['settle_s']}
                  for s in report['steps']]
    return trial, {'preview_only':True, 'physical_verified':False, 'profile':profile,
                   'target_xy_mm':list(xy), 'label':label,
                   'previous_nominal_s':original_report['nominal_wait_s'],
                   'candidate_nominal_s':report['nominal_wait_s'],
                   'steps':comparison, 'plan_validation':report,
                   'note':'No serial opened. Retains positions and limits; requires on-site commissioning. Includes motion and settle only.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--profile',choices=PROFILES,required=True)
    parser.add_argument('--xy',type=float,nargs=2,required=True)
    parser.add_argument('--label',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    config=json.loads(args.config.read_text(encoding='utf-8-sig'))
    trial,report=make_trial(config,args.profile,args.xy,args.label)
    # Never replace the operator's source configuration or an earlier trial.
    args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir/'config.preview.json').write_text(json.dumps(trial,indent=2)+'\n',encoding='utf-8')
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'preview_only':True,'profile':args.profile,
                      'nominal_wait_s':report['candidate_nominal_s'],
                      'output_dir':str(args.output_dir)}))


if __name__=='__main__':main()
