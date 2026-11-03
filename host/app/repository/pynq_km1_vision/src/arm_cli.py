import argparse,json
from pathlib import Path
from .arm import ik,packet,execute

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='config.json')
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--pose',nargs=3,type=float,metavar=('X','Y','Z'))
    group.add_argument('--servo',nargs=2,type=int,metavar=('ID','PWM'))
    p.add_argument('--execute',action='store_true');a=p.parse_args();cfg=json.loads(Path(a.config).read_text(encoding='utf-8-sig'))
    duration=cfg['arm']['move_ms']
    pairs=list(enumerate(ik(*a.pose,cfg['arm'].get('alpha'),cfg['arm'].get('base_yaw_calibration')))) if a.pose else [tuple(a.servo)]
    plan=[('manual_commissioning',packet(pairs,duration),duration)]
    print(json.dumps(execute(plan,cfg,a.execute),indent=2))
if __name__=='__main__':main()
