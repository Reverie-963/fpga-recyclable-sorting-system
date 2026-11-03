from pathlib import Path
import sys,types,importlib.util,json,platform,subprocess,re,argparse
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r))
from src.arm import ik
fake=types.ModuleType('pyb')
class UART:
 def __init__(self,*a,**k):pass
 def init(self,*a,**k):pass
 def write(self,*a):raise AssertionError('No writes allowed')
fake.UART=UART;sys.modules['pyb']=fake
parser=argparse.ArgumentParser(description='Compare arm IK with vendor reference without opening serial devices.')
parser.add_argument('vendor',nargs='?',type=Path)
parser.add_argument('--output',type=Path,default=r/'report/vendor_ik_comparison.json')
args=parser.parse_args()
if args.vendor:
 p=args.vendor
else:
 candidates=list((r.parent/'reference/km1').glob('4*/KM1*V3.1/kinematics.py'))
 if not candidates: candidates=list(r.parent.glob('03*/4*/KM1*V3.1/kinematics.py'))
 if len(candidates)!=1: raise SystemExit('Provide the vendor kinematics.py path explicitly; expected one V3.1 reference.')
 p=candidates[0]
spec=importlib.util.spec_from_file_location('vendor',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);v=m.Kinematics()
count=0;maxdiff=0;reach_mismatch=0
for x in range(-100,101,25):
 for y in range(50,221,20):
  for z in range(20,141,20):
   for a in range(-110,1,10):
    try: old=v.kinematics_analysis(x,y,z,a)
    except (ValueError,ZeroDivisionError): old=None
    try: new=ik(x,y,z,a)
    except ValueError: new=None
    oldp=list(map(int,re.findall(r'P(\d+)',old))) if isinstance(old,str) else None
    if (oldp is None)!=(new is None):reach_mismatch+=1
    if oldp is not None and new is not None:count+=1;maxdiff=max(maxdiff,max(abs(u-w) for u,w in zip(oldp,new)))
result={'compared_reachable_poses':count,'reachability_mismatches':reach_mismatch,'max_pwm_difference':maxdiff,'source':str(p),'note':'Floating-point pi differs slightly; no serial device opened.'}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(result,indent=2));print(result)
assert reach_mismatch==0 and maxdiff<=1
