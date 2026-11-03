import copy,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from src.moving_targets import Tracker,intercept,verify_deposit
from src.recycling_run import validate_dynamic,run


def cfg():return json.loads(Path('config/contest_board.json').read_text())
def obj(x=20,y=20):return {'center':[x,y],'bbox':[x-5,y-5,10,10],'area':100}
CAL={'homography':[[1,0,0],[0,1,100],[0,0,1]]}

class DynamicTests(unittest.TestCase):
 def test_motion_association_preserves_id_and_delayed_label(self):
  t=Tracker(cfg()['dynamic']);t.update([obj()],1,0,CAL);snapshot=copy.deepcopy(t.tracks)
  t.update([obj(y=24)],1.2,4,CAL);t.bind(snapshot,[{'xyxy':[15,15,25,25],'label':'plastic_cap','confidence':.9}],1)
  self.assertEqual(t.tracks[1]['label'],'plastic_cap');self.assertEqual(t.tracks[1]['xy'],[20,124])
 def test_ambiguity_breaks_identity(self):
  t=Tracker(cfg()['dynamic']);t.update([obj()],1,0,CAL)
  t.update([obj(x=19),obj(x=21)],1.1,0,CAL)
  self.assertNotIn(1,t.tracks);self.assertTrue(all(o['label'] is None for o in t.tracks.values()))
 def test_old_response_cannot_label_new_object(self):
  t=Tracker(cfg()['dynamic']);t.update([obj()],1,0,CAL);old=copy.deepcopy(t.tracks)
  t.update([obj(x=60)],1.1,0,CAL);t.bind(old,[{'xyxy':[15,15,25,25],'label':'plastic_cap','confidence':.9}],1)
  self.assertIsNone(next(iter(t.tracks.values()))['label'])
 def test_wrong_class_invalidates_track(self):
  t=Tracker(cfg()['dynamic']);t.update([obj()],1,0,CAL);s=copy.deepcopy(t.tracks)
  for label in ['plastic_cap','wood_piece']:t.bind(s,[{'xyxy':[15,15,25,25],'label':label,'confidence':.9}],1)
  self.assertNotIn(1,t.tracks)
 def test_predict_window_speed_staleness_and_error_budget(self):
  c=cfg();t={'id':1,'xy':[0,150],'belt':0,'stamp':1,'seen':5,'label':'plastic_cap','confidence':.9,'semantic_stamp':1}
  s={'position_mm':0,'time':1,'velocity_mm_s':5,'uncertainty_s':.001}
  hit=intercept(t,s,1,2,c);self.assertEqual(hit['xy'],[0,160])
  self.assertIsNone(intercept(t,{**s,'velocity_mm_s':-1},1,2,c))
  self.assertIsNone(intercept(t,s,4,2,c))
  self.assertIsNone(intercept(t,s,1,20,c))
 def test_belt_departure_is_not_grasp_success(self):
  c=cfg();c['verification']['bin_rois']={'plastic_cap':[70,70,20,20]}
  before=np.zeros((100,100),np.uint8);before[10:30,10:30]=255;after=np.zeros_like(before)
  self.assertEqual(verify_deposit(before,after,'plastic_cap',c)['status'],'needs_review')
  after[70:90,70:90]=255;self.assertEqual(verify_deposit(before,after,'plastic_cap',c)['status'],'bin_visual_evidence_pass')
 def test_execute_requires_dynamic_commissioning(self):
  with self.assertRaisesRegex(ValueError,'Commission'):validate_dynamic(cfg(),True)
 def test_bad_axis_window_and_nonfinite_rejected(self):
  for k,v in [('axis_xy',[1,1]),('pick_window_mm',[0,1,0,1]),('capture_delay_s',float('nan')),('grip_contact_s',4)]:
   c=cfg();c['dynamic'][k]=v
   with self.assertRaises(ValueError):validate_dynamic(c)

 def controller(self,fault=None):
  c=cfg();c['vision'].update(backend='cpu',min_area=10,max_area=1000);c['camera'].update(width=160,height=120)
  c['arm'].update(commissioned=True,move_ms=300,settle_s=.01);c['grasp'].update(clearance_px=2,min_interior_px=2)
  c['dynamic'].update(commissioned=True,capture_delay_s=0,min_track_frames=2,max_track_age_s=.5,track_max_gap_s=.5,association_mm=5,semantic_max_age_s=4,schedule_slack_s=.08)
  c['verification']['bin_rois']={label:[130,85,25,25] for label in c['arm']['bins']};c['verification']['min_bin_gain_px']=20
  state={'time':10.,'steps':[],'done':False,'polling':False}
  class Cam:
   def __init__(self,*a):self.i=0
   def read(self,after=0):
    state['time']+=.02;self.i+=1;im=np.full((120,160,3),240,np.uint8)
    if not state['done']:
     y=int(40+4*(state['time']-10));im[y:y+12,65:77]=0
    else:im[90:105,135:150]=0
    return self.i,state['time'],im
   def close(self):pass
  class Enc:
   def __init__(self,*a):pass
   def sample(self):
    jump=20 if fault=='drift' and 'above_target' in state['steps'] else 0
    return {'time':state['time'],'position_mm':2*(state['time']-10)+jump,'velocity_mm_s':2,'uncertainty_s':0,'ticks':int(state['time']*1e8)}
  class Seg:
   def __init__(self,*a,**kw):self.overlay=None
   def close(self):pass
  class Client:
   def __init__(self,*a):pass
   def predict(self,im,fid,stamp):
    ys,xs=np.where(im[:,:,0]<100)
    return {'detections':[] if not len(xs) else [{'label':'plastic_cap','confidence':.99,'xyxy':[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]}]}
  class Result:
   def __init__(self,value):self.value=value
   def done(self):return True
   def result(self):return self.value
  class Pool:
   def __init__(self,*a,**kw):pass
   def submit(self,fn,*a):return Result(fn(*a))
   def shutdown(self,**kw):pass
  class Arm:
   def __init__(self,*a):pass
   def prepare(self):pass
   def close(self):pass
   def send(self,step,poll,guard):
    guard();state['steps'].append(step[0]);end=state['time']+step[2]/1000+.01
    while state['time']<end:poll();state['time']+=.02
    if fault=='late' and step[0]=='above_target':state['time']+=.2
    if step[0]=='return':state['done']=True
  with tempfile.TemporaryDirectory() as directory:
   old=os.getcwd();os.chdir(directory)
   try:
    c['calibration']='cal.json';Path('cal.json').write_text(json.dumps({'image_size':[160,120],'pixel_hull':[[1,1],[115,1],[115,115],[1,115]],'homography':[[.5,0,-35],[0,.5,130],[0,0,1]]}))
    with patch('src.recycling_run.LatestCamera',Cam),patch('src.recycling_run.Encoder',Enc),patch('src.pl.Segmenter',Seg),patch('src.recycling_run.DetectionClient',Client),patch('src.recycling_run.MovingArm',Arm),patch('src.recycling_run.ThreadPoolExecutor',Pool),patch('time.monotonic',lambda:state['time']):
     if fault:
      with self.assertRaises((RuntimeError,TimeoutError)):run(c,True,1,30)
     else:self.assertEqual(run(c,True,1,30)['completed'],1)
   finally:os.chdir(old)
  return state
 def test_conveyor_controller_executes_intercept_and_bin_verification(self):self.assertIn('close',self.controller()['steps'])
 def test_encoder_drift_blocks_close(self):self.assertNotIn('close',self.controller('drift')['steps'])
 def test_late_motion_blocks_descent(self):self.assertNotIn('descend',self.controller('late')['steps'])


class EncoderDriverTests(unittest.TestCase):
 def test_wraparound_atomic_registers_and_fault_detection(self):
  from types import SimpleNamespace
  from src.conveyor import Encoder
  class IO:
   def __init__(self):self.command=0;self.ack=0;self.index=-1;self.errors=0
   def write(self,offset,value):
    if offset==0:
     self.command=value
     if value&256!=self.ack:
      self.ack=value&256;self.index+=1
   def read(self,offset):
    if offset==0:return self.command
    return {0:0x4b4d3501,1:self.ack>>8,2:[0xfffffffe,2,6][self.index],3:100000000+20000000*self.index,4:0,5:self.errors}[self.command&15]
  io=IO();e=Encoder(SimpleNamespace(conveyor_gpio=SimpleNamespace(mmio=io)),cfg()['encoder'])
  first=e.sample();second=e.sample()
  self.assertEqual(first['position_mm'],0);self.assertAlmostEqual(second['position_mm'],.2);self.assertAlmostEqual(second['velocity_mm_s'],1)
  io.errors=1
  with self.assertRaisesRegex(RuntimeError,'quadrature'):e.sample()

if __name__=='__main__':unittest.main()
