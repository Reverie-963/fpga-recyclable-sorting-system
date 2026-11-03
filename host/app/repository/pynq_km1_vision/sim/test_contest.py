import copy,json,os,tempfile,threading,time,unittest
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import cv2,numpy as np
from src.detection import validate_detections
from src.detect_server import make_server
from src.detect_remote import DetectionClient
from src.grasp_geometry import fuse,match_target,verify_masks
from src.contest_run import run
TOKEN='test_token_012345678901234567890123'

def cfg():
 c=json.loads(Path('config/static_board.json').read_text());c['vision'].update(backend='cpu',min_area=10,max_area=3000,stable_frames=1)
 c['camera'].update(width=120,height=100);c['grasp'].update(clearance_px=3,min_interior_px=2)
 c['arm']['commissioned']=True;c['verification']['bin_rois']={'block':[90,70,25,25],'washer':[90,40,25,25]}
 return c

class Dummy:
 model_id='a'*64;runtime_id='b'*64;labels=['block'];providers=['test']
 def predict(self,im):
  return {'model_id':self.model_id,'runtime_id':self.runtime_id,'providers':self.providers,
    'image_size':[im.shape[1],im.shape[0]],'detections':[{'label':'block','confidence':.9,'xyxy':[10,10,25,25]}],'timing':{'total_ms':1}}

class ContestTests(unittest.TestCase):
 def test_invalid_detection_rejected(self):
  for box,score in [([0,0,200,5],.9),([4,0,2,5],.9),([0,0,5,5],float('nan'))]:
   with self.assertRaises(ValueError):validate_detections([{'label':'block','confidence':score,'xyxy':box}],100,100)
 def test_ring_grasp_avoids_hole(self):
  mask=np.zeros((100,120),np.uint8);cv2.circle(mask,(40,40),15,255,-1);cv2.circle(mask,(40,40),7,0,-1)
  choices=fuse([{'label':'washer','confidence':.9,'xyxy':[24,24,56,56]}],mask,cfg(),None)
  self.assertEqual(len(choices),1);x,y=map(int,choices[0]['center']);self.assertEqual(mask[y,x],255)
  self.assertGreater(np.hypot(x-40,y-40),7)
 def test_multiple_objects_and_ambiguous_boxes(self):
  mask=np.zeros((100,120),np.uint8);mask[10:25,10:25]=255;mask[40:60,50:70]=255
  ds=[{'label':'block','confidence':.9,'xyxy':[9,9,26,26]},{'label':'washer','confidence':.8,'xyxy':[49,39,71,61]}]
  self.assertEqual(len(fuse(ds,mask,cfg(),None)),2)
  ds.append({'label':'washer','confidence':.7,'xyxy':[9,9,26,26]})
  self.assertEqual(len(fuse(ds,mask,cfg(),None)),1)
 def test_verification_requires_source_and_bin(self):
  before=np.zeros((100,120),np.uint8);before[10:25,10:25]=255;after=before.copy();after[75:90,95:110]=255
  target={'bbox':[10,10,15,15]}
  self.assertEqual(verify_masks(before,after,target,'block',cfg())['status'],'needs_review')
  after[10:25,10:25]=0
  self.assertEqual(verify_masks(before,after,target,'block',cfg())['status'],'visual_evidence_pass')
 def test_protocol_v2_roundtrip_freshness_fingerprint_and_auth(self):
  model=Dummy();server=make_server(('127.0.0.1',0),model,TOKEN);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
  try:
   c=cfg()['inference'];c.update(url='http://127.0.0.1:'+str(server.server_port),timeout_s=.5,max_result_age_s=1)
   with patch.dict(os.environ,KM1_TOKEN=TOKEN):
    client=DetectionClient(c);frame=np.zeros((100,120,3),np.uint8)
    result=client.predict(frame,1,time.monotonic());self.assertEqual(len(result['detections']),1)
    with self.assertRaises(TimeoutError):client.predict(frame,2,time.monotonic()-2)
    model.runtime_id='c'*64
    with self.assertRaisesRegex(ValueError,'Changed runtime'):client.predict(frame,3,time.monotonic())
   with self.assertRaises(HTTPError) as err:urlopen(c['url']+'/health')
   self.assertEqual(err.exception.code,401)
  finally:server.shutdown();server.server_close();thread.join()
 def controller(self,verified=True,changing=False):
  c=cfg();c['inference']['max_result_age_s']=2
  before=np.full((100,120,3),240,np.uint8);before[15:35,20:40]=0
  after=before.copy()
  if verified:after[15:35,20:40]=240;after[74:94,94:114]=0
  state={'moved':False,'writes':0,'queries':0}
  class Camera:
   def __init__(self,*a):self.i=0
   def read(self,after=0):
    self.i+=1
    return self.i,time.monotonic(),globals_frame()
   def close(self):pass
  def globals_frame():return after.copy() if state['moved'] else before.copy()
  class Client:
   def __init__(self,*a):pass
   def predict(self,frame,fid,stamp):
    state['queries']+=1
    ds=[] if changing and state['queries']>1 else [{'label':'block','confidence':.9,'xyxy':[19,14,41,36]}]
    return {'detections':ds,'rpc_ms':1,'timing':{'total_ms':1}}
  def executor(plan,c,enabled=False,before_motion=None):
   if not enabled:return []
   if before_motion:before_motion()
   state['writes']+=1;state['moved']=True
   return {'status':'sequence_sent_unverified'}
  with tempfile.TemporaryDirectory() as d:
   old=os.getcwd();os.chdir(d)
   try:
    Path('cal.json').write_text(json.dumps({'image_size':[120,100],'pixel_hull':[[1,1],[80,1],[80,68],[1,68]],'homography':[[.1,0,0],[0,.1,150],[0,0,1]]}))
    c['calibration']='cal.json'
    with patch('src.contest_run.LatestCamera',Camera),patch('src.contest_run.DetectionClient',Client),patch('src.contest_run.execute',executor):
     if changing or not verified:
      with self.assertRaises(RuntimeError):run(c,True,max_tasks=1,frames=3)
     else:self.assertEqual(run(c,True,max_tasks=1,frames=3)['completed'],1)
   finally:os.chdir(old)
  return state
 def test_batch_verified_then_finishes(self):self.assertEqual(self.controller()['writes'],1)
 def test_uncertain_placement_stops_without_retry(self):self.assertEqual(self.controller(verified=False)['writes'],1)
 def test_changed_target_sends_no_motion(self):self.assertEqual(self.controller(changing=True)['writes'],0)

if __name__=='__main__':unittest.main()

class ContinuousBatchTests(unittest.TestCase):
 def test_two_tasks_reobserve_and_verify_before_continuing(self):
  c=cfg();c['verification']['min_bin_gain_px']=20
  state={'step':0,'calls':0}
  def frame():
   im=np.full((100,120,3),240,np.uint8)
   if state['step']<1:im[15:30,15:30]=0
   if state['step']<2:im[15:30,50:65]=0
   if state['step']>=1:im[75:82,95:105]=0
   if state['step']>=2:im[85:92,95:105]=0
   return im
  class Camera:
   def __init__(self,*a):self.i=0
   def read(self,after=0):self.i+=1;return self.i,time.monotonic(),frame()
   def close(self):pass
  class Client:
   def __init__(self,*a):pass
   def predict(self,im,fid,stamp):
    ds=[]
    if state['step']<1:ds.append({'label':'block','confidence':.95,'xyxy':[14,14,31,31]})
    if state['step']<2:ds.append({'label':'block','confidence':.9,'xyxy':[49,14,66,31]})
    return {'detections':ds,'timing':{'total_ms':1}}
  def execute_mock(plan,config,enabled=False,before_motion=None):
   if not enabled:return []
   before_motion();state['step']+=1;state['calls']+=1;return {'status':'sequence_sent_unverified'}
  with tempfile.TemporaryDirectory() as d:
   old=os.getcwd();os.chdir(d)
   try:
    Path('cal.json').write_text(json.dumps({'image_size':[120,100],'pixel_hull':[[1,1],[80,1],[80,68],[1,68]],'homography':[[.1,0,0],[0,.1,150],[0,0,1]]}));c['calibration']='cal.json'
    with patch('src.contest_run.LatestCamera',Camera),patch('src.contest_run.DetectionClient',Client),patch('src.contest_run.execute',execute_mock):
     result=run(c,True,2,10)
    self.assertEqual(result['completed'],2);self.assertEqual(state['calls'],2)
    self.assertEqual(len(list(Path('board/tasks').glob('*.json'))),2)
   finally:os.chdir(old)
