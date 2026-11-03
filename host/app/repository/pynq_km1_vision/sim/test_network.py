
import base64,json,os,tempfile,threading,time,unittest
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import cv2,numpy as np
from src.pc_server import make_server
from src.remote import RemotePredictor
from src.control import same_target,claim_task,verify_outcome
from src.arm import execute,packet

TOKEN='test_token_with_32_characters_123456'
class Dummy:
    model_id='a'*64;metric='ridge_margin';labels=['block','washer'];providers=['numpy_cpu']
    def predict(self,crop):return {'label':'block','margin':.8,'score_kind':self.metric,'model_id':self.model_id,'providers':self.providers}

class NetworkTests(unittest.TestCase):
    def setUp(self):
        self.server=make_server(('127.0.0.1',0),Dummy(),TOKEN)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.url='http://127.0.0.1:'+str(self.server.server_port)
        self.env=patch.dict(os.environ,{'KM1_TOKEN':TOKEN});self.env.start()
        self.cfg={'url':self.url,'timeout_s':.5,'max_result_age_s':1,'score_kind':'ridge_margin'}
    def tearDown(self):self.server.shutdown();self.server.server_close();self.thread.join();self.env.stop()
    def test_real_http_roundtrip(self):
        out=RemotePredictor(self.cfg).predict(np.zeros((20,30,3),np.uint8),1,time.monotonic())
        self.assertEqual(out['label'],'block');self.assertGreater(out['request_bytes'],0)
    def test_stale_rejected(self):
        with self.assertRaises(TimeoutError):RemotePredictor(self.cfg).predict(np.zeros((20,30,3),np.uint8),1,time.monotonic()-2)
    def test_score_semantics_rejected(self):
        self.cfg['score_kind']='probability_margin'
        with self.assertRaises(ValueError):RemotePredictor(self.cfg).predict(np.zeros((20,30,3),np.uint8),1,time.monotonic())
    def test_model_mismatch_rejected(self):
        self.cfg['expected_model_id']='b'*64
        with self.assertRaises(ValueError):RemotePredictor(self.cfg).predict(np.zeros((20,30,3),np.uint8),1,time.monotonic())
    def test_authentication(self):
        with self.assertRaises(HTTPError) as e:urlopen(self.url+'/health')
        self.assertEqual(e.exception.code,401)
    def test_duplicate_and_malformed_request(self):
        _,png=cv2.imencode('.png',np.zeros((2,2,3),np.uint8))
        msg={'protocol':1,'frame_id':1,'request_id':'repeat-request-1','png':base64.b64encode(png).decode()}
        def post(data):return urlopen(Request(self.url+'/infer',json.dumps(data).encode(),{'Authorization':'Bearer '+TOKEN}))
        with post(msg) as response:self.assertEqual(response.status,200)
        with self.assertRaises(HTTPError) as e:post(msg)
        self.assertEqual(e.exception.code,409)
        msg['request_id']='different-request';msg['png']='bad'
        with self.assertRaises(HTTPError) as e:post(msg)
        self.assertEqual(e.exception.code,400)
    def test_reply_identity_rejected(self):
        class Response:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self,*args):return json.dumps({'protocol':1,'request_id':'wrong','frame_id':99}).encode()
        client=RemotePredictor(self.cfg)
        with patch.object(client.opener,'open',return_value=Response()):
            with self.assertRaises(ValueError):client.predict(np.zeros((2,2,3),np.uint8),1,time.monotonic())

    def test_board_remote_dry_and_moved_target_no_actuation(self):
        from src import baseline_run as run
        cfg=json.loads(Path('config/baseline_local.json').read_text())
        cfg['camera'].update(width=160,height=120);cfg['vision']['max_area']=10000
        cfg['inference'].update(mode='remote',url=self.url,timeout_s=.5,max_result_age_s=1.)
        cfg['arm']['commissioned']=True
        image=np.full((120,160,3),240,np.uint8);image[45:75,60:100]=0
        class Camera:
            def __init__(self,moved):self.fid=0;self.moved=moved
            def read(self,after=0,timeout=3):
                self.fid+=1;frame=image.copy()
                if self.moved and self.fid>5:frame=np.roll(frame,20,axis=1)
                return self.fid,time.monotonic(),frame
            def close(self):pass
        cwd=os.getcwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder);Path('data').mkdir()
                Path('data/calibration.json').write_text(json.dumps({'homography':[[1,0,-80],[0,1,110],[0,0,1]],'image_size':[160,120],
                    'pixel_hull':[[1,1],[158,1],[158,118],[1,118]]}))
                Path('config').mkdir(exist_ok=True)
                Path('config/baseline_local.json').write_text(json.dumps(cfg))
                with patch('src.baseline_run.LatestCamera',return_value=Camera(False)),patch('sys.argv',['run','--once','--frames','6']),patch('builtins.print'):
                    run.main()
                result=json.loads(Path('board/last_plan.json').read_text())
                self.assertEqual(result['prediction']['label'],'block');self.assertEqual(len(result['plan']),12)
                with patch('src.baseline_run.LatestCamera',return_value=Camera(True)),patch('sys.argv',['run','--execute','--once','--frames','6']),patch('builtins.print'),patch('serial.Serial') as serial:
                    with self.assertRaises(RuntimeError):run.main()
                    serial.assert_not_called()
            finally:os.chdir(cwd)

class ControlTests(unittest.TestCase):
    def test_target_motion_and_appearance(self):
        a={'center':[20,20],'bbox':[10,10,20,20],'area':400};b={**a,'center':[30,20]}
        frame=np.zeros((50,50,3),np.uint8)
        self.assertFalse(same_target(a,frame,b,frame,4))
        self.assertTrue(same_target(a,frame,a,frame,4))
        self.assertFalse(same_target(a,frame,a,np.full_like(frame,255),4))
    def test_task_journal_prevents_replay(self):
        with tempfile.TemporaryDirectory() as p:
            claim_task(p,'task-123',{})
            with self.assertRaises(FileExistsError):claim_task(p,'task-123',{})
    def test_post_grasp_visual_evidence(self):
        cfg={'vision':{'background_bgr':[240]*3,'threshold':100},'verification':{'bin_rois':{'block':[60,60,20,20]},'min_bin_gain_px':100}}
        before=np.full((100,100,3),240,np.uint8);before[10:30,10:30]=0
        after=np.full_like(before,240);after[60:80,60:80]=0
        out=verify_outcome(before,after,{'bbox':[10,10,20,20]},'block',cfg)
        self.assertEqual(out['status'],'visual_evidence_pass')
        self.assertEqual(verify_outcome(before,before,{'bbox':[10,10,20,20]},'block',cfg)['status'],'needs_review')
    def test_guard_failure_sends_no_commands(self):
        from unittest.mock import MagicMock
        port=MagicMock();cfg={'arm':{'commissioned':True,'port':'TEST','settle_s':0}}
        def guard():raise RuntimeError('moved')
        with patch('serial.Serial',return_value=port),patch('src.arm.time.sleep'):
            with self.assertRaises(RuntimeError):execute([('a',packet([(5,1000)],1000),1000)],cfg,True,before_motion=guard)
        port.write.assert_not_called();port.close.assert_called_once()

if __name__=='__main__':unittest.main(verbosity=2)
