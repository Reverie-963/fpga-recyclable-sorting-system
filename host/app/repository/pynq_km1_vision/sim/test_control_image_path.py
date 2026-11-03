import base64,json,tempfile,threading,unittest,urllib.request,urllib.error
from pathlib import Path
import cv2,numpy as np
from src.yoloe_preview_server import make_server
class Detector:
 models={'supervised':None}
 def __init__(self):self.normal=0;self.control=0
 def predict(self,image,prompt_mode=None):self.normal+=1;return dict(image_size=[8,6],instances=[],model_version='weight')
 def predict_control(self,image,prompt_mode=None):self.control+=1;return dict(image_size=[8,6],instances=[],model_version='weight')
 def health(self):return {'model_loaded':True}
class Tests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.output=Path(self.temp.name)/'uploads';self.detector=Detector()
  self.server=make_server(self.detector,0,self.output);threading.Thread(target=self.server.serve_forever,daemon=True).start();self.addCleanup(self.server.server_close);self.addCleanup(self.server.shutdown)
  _,png=cv2.imencode('.png',np.zeros((6,8,3),np.uint8));self.payload=dict(image_base64=base64.b64encode(png).decode(),prompt_mode='supervised')
 def post(self,flag):
  body=dict(self.payload,control_only=flag);req=urllib.request.Request('http://127.0.0.1:'+str(self.server.server_port)+'/detect',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
  return json.load(urllib.request.urlopen(req,timeout=3))
 def test_control_has_same_semantics_no_preview_or_disk_artifacts(self):
  result=self.post(True);self.assertEqual(result['instances'],[]);self.assertTrue(result['control_only']);self.assertEqual(self.detector.control,1);self.assertEqual(self.detector.normal,0);self.assertEqual(list(self.output.iterdir()),[]);self.assertNotIn('overlay_base64',result)
 def test_normal_preview_retains_artifact_recording(self):
  self.post(False);self.assertEqual(self.detector.normal,1);self.assertEqual(len(list(self.output.iterdir())),1)
 def test_invalid_boolean_rejected_before_inference(self):
  with self.assertRaises(urllib.error.HTTPError) as e:self.post('true')
  self.assertEqual(e.exception.code,400);self.assertEqual(self.detector.control,0)
