import base64,copy,hashlib,json,sys,tempfile,threading,unittest,urllib.request,urllib.error
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]));sys.path.insert(0,str(Path(__file__).parents[2]/'pynq_km1_vision'))
from execution_visual_relay import make_execution_visual_relay
class Tests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.detect_calls=0;self.changing=False;self.health_calls=0;outer=self
  class Upstream(BaseHTTPRequestHandler):
   def log_message(self,*a):pass
   def do_GET(self):
    outer.health_calls+=1;generation=str(outer.health_calls) if outer.changing else 'stable'
    body=json.dumps(dict(model_sha256='weight',configuration_sha256='configuration',runtime_id=generation,ready=True,execution={'device':'cuda:0'})).encode();self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
   def do_POST(self):outer.detect_calls+=1;self.send_error(500)
  u=ThreadingHTTPServer(('127.0.0.1',0),Upstream);threading.Thread(target=u.serve_forever,daemon=True).start();self.addCleanup(u.server_close);self.addCleanup(u.shutdown)
  p=dict(id='p',profile_sha256='profile',model_sha256='weight',upstream_configuration_sha256='configuration',device='cuda:0',policy={'model':{'name':'x'}},settings={'settings_sha256':'settings','policy_sha256':'policy'},service_url='http://127.0.0.1:'+str(u.server_port))
  camera=Path(self.temp.name)/'camera.json';camera.write_text(json.dumps({'image_size':[8,6]}));self.camera_hash=hashlib.sha256(camera.read_bytes()).hexdigest()
  marker_patch=patch('src.station_geometry.detect_markers',return_value=[{'id':1}]);marker_patch.start();self.addCleanup(marker_patch.stop)
  self.server=make_execution_visual_relay(0,'private-token',{'default_model':'p','profiles':[p]},camera);threading.Thread(target=self.server.serve_forever,daemon=True).start();self.addCleanup(self.server.server_close);self.addCleanup(self.server.shutdown)
  self.image=np.zeros((6,8,3),np.uint8);_,png=cv2.imencode('.png',self.image);self.payload=dict(image_base64=base64.b64encode(png).decode(),control_only=True)
 def request(self,token='private-token'):
  req=urllib.request.Request('http://127.0.0.1:'+str(self.server.server_port)+'/markers',data=json.dumps(self.payload).encode(),headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
  return json.load(urllib.request.urlopen(req,timeout=3))
 def test_exact_frame_markers_without_neural_inference(self):
  r=self.request();self.assertFalse(r['semantic_inference_performed']);self.assertEqual(r['scope'],'fixed_markers_only');self.assertEqual(r['fixed_marker_observation']['camera_sha256'],self.camera_hash);self.assertEqual(r['source_bgr_sha256'],hashlib.sha256(self.image.tobytes()).hexdigest());self.assertEqual(self.detect_calls,0);self.assertEqual(self.health_calls,2)
 def test_generation_change_rejects_markers(self):
  self.changing=True
  with self.assertRaises(urllib.error.HTTPError) as e:self.request()
  self.assertEqual(e.exception.code,503);self.assertEqual(self.detect_calls,0)
 def test_authentication_required_before_processing(self):
  with self.assertRaises(urllib.error.HTTPError) as e:self.request('wrong')
  self.assertEqual(e.exception.code,403);self.assertEqual(self.health_calls,0)
