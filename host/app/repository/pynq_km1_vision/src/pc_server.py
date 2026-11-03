"""LAN-only authenticated image inference. This process has no actuator API."""
import argparse,base64,collections,hmac,json,os,struct,time
from http.server import BaseHTTPRequestHandler,HTTPServer
from pathlib import Path
import cv2,numpy as np
from .inference import Predictor
MAX_BODY=2_000_000

def make_server(address,predictor,token):
    if len(token)<24:raise ValueError('KM1_TOKEN must have at least 24 characters')
    seen=collections.OrderedDict()
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup();self.connection.settimeout(3)
        def log_message(self,*args):pass
        def respond(self,status,payload):
            data=json.dumps(payload,allow_nan=False).encode();self.send_response(status)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers()
            try:self.wfile.write(data)
            except (BrokenPipeError,ConnectionResetError):pass
        def authorized(self):return hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+token)
        def do_GET(self):
            if not self.authorized():return self.respond(401,{'error':'unauthorized'})
            if self.path!='/health':return self.respond(404,{'error':'not_found'})
            self.respond(200,{'protocol':1,'model_id':predictor.model_id,'score_kind':predictor.metric,'labels':predictor.labels,'providers':predictor.providers})
        def do_POST(self):
            if not self.authorized():return self.respond(401,{'error':'unauthorized'})
            if self.path!='/infer':return self.respond(404,{'error':'not_found'})
            try:
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<=MAX_BODY:raise ValueError('Payload too large or empty')
                raw=self.rfile.read(n)
                if len(raw)!=n:raise ValueError('Truncated request')
                msg=json.loads(raw);rid=msg['request_id'];fid=msg['frame_id']
                if msg.get('protocol')!=1 or not isinstance(rid,str) or not 8<=len(rid)<=120 or type(fid)!=int or fid<0:raise ValueError('Bad request identity')
                if rid in seen:return self.respond(409,{'error':'duplicate_request'})
                png=base64.b64decode(msg['png'],validate=True)
                if len(png)<24 or png[:8]!=b'\x89PNG\r\n\x1a\n':raise ValueError('Expected PNG')
                w,h=struct.unpack('>II',png[16:24])
                if not (1<=w<=1920 and 1<=h<=1920 and w*h<=1920*1080):raise ValueError('Image dimensions too large')
                crop=cv2.imdecode(np.frombuffer(png,np.uint8),cv2.IMREAD_COLOR)
                if crop is None or crop.shape[:2]!=(h,w):raise ValueError('Bad PNG')
                seen[rid]=None
                while len(seen)>4096:seen.popitem(last=False)
                t=time.perf_counter();result=predictor.predict(crop)
                self.respond(200,{'protocol':1,'request_id':rid,'frame_id':fid,'inference_ms':(time.perf_counter()-t)*1000,**result})
            except (ValueError,KeyError,TypeError,TimeoutError):self.respond(400,{'error':'invalid_request'})
            except Exception:self.respond(500,{'error':'inference_failed'})
    # One request in flight: no unbounded inference queue or model thread races.
    return HTTPServer(address,Handler)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='config/pc_ridge.json');p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8765);a=p.parse_args()
    model=Predictor(json.loads(Path(a.config).read_text(encoding='utf-8-sig')));server=make_server((a.host,a.port),model,os.environ.get('KM1_TOKEN',''))
    print(json.dumps({'listening':server.server_address,'model_id':model.model_id,'providers':model.providers,'score_kind':model.metric}),flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        server.server_close();profile=model.close()
        if profile:print('ORT profile:',profile)
if __name__=='__main__':main()
