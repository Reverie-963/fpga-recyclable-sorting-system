import base64,json,math,os,time,uuid
from urllib.request import Request,urlopen,build_opener,ProxyHandler
from urllib.parse import urlparse
import cv2

class RemotePredictor:
    def __init__(self,cfg):
        self.cfg=cfg;self.token=os.environ.get(cfg.get('token_env','KM1_TOKEN'),'')
        if len(self.token)<24:raise ValueError('Missing shared KM1_TOKEN (>=24 chars)')
        parsed=urlparse(cfg['url'])
        if parsed.scheme not in ('http','https') or not parsed.hostname or parsed.username:raise ValueError('Invalid inference URL')
        self.url=cfg['url'].rstrip('/');self.session=uuid.uuid4().hex;self.model_id=None
        # A LAN device URL must not be sent through desktop/system HTTP proxies.
        self.opener=build_opener(ProxyHandler({}))
    def predict(self,crop,frame_id,captured_at):
        rid=self.session+':'+str(frame_id)
        ok,png=cv2.imencode('.png',crop)
        if not ok:raise ValueError('PNG encode failure')
        msg={'protocol':1,'request_id':rid,'frame_id':frame_id,'png':base64.b64encode(png).decode()}
        raw=json.dumps(msg).encode();request=Request(self.url+'/infer',raw,{'Content-Type':'application/json','Authorization':'Bearer '+self.token})
        started=time.monotonic()
        with self.opener.open(request,timeout=self.cfg['timeout_s']) as response:
            payload=response.read(65537)
        if len(payload)>65536:raise ValueError('Response too large')
        result=json.loads(payload);now=time.monotonic()
        if now-captured_at>self.cfg['max_result_age_s']:raise TimeoutError('Expired vision result')
        if result.get('protocol')!=1 or result.get('request_id')!=rid or result.get('frame_id')!=frame_id:raise ValueError('Mismatched response')
        if result.get('score_kind')!=self.cfg['score_kind']:raise ValueError('Wrong score threshold semantics')
        margin=result.get('margin');label=result.get('label');model_id=result.get('model_id')
        if not isinstance(label,str) or not isinstance(margin,(int,float)) or not math.isfinite(margin) or margin<0:raise ValueError('Invalid prediction')
        if not isinstance(model_id,str) or len(model_id)!=64:raise ValueError('Missing model fingerprint')
        expected=self.cfg.get('expected_model_id') or self.model_id
        if expected and expected!=model_id:raise ValueError('Model changed during run')
        self.model_id=model_id
        result['rpc_ms']=(now-started)*1000;result['request_bytes']=len(raw)
        return result
