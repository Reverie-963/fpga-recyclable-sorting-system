"""Authenticated image-only routing for registered execution profiles."""
import base64
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import threading
import urllib.request
import uuid
from urllib.parse import urlsplit


class ProfileRouter:
    def __init__(self,catalog,request_json):
        self.profiles={p['id']:p for p in catalog['profiles']};self.default=catalog['default_model']
        self.request=request_json;self.lock=threading.RLock();self.routes={};self.defaults={}
        for p in self.profiles.values():
            if p.get('service_url'):
                route_id=uuid.uuid4().hex;self.routes[route_id]=(p['id'],p['service_url'],route_id);self.defaults[p['id']]=route_id

    def validate(self,p,value):
        actual=value.get('model_sha256') or value.get('model_version')
        if actual is None and p['model_sha256'] in value.get('model_hashes',{}).values():actual=p['model_sha256']
        if actual!=p['model_sha256']:raise ValueError('execution_upstream_weight_changed')
        if p.get('device') and value.get('execution',{}).get('device')!=p['device']:raise ValueError('execution_upstream_device_changed')
        expected=p.get('upstream_configuration_sha256')
        if expected and value.get('configuration_sha256')!=expected:raise ValueError('execution_upstream_configuration_changed')
        if not expected and 'settings' in value and value['settings']!=p['settings']:
            raise ValueError('execution_upstream_settings_changed')

    def register(self,data):
        if set(data)!={'profile_id','profile_sha256','url'}:raise ValueError('invalid_execution_route_registration')
        p=self.profiles[data['profile_id']]
        if p['profile_sha256']!=data['profile_sha256']:raise ValueError('execution_profile_changed')
        parsed=urlsplit(data['url'])
        if (parsed.scheme!='http' or parsed.hostname not in ('127.0.0.1','localhost') or parsed.path not in ('','/') or
                parsed.query or parsed.fragment or parsed.username or parsed.password):raise ValueError('execution_loopback_url_required')
        url=data['url'].rstrip('/')
        if url!=p.get('service_url') and parsed.port not in p['allowed_worker_ports']:raise ValueError('unregistered_execution_worker_port')
        health=self.request(url,'/health');self.validate(p,health)
        if not (health.get('ready') or health.get('model_loaded')):raise ValueError('execution_upstream_not_ready')
        with self.lock:
            if len(self.routes)>=4096:raise ValueError('execution_route_registration_limit')
            route_id=uuid.uuid4().hex;self.routes[route_id]=(p['id'],url,route_id)
        return dict(profile_id=p['id'],profile_sha256=p['profile_sha256'],route_id=route_id,registered=True)

    def route(self,profile_id,route_id=None):
        with self.lock:
            p=self.profiles[profile_id or self.default]
            route_id=route_id or self.defaults.get(p['id'])
            if route_id not in self.routes:raise ValueError('execution_profile_not_loaded')
            registered,url,nonce=self.routes[route_id]
            if registered!=p['id']:raise ValueError('execution_profile_route_mismatch')
            return p,url,nonce

    def generation(self,p,nonce,value):
        generation=value.get('generation',value.get('runtime_id'))
        if generation is None:raise ValueError('execution_upstream_generation_missing')
        return hashlib.sha256(json.dumps([p['profile_sha256'],nonce,generation]).encode()).hexdigest()

    def health(self,profile_id,route_id=None):
        p,url,nonce=self.route(profile_id,route_id);h=self.request(url,'/health');self.validate(p,h)
        return dict(ready=bool(h.get('ready') or h.get('model_loaded')),settings=p['settings'],
            generation=self.generation(p,nonce,h),model_hashes={p['policy']['model']['name']:p['model_sha256']},
            execution=h['execution'],station_filter_config_sha256=h.get('station_filter_config_sha256'),markers_only_supported=True,
            execution_profile=dict(id=p['id'],profile_sha256=p['profile_sha256'],configuration_sha256=p['upstream_configuration_sha256']))

    def predict(self,profile_id,payload,route_id=None):
        p,url,nonce=self.route(profile_id,route_id);r=self.request(url,'/detect',payload);self.validate(p,r)
        from src.prompted_recyclables import classify
        labels={row['official_label'] for row in p['policy']['classes']}
        instances=[]
        for obj in r['instances']:
            if obj.get('official_label',obj.get('label')) not in labels:
                raise ValueError('execution_unregistered_detection_class')
            instances.append(classify(p['policy'],p['settings'],obj))
        return dict(r,instances=instances,model_sha256=p['model_sha256'],settings_sha256=p['settings']['settings_sha256'],
            policy_sha256=p['settings']['policy_sha256'],generation=self.generation(p,nonce,r),
            execution_profile=dict(id=p['id'],profile_sha256=p['profile_sha256'],configuration_sha256=p['upstream_configuration_sha256']))


def make_execution_visual_relay(port,token,catalog,camera_path):
    import cv2
    import numpy as np
    from src.station_geometry import detect_markers
    camera_path=Path(camera_path);camera_raw=camera_path.read_bytes();camera=json.loads(camera_raw)
    camera_hash=hashlib.sha256(camera_raw).hexdigest();opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def request(base,path,payload=None):
        req=urllib.request.Request(base+path,data=json.dumps(payload).encode() if payload is not None else None,
            headers={'Content-Type':'application/json'})
        with opener.open(req,timeout=12) as response:raw=response.read(18_000_001)
        if len(raw)>18_000_000:raise ValueError('execution_reply_too_large')
        return json.loads(raw)
    router=ProfileRouter(catalog,request);gate=threading.BoundedSemaphore(3)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,status,value):
            raw=json.dumps(value,allow_nan=False).encode()
            try:
                self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Cache-Control','no-store')
                self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
            except OSError:pass
        def authorized(self):return hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+token)
        def do_GET(self):
            if self.path!='/health':return self.reply(404,dict(error='image_relay_only'))
            try:return self.reply(200,router.health(self.headers.get('X-Execution-Profile'),self.headers.get('X-Execution-Route')))
            except (OSError,ValueError,KeyError):return self.reply(503,dict(error='execution_visual_profile_unavailable'))
        def do_POST(self):
            if not self.authorized():return self.reply(403,dict(error='execution_visual_authentication_required'))
            if self.path not in ('/detect','/markers','/execution-model/register'):return self.reply(404,dict(error='image_relay_only'))
            if not gate.acquire(blocking=False):return self.reply(409,dict(error='execution_visual_capacity_busy'))
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=12_000_000 or self.headers.get('Transfer-Encoding'):raise ValueError('invalid_execution_visual_body')
                raw=self.rfile.read(size)
                if len(raw)!=size:raise ValueError('truncated_execution_visual_body')
                data=json.loads(raw)
                if self.path=='/execution-model/register':return self.reply(200,router.register(data))
                if set(data)-{'image_base64','control_only'}:raise ValueError('execution_image_only')
                image=cv2.imdecode(np.frombuffer(base64.b64decode(data['image_base64'],validate=True),np.uint8),cv2.IMREAD_COLOR)
                if image is None or list(image.shape[1::-1])!=camera['image_size']:raise ValueError('execution_image_size_changed')
                if camera_path.read_bytes()!=camera_raw:raise ValueError('execution_camera_parameters_changed')
                profile_id=self.headers.get('X-Execution-Profile');route_id=self.headers.get('X-Execution-Route')
                if self.path=='/markers':
                    health=router.health(profile_id,route_id)
                    selected,_,_=router.route(profile_id,route_id)
                    detections=detect_markers(image,camera)
                    if router.health(profile_id,route_id)!=health:raise ValueError('execution_marker_generation_changed')
                    result=dict(health,model_sha256=selected['model_sha256'],
                        settings_sha256=health['settings']['settings_sha256'],policy_sha256=health['settings']['policy_sha256'],
                        source_bgr_sha256=hashlib.sha256(image.tobytes()).hexdigest(),image_size=list(image.shape[1::-1]),
                        scope='fixed_markers_only',semantic_inference_performed=False,marker_compute_device='pc_cpu')
                else:
                    result=router.predict(profile_id,dict(image_base64=data['image_base64'],control_only=True),route_id)
                    detections=detect_markers(image,camera)
                digest=hashlib.sha256(image.tobytes()).hexdigest()
                if result.get('source_bgr_sha256')!=digest:raise ValueError('execution_visual_pixels_changed')
                result.pop('overlay_base64',None)
                result['fixed_marker_observation']=dict(location='pc_vision',camera_sha256=camera_hash,source_bgr_sha256=digest,detections=detections)
                return self.reply(200,result)
            except (OSError,ValueError,KeyError,TypeError):return self.reply(503,dict(error='execution_visual_identity_or_transport_failed'))
            finally:gate.release()
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler);server.profile_router=router
    return server
