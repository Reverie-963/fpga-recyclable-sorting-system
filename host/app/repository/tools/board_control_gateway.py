"""A PC relay for board task admission; all planning remains on the board."""
import http.client
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import hmac
import json
import threading
import time
import urllib.request
import uuid

from station_vision_bridge import _BorrowedChannelSocket


class BoardRPC:
    def __init__(self,client,port,token):
        self.client,self.port,self.token=client,port,token
        self.transport=client.get_transport()
        self.lock=threading.Lock()

    def request(self,path,data=None):
        transport=self.client.get_transport()
        if transport is not self.transport or not transport.is_active():
            raise ConnectionError('board_control_ssh_lost_no_reconnect')
        with self.lock:
            channel=transport.open_channel('direct-tcpip',('127.0.0.1',self.port),('127.0.0.1',0),timeout=3)
        channel.settimeout(3)
        connection=http.client.HTTPConnection('127.0.0.1',self.port,timeout=3)
        connection.sock=_BorrowedChannelSocket(channel)
        try:
            connection.request('POST' if data is not None else 'GET',path,
                body=json.dumps(data).encode() if data is not None else None,
                headers={'Authorization':'Bearer '+self.token,'Content-Type':'application/json'})
            response=connection.getresponse();raw=response.read(8_500_001)
            if len(raw)>8_500_000:raise ValueError('oversized_board_task_reply')
            value=json.loads(raw)
            if response.status>=400:raise ValueError(value.get('error','board_task_rejected'))
            return value
        finally:connection.close();channel.close()


def make_gateway(rpc,token,upstream='http://127.0.0.1:8811',port=18813,idle_check=lambda:None,timing_provider=None,lease_factory=None,motion_options_provider=None):
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def plan(payload):
        if not isinstance(payload,dict) or set(payload)-{'material','client','request_id','object_height_mm','height_plan'}:
            raise ValueError('board_preview_accepts_material_only')
        from src.prompted_height import object_height
        object_height(payload.get('object_height_mm'))
        from src.sorting_item_heights import height_plan
        height_plan(payload.get('height_plan'))
        if payload.get('object_height_mm') is not None and payload.get('height_plan') is not None:
            raise ValueError('instance_and_shared_height_conflict')
        idle_check()
        # During transition the existing PC executor must see the same lease.
        # A board-only lease cannot prevent old PC entry points from starting.
        with lease_factory() if lease_factory is not None else nullcontext() as owner:
            return owned_plan(payload,owner.check if owner is not None else lambda:None)

    def owned_plan(payload,owner_check):
        owner_check()
        with opener.open(upstream+'/health',timeout=3) as response:health=json.load(response)
        expected={k:health[k] for k in ('settings','generation','model_hashes','execution')}
        if 'station_filter_config_sha256' in health:
            expected['station_filter_config_sha256']=health['station_filter_config_sha256']
        if payload.get('object_height_mm') is not None:expected['object_height_mm']=payload['object_height_mm']
        if payload.get('height_plan') is not None:
            expected['height_plan']=payload['height_plan']
            expected['configuration_sha256']=health.get('configuration_sha256',health['settings']['settings_sha256'])
        if timing_provider is not None:expected['observation_timing']=timing_provider()
        if motion_options_provider is not None:expected['motion_options']=motion_options_provider()
        request=dict(owner=payload.get('client','console-8810'),request_id=payload.get('request_id',uuid.uuid4().hex),
                     material=payload.get('material','AUTO'),expected=expected)
        accepted=rpc.request('/tasks',request)
        started=heartbeat=time.monotonic()
        try:
            while time.monotonic()-started<90:
                owner_check()
                if time.monotonic()-heartbeat>=2:
                    rpc.request('/heartbeat',dict(lease=accepted['lease']));heartbeat=time.monotonic()
                result=rpc.request('/tasks/'+accepted['task_id'])
                if result['status']=='finished':return result['report']
                time.sleep(.1)
            raise TimeoutError('board_preview_deadline_exceeded')
        except (OSError,ValueError):
            try:rpc.request('/stop',dict(reason='Board preview transport/ownership ended; no physical motion was enabled.'))
            except (OSError,ValueError):pass
            raise

    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def setup(self):super().setup();self.request.settimeout(5)
        def allowed(self):
            return (self.headers.get('Host') in ('127.0.0.1:'+str(server.server_port),'localhost:'+str(server.server_port)) and
                not self.headers.get('Origin') and hmac.compare_digest(self.headers.get('Authorization','').encode(),('Bearer '+token).encode()))
        def reply(self,status,value):
            body=json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
            try:
                self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
                self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store')
                self.end_headers();self.wfile.write(body)
            except OSError:pass
        def do_GET(self):
            if not self.allowed():return self.reply(403,dict(error='local_board_gateway_auth_required'))
            if self.path!='/health':return self.reply(404,dict(error='not_found'))
            try:return self.reply(200,rpc.request('/health'))
            except (OSError,ValueError) as exc:return self.reply(503,dict(error=str(exc)))
        def do_POST(self):
            if not self.allowed():return self.reply(403,dict(error='local_board_gateway_auth_required'))
            if self.path!='/plan':return self.reply(404,dict(error='board_gateway_preview_only'))
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=524288 or self.headers.get('Transfer-Encoding'):raise ValueError('invalid_board_gateway_body')
                raw=self.rfile.read(size)
                if len(raw)!=size:raise ValueError('truncated_board_gateway_body')
                return self.reply(200,plan(json.loads(raw)))
            except (OSError,ValueError,KeyError,TypeError) as exc:return self.reply(409,dict(error=str(exc)))
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler);server.daemon_threads=True
    return server
