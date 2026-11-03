"""PC visual relay and thin HTTP/SSH links; no planning or actuator imports."""
import base64
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import http.client
import json
from pathlib import Path
import select
import socket
import threading
import time
import urllib.request
from urllib.parse import urlsplit
from paramiko import SSHException


class _BorrowedChannelSocket:
    """Keep an HTTP/1.0 response's channel alive until its body is consumed."""
    def __init__(self,channel):self.channel=channel
    def __getattr__(self,name):return getattr(self.channel,name)
    def close(self):pass


def preview_display(report):
    """Project board-produced facts into the existing perception page schema."""
    if report.get('compute_location')!='pynq_ps':raise ValueError('board_preview_identity_required')
    choice,steps=report.get('choice'),report.get('plan',[])
    ready=report.get('status')=='preview_only' and isinstance(choice,dict) and bool(steps)
    decisions=[dict(official_label=row['label'],material=row.get('material'),confidence=row.get('confidence'))
        for frame in report.get('original_labels_by_frame',[]) for row in frame]
    return dict(status='preview_ready' if ready else report.get('status','blocked'),reason=report.get('reason'),
        candidates=[dict(choice=choice,steps=steps)] if ready else [],rejected=report.get('rejected_candidates',[]),
        decisions=decisions,compute_location='pynq_ps',physical_motion=False,board_report=report)


def make_visual_relay(port,token,upstream,camera_path):
    import cv2
    import numpy as np
    from src.station_geometry import detect_markers
    camera_path=Path(camera_path)
    camera_raw=camera_path.read_bytes();camera=json.loads(camera_raw)
    camera_hash=hashlib.sha256(camera_raw).hexdigest()
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    gate=threading.BoundedSemaphore(3)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,code,value):
            raw=json.dumps(value,allow_nan=False).encode()
            try:
                self.send_response(code);self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store')
                self.end_headers();self.wfile.write(raw)
            except OSError:pass
        def forward(self,payload=None):
            request=urllib.request.Request(upstream+self.path,
                data=json.dumps(payload).encode() if payload is not None else None,
                headers={'Content-Type':'application/json'})
            with opener.open(request,timeout=12) as response:raw=response.read(18_000_001)
            if len(raw)>18_000_000:raise ValueError('visual_reply_too_large')
            return json.loads(raw)
        def do_GET(self):
            if self.path!='/health':return self.reply(404,{'error':'vision_only'})
            try:return self.reply(200,self.forward())
            except Exception:return self.reply(503,{'error':'visual_upstream_unavailable'})
        def do_POST(self):
            if self.path!='/detect':return self.reply(404,{'error':'vision_only_no_planning'})
            if not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+token):
                return self.reply(403,{'error':'visual_authentication_required'})
            if not gate.acquire(blocking=False):return self.reply(409,{'error':'visual_capacity_busy_no_queue'})
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=12_000_000 or self.headers.get('Transfer-Encoding'):raise ValueError('invalid_visual_body')
                raw=self.rfile.read(size)
                if len(raw)!=size:raise ValueError('truncated_visual_body')
                payload=json.loads(raw)
                if set(payload)-{'image_base64','control_only'}:raise ValueError('visual_image_only')
                image=cv2.imdecode(np.frombuffer(base64.b64decode(payload['image_base64'],validate=True),np.uint8),cv2.IMREAD_COLOR)
                if image is None or list(image.shape[1::-1])!=camera['image_size']:raise ValueError('visual_input_size_changed')
                if camera_path.read_bytes()!=camera_raw:raise ValueError('visual_camera_file_changed')
                value=self.forward(dict(image_base64=payload['image_base64'],control_only=True))
                digest=hashlib.sha256(image.tobytes()).hexdigest()
                if value.get('source_bgr_sha256')!=digest:raise ValueError('visual_pixels_changed')
                value.pop('overlay_base64',None)
                value['fixed_marker_observation']=dict(location='pc_vision',camera_sha256=camera_hash,
                    source_bgr_sha256=digest,detections=detect_markers(image,camera))
                return self.reply(200,value)
            except (OSError,ValueError,KeyError,TypeError):return self.reply(503,{'error':'visual_identity_or_transport_failed'})
            finally:gate.release()
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)


class ReverseVisualLink:
    def __init__(self,client,port):
        self.client,self.port=client,port;self.transport=client.get_transport()
        self.stop=threading.Event();self.gate=threading.BoundedSemaphore(4)
        self.channels=set();self.lock=threading.Lock()
    def start(self):
        def accepted(channel,*unused):
            if not self.gate.acquire(blocking=False):channel.close();return
            def pump():
                sock=None
                with self.lock:self.channels.add(channel)
                try:
                    sock=socket.create_connection(('127.0.0.1',self.port),timeout=3)
                    sock.settimeout(15);channel.settimeout(15)
                    last=time.monotonic()
                    while not self.stop.is_set():
                        if time.monotonic()-last>15:return
                        readable,_,_=select.select([sock,channel],[],[],.25)
                        for current in readable:
                            data=current.recv(65536)
                            if not data:return
                            last=time.monotonic()
                            (channel if current is sock else sock).sendall(data)
                except (OSError,EOFError):pass
                finally:
                    channel.close()
                    if sock:sock.close()
                    with self.lock:self.channels.discard(channel)
                    self.gate.release()
            threading.Thread(target=pump,daemon=True).start()
        self.transport.request_port_forward('127.0.0.1',self.port,handler=accepted)
    def close(self):
        self.stop.set()
        try:self.transport.cancel_port_forward('127.0.0.1',self.port)
        except Exception:pass
        with self.lock:
            for channel in list(self.channels):channel.close()


class PinnedBoardHTTP:
    channel_open_timeout_s=3
    control_response_timeout_s=4
    def __init__(self,client,port,token,*,control_client=None):
        self.client,self.port,self.token=client,port,token
        self.transport=client.get_transport();self.lock=threading.Lock()
        self.control_client=control_client
        self.control_transport=control_client.get_transport() if control_client is not None else None
        if control_client is not None and self.control_transport is self.transport:
            raise ValueError('independent_control_transport_required')
        self.control_lock=threading.Lock()
        self.display_gate=threading.BoundedSemaphore(3)
    def request(self,path,body=None,content_type='application/json'):
        route=urlsplit(path).path
        control=(route in ('/operation','/board-control','/execution-model') or route.startswith(('/operation/','/execution-model/')) or
                 route=='/board-preview-report' and body is None)
        client=self.control_client if control and self.control_client is not None else self.client
        transport=self.control_transport if control and self.control_client is not None else self.transport
        lock=self.control_lock if control and self.control_client is not None else self.lock
        if client.get_transport() is not transport or not transport.is_active():
            raise ConnectionError('board_control_link_lost_no_reconnect')
        # Display traffic has no queue and cannot consume the dedicated control
        # connection. Do not retry a POST whose admission/result is unknown.
        if not control and not self.display_gate.acquire(blocking=False):
            raise ConnectionError('board_display_capacity_busy_no_queue')
        channel=connection=None
        try:
            if not lock.acquire(timeout=.25):raise ConnectionError('board_channel_open_busy_no_queue')
            try:
                channel=transport.open_channel('direct-tcpip',('127.0.0.1',self.port),('127.0.0.1',0),timeout=self.channel_open_timeout_s)
            finally:lock.release()
            timeout=self.control_response_timeout_s if control else 15
            if control and body is not None:
                try:action=json.loads(body).get('action')
                except (ValueError,TypeError,AttributeError):action=None
                if route=='/operation' and action=='reset' or route=='/execution-model/bind':timeout=12
            channel.settimeout(timeout)
            connection=http.client.HTTPConnection('127.0.0.1',self.port,timeout=timeout)
            connection.sock=_BorrowedChannelSocket(channel)
            connection.request('POST' if body is not None else 'GET',path,body=body,headers={
                'Authorization':'Bearer '+self.token,'Content-Type':content_type})
            reply=connection.getresponse();raw=reply.read(18_000_001)
            if len(raw)>18_000_000:raise ValueError('board_control_reply_too_large')
            return reply.status,raw,dict(reply.getheaders())
        except (SSHException,EOFError) as exc:
            raise ConnectionError('board_control_ssh_unavailable_no_retry') from exc
        finally:
            if connection is not None:connection.close()
            if channel is not None:channel.close()
            if not control:self.display_gate.release()


def make_control_proxy(port,board,visual_console):
    server=None
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    visual_routes={'/health','/live-result','/latest-result','/overlay.jpg','/camera.jpg','/live','/detect','/examples'}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def allowed(self):
            hosts={'127.0.0.1:'+str(server.server_port),'localhost:'+str(server.server_port)}
            return self.headers.get('Host') in hosts and (not self.headers.get('Origin') or
                self.headers['Origin'] in {'http://'+h for h in hosts})
        def reply(self,status,raw,headers=None):
            try:
                mime=next((v for k,v in (headers or {}).items() if k.lower()=='content-type'),'application/json')
                if mime.startswith('text/html'):
                    raw=raw.replace('读取三个新相机帧'.encode(),'读取当前单帧相机图像'.encode())
                    raw=raw.replace('三帧稳定性'.encode(),'当前单帧观察'.encode())
                self.send_response(status)
                self.send_header('Content-Type',mime)
                for key,value in (headers or {}).items():
                    if key.lower() in ('x-frame-id','x-stream-id','x-frame-age-ms'):
                        self.send_header(key,value)
                self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store')
                self.end_headers();self.wfile.write(raw)
            except OSError:pass
        def forward(self):
            if not self.allowed():return self.reply(403,b'{"error":"local_origin_required"}')
            parsed=urlsplit(self.path)
            if parsed.scheme or parsed.netloc:return self.reply(400,b'{"error":"relative_path_required"}')
            if self.command=='GET' and parsed.path=='/proxy-health':
                return self.reply(200,json.dumps(dict(
                    transport_contract='isolated-control-v1' if getattr(board,'control_client',None) is not None else 'shared-control-v0',
                    control_channel_open_timeout_s=getattr(board,'channel_open_timeout_s',None),
                    control_response_timeout_s=getattr(board,'control_response_timeout_s',None),
                    automatic_retry=False,control_location='pynq_ps')).encode())
            body=None
            try:
                if self.command=='POST':
                    size=int(self.headers.get('Content-Length','0'))
                    if not 0<size<=18_000_000 or self.headers.get('Transfer-Encoding'):raise ValueError('invalid_body')
                    body=self.rfile.read(size)
                    if len(body)!=size:raise ValueError('truncated_body')
                path=parsed.path
                is_visual=(path in visual_routes or path.startswith('/examples/') or
                           path.startswith('/prompted/') and path!='/prompted/plan')
                if is_visual:
                    request=urllib.request.Request(visual_console+self.path,data=body,
                        headers={'Content-Type':self.headers.get('Content-Type','application/json')})
                    with opener.open(request,timeout=15) as response:
                        status,raw,headers=response.status,response.read(18_000_001),dict(response.headers)
                else:
                    status,raw,headers=board.request(self.path,body,self.headers.get('Content-Type','application/json'))
                    if path=='/prompted/plan' and status==200:
                        raw=json.dumps(preview_display(json.loads(raw)),ensure_ascii=False).encode()
                if len(raw)>18_000_000:raise ValueError('reply_too_large')
                return self.reply(status,raw,headers)
            except (OSError,ValueError,http.client.HTTPException,SSHException,EOFError):
                return self.reply(503,b'{"error":"board_control_or_visual_link_unavailable_no_fallback"}')
        def do_GET(self):self.forward()
        def do_POST(self):self.forward()
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    return server
