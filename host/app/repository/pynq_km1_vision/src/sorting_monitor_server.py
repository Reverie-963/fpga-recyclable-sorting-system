"""Local station dashboard with an explicit supervised single-attempt entry."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
from urllib.parse import urlsplit, parse_qs
import urllib.request
import urllib.error

from .monitor_control import ControlTelemetry, ROOT
from .monitor_tasks import TaskTelemetry,current_workflow
from .monitor_grasp import display_candidates
from .monitor_fpga_benchmark import fpga_benchmark_summary
from .monitor_station_vision import station_vision_status,bind_station_baseline,saved_acceleration_summary
from .monitor_operation import default_operations,OPERATION_ROUTES,scoped_request


def make_server(port=8779, upstream='http://127.0.0.1:8778', telemetry=None, operations=None,
                resident_voice=False, voice_port='auto', voice_factory=None, voice_direct=False):
    if voice_direct and not resident_voice:
        raise ValueError('direct_voice_requires_resident_listener')
    address = urlsplit(upstream)
    if (address.scheme != 'http' or address.hostname not in ('127.0.0.1','localhost') or
        not address.port or address.path not in ('','/') or address.username or address.password or
        address.query or address.fragment):
        raise ValueError('Use the existing local recognition service')
    upstream = upstream.rstrip('/')
    reader = telemetry if telemetry is not None else TaskTelemetry()
    operation = operations if operations is not None else default_operations(ROOT)
    voice=None
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    server = None

    def request(route, body=None):
        call = urllib.request.Request(upstream+route, data=body,
            headers={'Content-Type':'application/json'} if body is not None else {})
        try:
            scale=getattr(getattr(operation,'backend',None),'wait_time_scale',1)
            reply = opener.open(call, timeout=(25 if body is not None else 3)*scale)
        except urllib.error.HTTPError as exc:
            reply = exc
        with reply:
            payload = reply.read(15_000_001)
            if len(payload)>15_000_000:
                raise ValueError('Oversized recognizer reply')
            return reply.status, payload, {k:v for k,v in reply.headers.items()
                if k.lower() in ('content-type','x-frame-id','x-frame-age-ms')}

    class Handler(BaseHTTPRequestHandler):
        timeout = 20*getattr(getattr(operation,'backend',None),'wait_time_scale',1)

        def log_message(self,*_):
            pass

        def allowed(self):
            hosts = {'127.0.0.1:'+str(server.server_port),'localhost:'+str(server.server_port)}
            return self.headers.get('Host') in hosts and (not self.headers.get('Origin') or
                self.headers['Origin'] in {'http://'+h for h in hosts})

        def reply(self,status,payload,headers=None):
            headers = headers or {'Content-Type':'application/json; charset=utf-8'}
            if not isinstance(payload,bytes):
                payload=json.dumps(payload,ensure_ascii=False,allow_nan=False).encode('utf-8')
            try:
                self.send_response(status)
                for k,v in headers.items():self.send_header(k,v)
                self.send_header('Content-Length',str(len(payload)))
                self.send_header('Cache-Control','no-store')
                self.send_header('X-Content-Type-Options','nosniff')
                self.end_headers()
                self.wfile.write(payload)
            except (ConnectionError,TimeoutError):pass

        def do_GET(self):
            if not self.allowed():return self.reply(403,dict(error='local_origin_required'))
            route=urlsplit(self.path).path
            if route=='/':
                page=Path(__file__).with_name('monitor_assets').joinpath('dashboard.html').read_bytes()
                return self.reply(200,page,{'Content-Type':'text/html; charset=utf-8'})
            if route in ('/monitor.css','/monitor.js'):
                asset=Path(__file__).with_name('monitor_assets').joinpath(route[1:]).read_bytes()
                kind='text/css' if route.endswith('.css') else 'application/javascript'
                return self.reply(200,asset,{'Content-Type':kind+'; charset=utf-8'})
            if route=='/fpga-benchmark':
                return self.reply(200,fpga_benchmark_summary())
            if route=='/station-vision':
                return self.reply(200,station_vision_status(observe_runtime_regions=True))
            if route=='/station-benchmark':
                return self.reply(200,saved_acceleration_summary())
            if route=='/workflow':
                return self.reply(200,current_workflow())
            if route in OPERATION_ROUTES:
                data=operation.status()
                if voice is not None:
                    data['voice_service'].update(direct_commands=voice_direct,
                                                 page_confirmation_required=not voice_direct)
                if OPERATION_ROUTES[route] is not None:data['interface_material']=OPERATION_ROUTES[route]
                return self.reply(200,data)
            if route in ('/control-status','/grasp-record'):
                try:
                    status,raw,_=request('/health')
                    health=json.loads(raw) if status==200 else {}
                except (OSError,ValueError):health={}
                task=parse_qs(urlsplit(self.path).query).get('task_id',[None])[0]
                if task is not None and len(task)>200:return self.reply(400,dict(error='invalid_task_id'))
                try:
                    data=reader.grasp_frame(health,task_id=task) if route=='/grasp-record' else reader.status(health,task_id=task)
                    return self.reply(200,data)
                except (OSError,ValueError,KeyError,TypeError):return self.reply(200,dict(available=False,reason='record_unavailable'))
            if route not in ('/health','/camera.jpg','/live-result','/benchmark','/examples','/overlay.jpg','/latest-result') and not route.startswith('/examples/'):
                return self.reply(404,dict(error='not_found'))
            try:
                status,raw,headers=request(self.path)
                if route=='/live-result' and status==200:
                    raw=display_candidates(json.loads(raw))
                return self.reply(status,raw,headers)
            except (OSError,ValueError):return self.reply(502,dict(error='recognition_service_unavailable'))

        def do_POST(self):
            if not self.allowed():return self.reply(403,dict(error='local_origin_required'))
            if self.path in OPERATION_ROUTES:
                try:
                    length=int(self.headers.get('Content-Length','0'))
                    if (not 0<length<=524288 or self.headers.get('Transfer-Encoding') or
                            self.headers.get('Content-Type','').split(';')[0]!='application/json'):
                        raise ValueError('invalid_operation_body')
                    body=self.rfile.read(length)
                    if len(body)!=length:raise ValueError('incomplete_body')
                    return self.reply(200,operation.command(scoped_request(self.path,json.loads(body))))
                except (ValueError,TypeError,KeyError) as exc:
                    code=str(exc)
                    safe=code if code.startswith('reset_') or code in ('state_reset_not_supported','operation_busy_or_closed') else 'operation_rejected'
                    return self.reply(409,dict(error=safe,operation=operation.status()))
                except OSError:
                    return self.reply(503,dict(error='operation_unavailable'))
            if self.path=='/station-baseline':
                try:
                    length=int(self.headers.get('Content-Length','0'))
                    if not 0<length<=2048 or self.headers.get('Transfer-Encoding'):raise ValueError('invalid_body')
                    body=self.rfile.read(length)
                    if len(body)!=length:raise ValueError('incomplete_body')
                    return self.reply(200,bind_station_baseline(json.loads(body)))
                except (OSError,ValueError,KeyError,TypeError):return self.reply(409,dict(error='baseline_rejected'))
            if self.path not in ('/live','/detect'):
                return self.reply(404,dict(error='read_only_control_dashboard'))
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=8_500_000 or self.headers.get('Transfer-Encoding'):
                    raise ValueError('Invalid request body')
                body=self.rfile.read(length)
                if len(body)!=length:raise ValueError('Incomplete request')
                status,raw,headers=request(self.path,body)
                if self.path=='/detect' and status==200:raw=display_candidates(json.loads(raw))
                return self.reply(status,raw,headers)
            except ValueError:return self.reply(400,dict(error='invalid_request'))
            except OSError:return self.reply(502,dict(error='recognition_service_unavailable'))

    class Server(ThreadingHTTPServer):
        allow_reuse_address=False
        def server_close(self):
            if voice is not None:voice.close()
            operation.close()
            super().server_close()
        def server_bind(self):
            if hasattr(socket,'SO_EXCLUSIVEADDRUSE'):
                self.socket.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
            super().server_bind()
    try:
        server=Server(('127.0.0.1',port),Handler)
        if resident_voice:
            from .km1_voice_resident import ResidentVoice
            factory=voice_factory or ResidentVoice
            folder=getattr(operation.backend,'journal_dir',ROOT/'local-only/sorting-known-ledger')
            options=dict(port=voice_port)
            if voice_direct:options['direct_commands']=True
            voice=factory(operation,folder,ROOT/'local-only/sorting-voice-resident/events.jsonl',**options)
            voice.start()
    except BaseException:
        if voice is not None:voice.close()
        if server is not None:server.server_close()
        operation.close()
        raise
    return server


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8779)
    parser.add_argument('--upstream',default='http://127.0.0.1:8778')
    parser.add_argument('--voice-resident',action='store_true',help='Keep ASRPRO connected; each physical batch needs a fresh page grant')
    parser.add_argument('--voice-port',default='auto')
    parser.add_argument('--voice-direct',action='store_true',help='Dispatch fresh spoken sorting commands without a page grant; retain runtime station checks')
    args=parser.parse_args()
    if args.voice_direct and not args.voice_resident:parser.error('--voice-direct requires --voice-resident')
    server=make_server(args.port,args.upstream,resident_voice=args.voice_resident,
                       voice_port=args.voice_port,voice_direct=args.voice_direct)
    print('Sorting dashboard ready: http://127.0.0.1:'+str(server.server_port),flush=True)
    try:server.serve_forever()
    finally:server.server_close()


if __name__=='__main__':main()
