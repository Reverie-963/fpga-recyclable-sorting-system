"""Model selection in front of the existing console; control remains on PYNQ."""
import argparse
import copy
import hashlib
import io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT/'pynq_km1_vision/src/monitor_assets'
LIMIT = 18_000_000
WORKBENCH_STATIC = {'/', '/monitor.js', '/monitor.css', '/board-station.js'}


def local_url(value):
    parsed = urlsplit(value)
    if (parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost') or
            parsed.username or parsed.password or parsed.path not in ('', '/') or parsed.query or parsed.fragment):
        raise ValueError('loopback_service_url_required')
    return value.rstrip('/')


def exchange(base, path, body=None, timeout=8):
    request = urllib.request.Request(base+path, data=body,
        headers={'Content-Type': 'application/json'} if body is not None else {})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        response = opener.open(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        raw = response.read(LIMIT+1)
        if len(raw) > LIMIT:
            raise ValueError('upstream_reply_too_large')
        return response.status, raw, dict(response.headers)


def json_exchange(base, path, payload=None, timeout=8):
    code, raw, _ = exchange(base, path, None if payload is None else json.dumps(payload).encode(), timeout)
    if code != 200:
        raise ValueError('service_not_ready_'+str(code))
    return json.loads(raw)


class ModelSelection:
    def __init__(self, config, catalog_path, request_json=json_exchange,motion_bridge=None):
        self.config, self.catalog_path, self.request_json = copy.deepcopy(config), Path(catalog_path), request_json
        self.control = local_url(config['control_url'])
        self.models = {row['id']: copy.deepcopy(row) for row in config['models']}
        if len(self.models) != len(config['models']) or config['bound_model'] not in self.models:
            raise ValueError('invalid_model_catalog')
        for row in self.models.values():
            if row['kind'] == 'service':
                row['url'] = local_url(row['url'])
            elif row['kind'] not in ('checkpoint', 'unavailable'):
                raise ValueError('invalid_model_kind')
        self.selected = config['bound_model']
        self.current_url = self.models[self.selected]['url']
        self.lock = threading.RLock()
        self.revision = 0
        self.token = secrets.token_urlsafe(32)
        self.switching = False
        self.target = None
        self.error = None
        self.process = None
        self.activated_service = None
        self.done = threading.Event()
        self.motion_bridge=motion_bridge;self.binding_unknown=False;self.retained_candidates=[]
        if config.get('model_motion_follow') and motion_bridge is None:raise ValueError('execution_model_bridge_required')
        self.worker_pool=None
        if motion_bridge is not None:
            from model_workers import WorkerPool
            self.worker_pool=WorkerPool(catalog_path)

    def motion_matches(self,model=None):
        selected=self.selected if model is None else model
        return (not self.binding_unknown and self.motion_bridge.matches(selected)) if self.motion_bridge else selected==self.config['bound_model']

    def start_initial(self):
        if self.motion_bridge:
            # Recover the display route of an already running, PS-bound worker
            # before considering a load. Restarting the thin proxy must neither
            # fall back to its default model nor rebind the robot.
            try:
                if self.restore_running_binding():return
            except (OSError,ValueError,KeyError):pass
            target=self.config.get('initial_model',self.motion_bridge.binding['profile_id'])
            self.select(dict(model_id=target,revision=self.revision,token=self.token))

    def bound_route_health(self,binding):
        request=urllib.request.Request(self.motion_bridge.relay_url+'/health',headers={
            'Authorization':'Bearer '+self.motion_bridge.relay_token,
            'X-Execution-Profile':binding['profile_id'],'X-Execution-Route':binding['route_id']})
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request,timeout=5) as response:raw=response.read(2_000_001)
        if len(raw)>2_000_000:raise ValueError('execution_route_reply_too_large')
        return json.loads(raw)

    def restore_running_binding(self):
        binding=self.motion_bridge.status();row=self.models[binding['profile_id']]
        if row['kind']!='checkpoint' or self.worker_pool is None:return False
        found=self.worker_pool.find(row)
        if found is None:return False
        process,url=found;p=self.motion_bridge.profiles[row['id']]
        if row['sha256']!=p['model_sha256'] or row.get('configuration_sha256')!=p['upstream_configuration_sha256']:
            raise ValueError('running_worker_profile_changed')
        health=self.request_json(url,'/health');live=self.request_json(url,'/live-result');relay=self.bound_route_health(binding)
        for value,weight_key in ((health,'model_sha256'),(live,'model_version')):
            if (value.get(weight_key)!=row['sha256'] or value.get('configuration_sha256')!=row.get('configuration_sha256') or
                row.get('device') and value.get('execution',{}).get('device')!=row['device']):
                raise ValueError('running_worker_identity_changed')
        generation=health.get('generation',health.get('runtime_id'))
        expected=hashlib.sha256(json.dumps([p['profile_sha256'],binding['route_id'],generation]).encode()).hexdigest()
        if (generation is None or relay.get('generation')!=expected or relay.get('settings')!=p['settings'] or
            relay.get('execution_profile',{}).get('profile_sha256')!=binding['profile_sha256'] or
            not (health.get('ready') or health.get('model_loaded')) or not live.get('source_stream_id') or
            type(live.get('result_age_ms')) not in (int,float) or not 0<=live['result_age_ms']<=5000 or
            self.motion_bridge.status()!=binding):raise ValueError('running_execution_route_changed')
        with self.lock:
            self.selected,self.current_url,self.process=row['id'],url,process
            self.binding_unknown=False;self.error=None
        return True

    def snapshot(self):
        with self.lock:
            return self.selected, self.current_url, self.revision

    def status(self):
        with self.lock:
            bound=self.motion_bridge.binding['profile_id'] if self.motion_bridge and self.motion_bridge.binding else self.config['bound_model']
            models = [dict(id=row['id'], name=row['name'], class_count=len(row.get('classes', [])) or row.get('class_count'),
                available=not row.get('disabled_reason') and row['kind'] != 'unavailable',
                disabled_reason=row.get('disabled_reason'), bound=row['id'] == bound)
                for row in self.models.values()]
            return dict(selected=self.selected, selected_name=self.models[self.selected]['name'],
                bound_model=bound, bound_name=self.models[bound]['name'],
                motion_model_matches=self.motion_matches(), models=models,
                model_motion_follow=self.motion_bridge is not None,execution_binding_unknown=self.binding_unknown,
                execution_binding=self.motion_bridge.binding if self.motion_bridge else None,
                switching=self.switching, target=self.target, error=self.error, revision=self.revision, token=self.token,
                integrated_workbench=bool(self.config.get('integrated_workbench')),
                selected_configuration=dict(confidence=self.models[self.selected].get('confidence'),
                    reference_count=self.models[self.selected].get('reference_count',0),
                    weight_version=self.models[self.selected].get('weight_version')))

    def _idle(self):
        state = self.request_json(self.control, '/operation')
        if state.get('busy') or state.get('state') in ('checking', 'checking_preview', 'waiting_clear', 'standby', 'voice_waiting', 'executing', 'stopping'):
            raise ValueError('请先结束当前预演、分拣或待命，再切换模型。')

    def select(self, payload):
        if not isinstance(payload, dict) or set(payload) != {'model_id', 'revision', 'token'}:
            raise ValueError('invalid_model_selection')
        with self.lock:
            if not secrets.compare_digest(str(payload['token']), self.token):
                raise ValueError('页面已过期，请刷新。')
            if type(payload['revision']) is not int or payload['revision'] != self.revision:
                raise ValueError('模型已被其他页面切换，请刷新后重试。')
            row = self.models.get(payload['model_id'])
            if row is None or row['kind'] == 'unavailable' or row.get('disabled_reason'):
                raise ValueError('该模型尚未准备完成，不能切换。')
            if self.switching:
                raise ValueError('模型正在切换，请稍候。')
            if row['id'] == self.selected and self.motion_matches():
                return self.status()
            self._idle()
            self.switching, self.target, self.error = True, row['id'], None
            threading.Thread(target=self._load, args=(copy.deepcopy(row),), daemon=True).start()
            return self.status()

    def stop_process(self, process):
        if process is None or process.poll() is not None:
            return
        # The owned venv launcher has a child interpreter on Windows. Target
        # its exact live process tree, never any pre-existing model or trainer.
        if os.name == 'nt':
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=6,
                creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            process.wait(3)
        except subprocess.TimeoutExpired:
            process.kill()

    def _load(self, row):
        candidate = None
        activated = None
        preserve_candidate=False
        try:
            for name,expected in self.config.get('worker_sources_sha256',{}).items():
                path=(ROOT/name).resolve()
                if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                    raise ValueError('模型加载实现已变化，原选择保留。')
            if row['id'] == self.config['bound_model']:
                url = row['url']
            elif row['kind'] == 'service':
                url = row['url']
            else:
                weights = Path(row['weights'])
                if not weights.is_file() or hashlib.sha256(weights.read_bytes()).hexdigest() != row['sha256']:
                    raise ValueError('候选权重不存在或哈希不匹配。')
                old_port = urlsplit(self.current_url).port
                from model_workers import free_worker_port
                existing=self.worker_pool.find(row) if self.worker_pool else None
                port = free_worker_port(self.config['worker_ports'],old_port) if existing is None else int(urlsplit(existing[1]).port)
                url = 'http://127.0.0.1:'+str(port)
                folder = self.catalog_path.parent/'worker-logs'
                folder.mkdir(exist_ok=True)
                if existing is not None:candidate,url=existing
                else:
                    with (folder/(row['id']+'.stdout.log')).open('ab') as stdout, (folder/(row['id']+'.stderr.log')).open('ab') as stderr:
                        candidate = subprocess.Popen([self.config['worker_python'], '-u', '-m', 'src.console_candidate_monitor',
                            '--catalog', str(self.catalog_path), '--model', row['id'], '--port', str(port),
                            '--device', row.get('device', self.config.get('device', 'cuda:0'))],
                            cwd=ROOT/'pynq_km1_vision', stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                            env=dict(os.environ, PYTHONUTF8='1'), start_new_session=os.name != 'nt',
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                    if self.worker_pool:self.worker_pool.remember(candidate,row,url)
            deadline = time.monotonic()+75
            while not self.done.is_set():
                if candidate is not None and candidate.poll() is not None:
                    raise ValueError('模型加载进程退出，请查看本机运行记录。')
                try:
                    health = self.request_json(url, '/health', timeout=3)
                    actual = health.get('model_sha256') or health.get('model_hashes', {}).get('yoloe-11m-seg.pt')
                    if actual != row['sha256']:
                        raise ValueError('运行模型与登记权重不一致。')
                    if row.get('device') and health.get('execution', {}).get('device') != row['device']:
                        raise ValueError('实际计算设备与登记不一致，拒绝静默回退。')
                    if row.get('configuration_sha256') and health.get('configuration_sha256') != row['configuration_sha256']:
                        raise ValueError('运行配置不匹配，原选择保留。')
                    if not (health.get('ready') or health.get('model_loaded')):
                        raise OSError('loading')
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise ValueError('模型加载超时，原选择保留。')
                    self.done.wait(.3)
            if self.done.is_set():
                raise ValueError('console_closed')
            if row.get('protocol') == 'prompted':
                self.request_json(url, '/configure', {'enabled': True})
                if row['id'] != self.config['bound_model'] and not health.get('enabled', False):
                    activated = url
            # Require a real camera result before publishing the new selection.
            while not self.done.is_set():
                try:
                    live = self.request_json(url, '/live-result', timeout=3)
                    actual = live.get('model_version') or live.get('model_sha256')
                    if actual != row['sha256']:
                        raise ValueError('识别结果的模型身份不匹配。')
                    if row.get('device') and live.get('execution', {}).get('device') != row['device']:
                        raise ValueError('识别结果的计算设备不匹配。')
                    if row.get('configuration_sha256') and live.get('configuration_sha256') != row['configuration_sha256']:
                        raise ValueError('识别结果的配置不匹配。')
                    break
                except (OSError, ValueError) as exc:
                    if time.monotonic() >= deadline:
                        raise ValueError('未取得新模型的现场识别结果，原选择保留。') from exc
                    self.done.wait(.3)
            with self.lock:
                if self.done.is_set():
                    raise ValueError('console_closed')
                self._idle()
                for name,expected in self.config.get('worker_sources_sha256',{}).items():
                    if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:raise ValueError('模型加载实现已变化，原选择保留。')
                if self.motion_bridge:self.motion_bridge.bind(row,url)
                old, old_activated = self.process, self.activated_service
                self.process = candidate
                self.activated_service = activated
                self.selected, self.current_url = row['id'], url
                self.binding_unknown=False
                self.revision += 1
            candidate = None
            activated = None
            if old_activated:
                try:
                    self.request_json(old_activated, '/configure', {'enabled': False})
                except (OSError, ValueError):
                    pass
            self.stop_process(old)
            with self.lock:
                self.switching, self.target = False, None
        except Exception as exc:
            unknown=bool(getattr(exc,'preserve_candidate',False))
            preserve_candidate=unknown or bool(self.motion_bridge and self.motion_bridge.matches(row['id']))
            with self.lock:
                self.switching, self.target, self.error = False, None, (str(exc) or type(exc).__name__)[:240]
                if preserve_candidate:
                    if unknown:self.binding_unknown=True
                    self.retained_candidates.append((candidate,activated))
        finally:
            if not preserve_candidate:self.stop_process(candidate)
            if activated and not preserve_candidate:
                try:
                    self.request_json(activated, '/configure', {'enabled': False})
                except (OSError, ValueError):
                    pass

    def close(self):
        self.done.set()
        with self.lock:
            process, self.process = self.process, None
            activated, self.activated_service = self.activated_service, None
            retain=self.motion_bridge is not None and (self.binding_unknown or self.motion_bridge.matches(self.selected))
        if retain:return
        self.stop_process(process)
        if activated:
            try:
                self.request_json(activated, '/configure', {'enabled': False})
            except (OSError, ValueError):
                pass


def make_server(port, selection, *, workbench=None):
    server = workbench
    original = workbench.RequestHandlerClass if workbench is not None else BaseHTTPRequestHandler
    static_cache = getattr(workbench, '_model_static_assets', {})
    visual = {'/health', '/live-result', '/latest-result', '/detect', '/live', '/overlay.jpg'}
    class Handler(original):
        def log_message(self, *args):
            pass

        def allowed(self):
            hosts = {'127.0.0.1:'+str(server.server_port), 'localhost:'+str(server.server_port)}
            return self.headers.get('Host') in hosts and (not self.headers.get('Origin') or self.headers['Origin'] in {'http://'+h for h in hosts})

        def reply(self, code, value, mime='application/json; charset=utf-8'):
            raw = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode()
            if workbench is not None:
                headers = mime if isinstance(mime, dict) else {'Content-Type': mime}
                path = urlsplit(self.path).path
                if path in visual and getattr(self, '_visual_revision', selection.snapshot()[2]) != selection.snapshot()[2]:
                    code, raw, headers = 503, b'{"error":"model_changed_discard_old_response"}', {'Content-Type':'application/json'}
                if self.command == 'GET' and path in WORKBENCH_STATIC:
                    if code == 200:
                        static_cache[path] = (code, raw, headers)
                    elif code >= 500 and path in static_cache:
                        code, raw, headers = static_cache[path]
                if code == 200 and self.command == 'GET':
                    if path == '/':
                        raw = raw.replace(b'</body>', b'<script src="/model-switch.js"></script></body>')
                    elif path == '/operation' or path.startswith('/operation/'):
                        data = json.loads(raw)
                        if selection.motion_bridge:selection.motion_bridge.observe(data.get('execution_model'))
                        if not selection.motion_matches():
                            data['blockers'] = list(data.get('blockers', []))+[
                                'selected_execution_model_not_synchronized' if selection.motion_bridge else 'selected_model_recognition_only']
                        raw = json.dumps(data).encode()
                return super().reply(code, raw, headers)
            try:
                self.send_response(code)
                self.send_header('Content-Type', mime)
                self.send_header('Content-Length', str(len(raw)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(raw)
            except OSError:
                pass

        def delegate(self, body=None):
            if self.command == 'GET':
                return super().do_GET()
            stream = self.rfile
            length = self.headers.get('Content-Length')
            try:
                self.rfile = io.BytesIO(body)
                self.headers.replace_header('Content-Length',str(len(body)))
                return super().do_POST()
            finally:
                self.rfile = stream
                self.headers.replace_header('Content-Length',length)

        def dispatch(self):
            if not self.allowed():
                return self.reply(403, {'error': 'local_origin_required'})
            parsed = urlsplit(self.path)
            if parsed.scheme or parsed.netloc:
                return self.reply(400, {'error': 'relative_path_required'})
            path, body = parsed.path, None
            try:
                if self.command == 'POST':
                    length = int(self.headers.get('Content-Length', '0'))
                    if not 0 < length <= LIMIT or self.headers.get('Transfer-Encoding'):
                        raise ValueError('invalid_body')
                    body = self.rfile.read(length)
                    if len(body) != length:
                        raise ValueError('truncated_body')
                if path == '/model-switch/status' and self.command == 'GET':
                    return self.reply(200, selection.status())
                if path == '/model-switch/select' and self.command == 'POST':
                    return self.reply(202, selection.select(json.loads(body)))
                if path in ('/model-switch.js', '/model-preview.html') and self.command == 'GET':
                    name = 'model_switch.js' if path.endswith('.js') else 'model_preview.html'
                    return self.reply(200, (ASSETS/name).read_bytes(), 'application/javascript' if name.endswith('.js') else 'text/html; charset=utf-8')
                model, url, revision = selection.snapshot()
                if path in visual:
                    self._visual_revision = revision
                row = selection.models[model]
                alternate = model != selection.config['bound_model']
                if path == '/prompted/' and self.command == 'GET' and (workbench is None or alternate):
                    return self.reply(200, (ASSETS/'model_preview.html').read_bytes(), 'text/html; charset=utf-8')
                with selection.lock:
                    if self.command == 'POST' and (path == '/operation' or path.startswith('/operation/') or path == '/prompted/plan'):
                        payload = json.loads(body)
                        if path == '/prompted/plan' or payload.get('action') in ('preview', 'start', 'arm', 'voice_arm'):
                            if not selection.motion_matches() or selection.switching:
                                return self.reply(409, {'error': '模型与执行配置尚未同步完成，请等待切换完成。' if selection.motion_bridge else '当前选择用于识别对比；请切回执行绑定模型后再预演或抓取。'})
                            if selection.motion_bridge:
                                payload['execution_profile_sha256']=selection.motion_bridge.binding['profile_sha256']
                                payload['execution_binding_revision']=selection.motion_bridge.binding['revision']
                                body=json.dumps(payload).encode()
                        # One submission only. Never replay an uncertain control response.
                        if workbench is not None:
                            return self.delegate(body)
                        code, raw, headers = exchange(selection.control, self.path, body, timeout=8)
                        return self.reply(code, raw, headers.get('Content-Type', 'application/json'))
                if alternate and path.startswith('/prompted/') and not selection.motion_bridge:
                    return self.reply(409, {'error': '切换预览不修改执行模型的提示词、校准或参数。'})
                if workbench is not None and (not alternate or path not in visual):
                    return self.delegate(body)
                if path in visual:
                    route, payload = self.path, body
                    if row.get('protocol') == 'prompted':
                        if path == '/live':
                            route = '/configure'
                            payload = json.dumps({'enabled': json.loads(body)['enabled']}).encode()
                        elif path in ('/detect',) and body:
                            payload = json.dumps({k: v for k, v in json.loads(body).items() if k != 'prompt_mode'}).encode()
                    code, raw, headers = exchange(url, route, payload, timeout=15)
                    if code == 200 and row.get('protocol') == 'prompted' and path in ('/health', '/live-result', '/detect'):
                        value = json.loads(raw)
                        if path == '/health':
                            value.update(model_name=row['name'], model_loaded=True, model_sha256=row['sha256'],
                                runtime_id=value['settings']['settings_sha256'], live_enabled=value['enabled'],
                                default_prompt_mode='text', confidence_threshold=value['settings']['display_confidence'])
                        else:
                            value.update(model_version=value['model_sha256'], runtime_id=value['settings_sha256'],
                                frame_id=value.get('source_frame_id', str(value.get('generated_at'))))
                            value['instances'] = [dict(item, label=item['material']) for item in value['instances'] if item.get('visible')]
                        raw = json.dumps(value).encode()
                else:
                    code, raw, headers = exchange(selection.control, self.path, body, timeout=15 if self.command == 'GET' else 8)
                if self.command == 'GET' and path in ('/', '/monitor.js', '/monitor.css'):
                    # Only static interface assets may survive a transient board
                    # display failure. Never cache control state or observations.
                    if code == 200:
                        static_cache[path] = (code, raw, headers)
                    elif code >= 500:
                        if path in static_cache:
                            code, raw, headers = static_cache[path]
                        elif path == '/':
                            code, raw, headers = 200, (ASSETS/'model_console_offline.html').read_bytes(), {'Content-Type': 'text/html; charset=utf-8'}
                if path in visual and selection.snapshot()[2] != revision:
                    return self.reply(503, {'error': 'model_changed_discard_old_response'})
                if (path == '/operation' or path.startswith('/operation/')) and self.command == 'GET' and code == 200:
                    value = json.loads(raw)
                    if selection.motion_bridge:selection.motion_bridge.observe(value.get('execution_model'))
                    if not selection.motion_matches():value['blockers'] = list(value.get('blockers', []))+[
                        'selected_execution_model_not_synchronized' if selection.motion_bridge else 'selected_model_recognition_only']
                    raw = json.dumps(value).encode()
                if path == '/' and self.command == 'GET' and code == 200:
                    raw = raw.replace(b'</body>', b'<script src="/model-switch.js"></script></body>')
                    raw = raw.replace('GPU 识别整段'.encode(), '模型识别整段'.encode())
                    raw = raw.replace('GPU 生成掩码；CPU 提取并绘制轮廓。'.encode(), '模型生成掩码；CPU 提取并绘制轮廓。'.encode())
                if path == '/monitor.js' and code == 200:
                    raw = raw.replace(":'识别设备待核对',gpu?'ok':'warn'".encode(),
                        ":d.execution?.device==='cpu'?'CPU · '+(d.execution.precision||'fp32').toUpperCase():'识别设备待核对',gpu||d.execution?.device==='cpu'?'ok':'warn'".encode())
                return self.reply(code, raw, headers.get('Content-Type', 'application/json'))
            except (ValueError, KeyError, TypeError) as exc:
                return self.reply(409 if path == '/model-switch/select' else 400, {'error': str(exc)[:240]})
            except OSError:
                if self.command == 'GET' and path == '/':
                    return self.reply(200, (ASSETS/'model_console_offline.html').read_bytes().replace(
                        b'</body>', b'<script src="/model-switch.js"></script></body>'), 'text/html; charset=utf-8')
                return self.reply(503, {'error': '服务暂不可用；未重试控制命令，请查看当前状态。'})

        def do_GET(self):
            self.dispatch()

        def do_POST(self):
            self.dispatch()
    if workbench is None:
        server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    else:
        server.RequestHandlerClass = Handler
    return server


def attach_to_workbench(server, board, catalog_path):
    """Keep the native workbench and its dedicated PS control connection."""
    config = json.loads(Path(catalog_path).read_text(encoding='utf-8-sig'))
    config['integrated_workbench'] = True
    control = local_url(config['control_url'])
    def request_json(base, path, payload=None, timeout=8):
        if base == control and path == '/operation' and payload is None:
            code, raw, _ = board.request('/operation')
            if code != 200:
                raise ValueError('board_status_unavailable')
            return json.loads(raw)
        return json_exchange(base, path, payload, timeout)
    bridge=None
    if config.get('model_motion_follow'):
        from model_motion_bridge import ModelMotionBridge
        bridge=ModelMotionBridge(board,json.loads(Path(config['execution_profiles_file']).read_text(encoding='utf-8')),
            local_url(config['execution_relay_url']),Path(config['execution_relay_token_file']).read_text().strip())
    selection = ModelSelection(config, catalog_path, request_json=request_json,motion_bridge=bridge)
    # Keep the exact native UI available when display channels are briefly full.
    # These bounded GETs never cache observations, operation state or POSTs.
    server._model_static_assets = {}
    for path in sorted(WORKBENCH_STATIC):
        for _ in range(3):
            try:
                code, raw, headers = board.request(path)
                if code == 200:
                    server._model_static_assets[path] = (code, raw, headers)
                    break
            except OSError:
                pass
    make_server(server.server_port, selection, workbench=server)
    selection.start_initial()
    return selection


def make_redirect_server(port, destination):
    destination = local_url(destination)+'/'
    class Redirect(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_GET(self):
            self.send_response(302)
            self.send_header('Location', destination)
            self.send_header('Content-Length', '0')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
        def do_POST(self):
            raw = b'{"error":"workbench_moved_reload_main_page"}'
            self.send_response(409)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers();self.wfile.write(raw)
    return ThreadingHTTPServer(('127.0.0.1', port), Redirect)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog', type=Path)
    p.add_argument('--redirect-to')
    p.add_argument('--port', type=int, default=8830)
    args = p.parse_args()
    if args.redirect_to:
        server = make_redirect_server(args.port, args.redirect_to)
        try: server.serve_forever()
        finally: server.server_close()
        return
    if args.catalog is None:
        p.error('--catalog or --redirect-to is required')
    config = json.loads(args.catalog.read_text(encoding='utf-8-sig'))
    selection = ModelSelection(config, args.catalog.resolve())
    server = make_server(args.port, selection)
    if urlsplit(selection.control).port == server.server_port:
        raise ValueError('switcher_cannot_proxy_itself')
    print('Switchable model console: http://127.0.0.1:'+str(server.server_port), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        selection.close()


if __name__ == '__main__':
    main()
