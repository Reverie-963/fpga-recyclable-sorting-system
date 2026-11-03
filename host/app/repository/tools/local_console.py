"""Lightweight loopback workbench using the existing station API; no cloud traffic."""
import argparse
import base64
import json
import math
from pathlib import Path
import re
import secrets
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs

from cloud_console_bridge import Bridge, clean
from item_profile_store import ItemProfileStore


ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / 'pynq_km1_vision/cloud_console/ui'
OPERATIONS = {'preview', 'start', 'arm', 'voice_arm', 'cancel', 'stop', 'reset', 'home', 'heartbeat', 'register_heights', 'clear_heights','review'}
FIELDS = {'action', 'material', 'ready', 'reset_confirmed', 'home_confirmed', 'object_height_mm', 'height_plan',
          'owner', 'request_id', 'operation_id', 'expected_operation_id', 'height_registration_id','review_confirmed','review_epoch','review_request_id'}


class ConsoleDependencyUnavailable(RuntimeError):
    """Known local failure before the command reaches the station API."""


def validate_command(body):
    if not isinstance(body, dict) or body.get('kind') not in {'operation', 'select_model', 'detect', 'task'}:
        raise ValueError('不支持的操作')
    payload = body.get('payload')
    if not isinstance(payload, dict):
        raise ValueError('操作内容无效')
    kind = body['kind']
    if kind == 'operation':
        if set(payload) - FIELDS or payload.get('action') not in OPERATIONS:
            raise ValueError('不支持的动作')
        if not re.fullmatch(r'cloud-[a-f0-9-]{36}', str(payload.get('owner', ''))):
            raise ValueError('控制会话无效')
        if payload.get('material', 'AUTO') not in {'AUTO', 'PAPER', 'PLASTIC', 'METAL'}:
            raise ValueError('分拣类别无效')
        height = payload.get('object_height_mm')
        if payload.get('height_plan') is not None:
            if str(ROOT / 'pynq_km1_vision') not in sys.path:
                sys.path.insert(0, str(ROOT / 'pynq_km1_vision'))
            try:
                from src.sorting_item_heights import height_plan
            except ImportError as error:
                raise ConsoleDependencyUnavailable('中控运行环境不完整，请重新启动中控；本次操作未提交。') from error
            height_plan(payload['height_plan'])
            if height is not None:raise ValueError('不能同时指定逐件高度和共用高度')
        if height is not None and (isinstance(height, bool) or not isinstance(height, (float, int)) or not math.isfinite(height) or not 1 <= height <= 120):
            raise ValueError('高度应为 1–120 mm')
        for key in ('ready', 'reset_confirmed', 'home_confirmed','review_confirmed'):
            if key in payload and not isinstance(payload[key], bool):
                raise ValueError('现场确认无效')
        for key in ('request_id', 'operation_id', 'expected_operation_id', 'height_registration_id','review_epoch','review_request_id'):
            if payload.get(key) is not None and (not isinstance(payload[key], str) or not re.fullmatch(r'[a-zA-Z0-9_.-]{1,128}', payload[key])):
                raise ValueError('任务标识无效')
        if payload['action'] in {'start', 'arm', 'voice_arm', 'home', 'reset', 'register_heights'} and payload.get('ready') is not True:
            raise ValueError('请先核对并确认现场')
        if payload['action'] in {'start', 'arm', 'voice_arm', 'preview', 'register_heights'} and not payload.get('request_id'):
            raise ValueError('缺少本次操作标识')
        if payload['action']=='register_heights' and payload.get('height_plan') is None:
            raise ValueError('请先登记并填写每件物品的高度')
        if payload['action']=='review' and (payload.get('review_confirmed') is not True or payload.get('ready') is not True or
            not all(payload.get(k) for k in ('request_id','review_epoch','review_request_id'))):raise ValueError('请先核对现场，再确认已复核')
    elif kind == 'select_model':
        if set(payload) != {'model_id', 'revision'} or not isinstance(payload['model_id'], str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', payload['model_id']) or type(payload['revision']) is not int:
            raise ValueError('模型选择无效')
    elif kind == 'task':
        if set(payload) != {'task_id'} or not isinstance(payload['task_id'], str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,128}', payload['task_id']):
            raise ValueError('任务标识无效')
    elif set(payload) != {'image_base64'} or not isinstance(payload['image_base64'], str) or not 1 <= len(payload['image_base64']) <= 2100000 or not re.fullmatch(r'[A-Za-z0-9+/]*={0,2}', payload['image_base64']):
        raise ValueError('图片无效或超过 1.5 MB')
    return body


def create_server(port=8842, local_url='http://127.0.0.1:8810', bridge=None, profile_path=None):
    bridge = bridge or Bridge({'site_url': 'http://127.0.0.1', 'local_url': local_url})
    token = secrets.token_hex(32)
    normal = threading.Lock()
    profiles=ItemProfileStore(profile_path or ROOT/'local-only/item-profiles/profiles.json')

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, code, value, mime='application/json'):
            raw = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode()
            self.send_response(code)
            for key, val in {'Content-Type': mime, 'Content-Length': str(len(raw)), 'Cache-Control': 'no-store',
                             'X-Content-Type-Options': 'nosniff', 'X-Frame-Options': 'DENY',
                             'Referrer-Policy': 'same-origin',
                             'Content-Security-Policy': "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"}.items():
                self.send_header(key, val)
            self.end_headers()
            try:
                self.wfile.write(raw)
            except OSError:
                pass

        def valid_host(self):
            return self.headers.get('Host') in {'127.0.0.1:' + str(self.server.server_port), 'localhost:' + str(self.server.server_port)}

        def authorized(self):
            return (self.valid_host() and self.headers.get('X-Console-Session') == token and
                    (not self.headers.get('Origin') or self.headers['Origin'] == 'http://' + self.headers['Host']))

        def do_GET(self):
            if not self.valid_host():
                return self.reply(403, {'error': '需要本机访问'})
            url = urlsplit(self.path)
            if url.path == '/':
                html = (UI / 'index.html').read_text(encoding='utf8')
                script = 'window.__LOCAL_CONSOLE__=' + json.dumps({'token': token, 'station_url': bridge.local}) + ';\n' + (UI / 'station_reset.js').read_text(encoding='utf8') + '\n' + (UI / 'app.js').read_text(encoding='utf8')
                script='\n'.join((UI/name).read_text(encoding='utf8') for name in ('item_profiles.js','item_heights.js','item_profile_editor.js'))+'\n'+script
                html = html.replace('/*APP_STYLE*/', (UI / 'style.css').read_text(encoding='utf8')).replace('/*APP_SCRIPT*/', script)
                return self.reply(200, html.encode(), 'text/html; charset=utf-8')
            if url.path == '/favicon.svg':
                return self.reply(200, b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48"><rect width="48" height="48" rx="14" fill="#f1eaff"/><path d="M24 8l5 11 11 5-11 5-5 11-5-11-11-5 11-5z" fill="#8055ce"/></svg>', 'image/svg+xml')
            if url.path == '/api/frame':
                if self.headers.get('Sec-Fetch-Site') == 'cross-site':
                    return self.reply(403, {'error': '需要同源访问'})
                snap = bridge.current_snapshot()
                kind = parse_qs(url.query).get('kind', ['overlay'])[0]
                key = parse_qs(url.query).get('key', [''])[0]
                frame = bridge.display_frame(key) if kind != 'camera' and key else snap.get('camera' if kind == 'camera' else 'overlay')
                if not frame or frame.get('age_ms', 999999) > 7000:
                    return self.reply(503, {'error': '等待新鲜画面'})
                return self.reply(200, base64.b64decode(frame['base64']), 'image/jpeg')
            if not self.authorized():
                return self.reply(403, {'error': '请刷新本机控制台'})
            if url.path == '/api/operation':
                try:
                    return self.reply(200, clean(bridge.local_request('/operation', timeout=4)))
                except (OSError, ValueError):
                    return self.reply(503, {'error': '控制状态暂不可读，未继续重置'})
            if url.path == '/api/item-profiles':
                try:return self.reply(200,profiles.read())
                except (OSError,ValueError) as error:return self.reply(409,{'error':str(error)[:160]})
            if url.path == '/api/state':
                snapshot = bridge.current_snapshot()
                snapshot.pop('camera', None)
                snapshot.pop('overlay', None)
                age = (time.monotonic() - bridge.snapshot_at) * 1000
                snapshot['received_at'] = int(time.time() * 1000 - age)
                return self.reply(200, {'online': age < 6500 and bool(snapshot.get('models')), 'device_age_ms': age, 'snapshot': snapshot})
            return self.reply(404, {'error': 'not_found'})

        def do_POST(self):
            if not self.authorized() or self.headers.get('Origin') != 'http://' + self.headers['Host']:
                return self.reply(403, {'error': '需要本机同源操作'})
            if self.path not in ('/api/local-command','/api/item-profiles'):
                return self.reply(404, {'error': 'not_found'})
            acquired = False
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 2300000 or self.headers.get('Transfer-Encoding'):
                    raise ValueError('内容无效或过大')
                data=json.loads(self.rfile.read(size))
                if self.path=='/api/item-profiles':return self.reply(200,profiles.command(data))
                body = validate_command(data)
                cid = body.get('id')
                if not isinstance(cid, str) or str(uuid.UUID(cid)) != cid:
                    raise ValueError('请求标识无效')
                if (time.monotonic() - bridge.snapshot_at) * 1000 > 6500:
                    return self.reply(503, {'error': '工位状态过期，未提交操作'})
                urgent = body['kind'] == 'operation' and body['payload']['action'] in {'stop', 'heartbeat', 'cancel'}
                if not urgent:
                    acquired = normal.acquire(blocking=False)
                    if not acquired:
                        return self.reply(409, {'error': '上一项操作尚未确认'})
                result = bridge.execute({'id': cid, 'kind': body['kind'], 'payload': body['payload'], 'remaining_ms': 6500})
                return self.reply(200, result)
            except (ValueError, KeyError, TypeError) as error:
                return self.reply(400, {'error': str(error)[:160]})
            except ConsoleDependencyUnavailable as error:
                return self.reply(503, {'error': str(error), 'submitted': False})
            except OSError:
                return self.reply(503, {'error': '本机文件暂不可写，请稍后重试'})
            finally:
                if acquired:
                    normal.release()

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.bridge = bridge
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8842)
    parser.add_argument('--local-url', default='http://127.0.0.1:8810')
    parser.add_argument('--start', action='store_true')
    args = parser.parse_args()
    if args.start:
        folder = ROOT / 'local-only/local-console'
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / 'stdout.log').open('ab') as out, (folder / 'stderr.log').open('ab') as err:
            python = ROOT / '.venv' / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
            process = subprocess.Popen([str(python) if python.is_file() else sys.executable, '-u', str(Path(__file__).resolve()), '--port', str(args.port), '--local-url', args.local_url], stdin=subprocess.DEVNULL, stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
        print(json.dumps({'launcher_pid': process.pid, 'url': 'http://127.0.0.1:' + str(args.port), 'mechanical_commands_on_start': False}))
        return
    server = create_server(args.port, args.local_url)
    threading.Thread(target=server.bridge.collect, daemon=True).start()
    print('Local workbench: http://127.0.0.1:' + str(server.server_port), flush=True)
    try:
        server.serve_forever()
    finally:
        server.bridge.stop.set()
        server.bridge.pool.shutdown(wait=False, cancel_futures=True)
        server.server_close()


if __name__ == '__main__':
    main()
