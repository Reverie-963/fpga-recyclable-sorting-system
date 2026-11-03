"""Opt-in detector relay for a future SSH-forwarded board controller.

Only health and exact-PNG detection are supported. No cameras, station state,
actuators or models are owned by this module. Existing execution paths do not
import or activate it. The HTTP endpoints must remain on loopback; remote use
requires an explicitly established SSH tunnel, not a public listener.
"""
import base64
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import threading
import time
from urllib.parse import urlsplit
import urllib.request
import uuid

from .sorting_io import HTTPYoloe

CONTRACT = 'board-detector-relay-v1'
LIMIT = 15_000_000


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def local_url(url):
    address = urlsplit(url)
    if (address.scheme != 'http' or address.hostname not in ('127.0.0.1', 'localhost') or
            not address.port or address.path not in ('', '/') or
            any((address.username, address.password, address.query, address.fragment))):
        raise ValueError('detector_relay_requires_loopback')
    return url.rstrip('/')


def token_value(token):
    if not isinstance(token, str) or len(token) < 32 or not token.isascii() or not token.isalnum():
        raise ValueError('invalid_detector_relay_token')
    return token


def validate_request(value):
    if (not isinstance(value, dict) or set(value) != {'contract', 'request_id', 'route', 'payload'} or
            value['contract'] != CONTRACT or not isinstance(value['request_id'], str) or
            len(value['request_id']) != 32 or any(c not in '0123456789abcdef' for c in value['request_id'])):
        raise ValueError('invalid_detector_exchange')
    route, payload = value['route'], value['payload']
    if route == 'health' and payload is None:
        return
    if route != 'detect' or not isinstance(payload, dict) or set(payload) != {'image_base64', 'prompt_mode'}:
        raise ValueError('detector_route_not_allowed')
    if payload['prompt_mode'] not in ('supervised', 'text', 'visual'):
        raise ValueError('invalid_detector_prompt_mode')
    raw = base64.b64decode(payload['image_base64'], validate=True)
    if not 8 <= len(raw) <= 8_500_000 or not raw.startswith(b'\x89PNG\r\n\x1a\n'):
        raise ValueError('detector_requires_bounded_png')


class RelayTransport:
    def __init__(self, url, token, timeout, clock=time.monotonic):
        self.url, self.token = local_url(url), token_value(token)
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 30:
            raise ValueError('invalid_detector_relay_timeout')
        self.timeout, self.clock = timeout, clock
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(self, route, payload=None):
        request = dict(contract=CONTRACT, request_id=uuid.uuid4().hex, route=route, payload=payload)
        validate_request(request)
        body = encoded(request)
        if len(body) > LIMIT:
            raise ValueError('oversized_detector_request')
        started = self.clock()
        call = urllib.request.Request(self.url+'/exchange', data=body, headers={
            'Authorization': 'Bearer '+self.token, 'Content-Type': 'application/json'})
        # Single attempt. A broken/late exchange cannot become a fresh result.
        with self.opener.open(call, timeout=self.timeout) as reply:
            raw = reply.read(LIMIT+1)
        if self.clock()-started > self.timeout:
            raise TimeoutError('detector_relay_deadline_exceeded')
        if len(raw) > LIMIT:
            raise ValueError('oversized_detector_response')
        result = json.loads(raw)
        if (not isinstance(result, dict) or result.get('contract') != CONTRACT or result.get('request_id') != request['request_id'] or
                result.get('route') != route or not isinstance(result.get('payload'), dict) or
                result.get('payload_sha256') != hashlib.sha256(encoded(result['payload'])).hexdigest()):
            raise ValueError('detector_exchange_identity_mismatch')
        return result['payload']


class RelayDetector(HTTPYoloe):
    """Preserve deployed model/runtime, image hash, size and prompt guards."""
    def __init__(self, config, token, model_config=None):
        self.relay = RelayTransport(config['detector_url'], token, config['frame_timeout_s'])
        super().__init__(config, model_config)

    def request(self, route, data=None):
        return self.relay.request(route, data)


def make_server(port, token, upstream='http://127.0.0.1:8778', *, upstream_request=None, max_inflight=1, strip_overlay=False, marker_detector=None):
    """Create an opt-in PC relay; the caller owns its lifetime and SSH tunnel.

    Tests inject an upstream. No server is started on import or construction.
    """
    upstream, token = local_url(upstream), token_value(token)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    if type(max_inflight) is not int or not 1 <= max_inflight <= 3:
        raise ValueError('invalid_detector_relay_concurrency')
    available = threading.BoundedSemaphore(max_inflight)

    def forward(route, data):
        if upstream_request is not None:
            return upstream_request(route, data)
        call = urllib.request.Request(upstream+'/'+route,
            data=encoded(data) if data is not None else None,
            headers={'Content-Type': 'application/json'} if data is not None else {})
        with opener.open(call, timeout=25) as response:
            raw = response.read(LIMIT+1)
        if len(raw) > LIMIT:
            raise ValueError('oversized_upstream_response')
        return json.loads(raw)

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.request.settimeout(5)

        def log_message(self, *args):
            pass

        def reply(self, status, value):
            body = encoded(value)
            try:
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)
            except OSError:
                pass

        def do_GET(self):
            self.reply(405, dict(error='post_exchange_only'))

        def do_POST(self):
            expected_hosts = {'127.0.0.1:'+str(server.server_port), 'localhost:'+str(server.server_port)}
            if (self.headers.get('Host') not in expected_hosts or self.headers.get('Origin') or
                    not hmac.compare_digest(self.headers.get('Authorization', '').encode(), ('Bearer '+token).encode())):
                return self.reply(403, dict(error='authenticated_loopback_required'))
            if self.path != '/exchange':
                return self.reply(404, dict(error='not_found'))
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if (not 0 < size <= LIMIT or self.headers.get('Transfer-Encoding') or
                        self.headers.get_content_type() != 'application/json'):
                    raise ValueError('invalid_detector_request_body')
                raw = self.rfile.read(size)
                if len(raw) != size:
                    raise ValueError('truncated_detector_request')
                value = json.loads(raw)
                validate_request(value)
            except (ValueError, TypeError):
                return self.reply(400, dict(error='invalid_detector_exchange'))
            if not available.acquire(blocking=False):
                return self.reply(409, dict(error='detector_relay_busy_no_queue'))
            try:
                payload = forward(value['route'], value['payload'])
                if not isinstance(payload, dict):
                    raise ValueError('invalid_upstream_response')
                if strip_overlay:
                    payload=dict(payload)
                    payload.pop('overlay_base64',None)
                if marker_detector is not None and value['route']=='detect':
                    payload=dict(payload,fixed_marker_observation=marker_detector(value['payload']))
                result = dict(contract=CONTRACT, request_id=value['request_id'], route=value['route'],
                              payload=payload, payload_sha256=hashlib.sha256(encoded(payload)).hexdigest())
                if len(encoded(result)) > LIMIT:
                    raise ValueError('oversized_detector_response')
                self.reply(200, result)
            except (OSError, ValueError, TypeError):
                self.reply(502, dict(error='detector_upstream_failed_no_retry'))
            finally:
                available.release()

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.daemon_threads = True
    return server
