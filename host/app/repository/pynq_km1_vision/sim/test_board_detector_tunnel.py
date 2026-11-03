"""Socketpair SSH-channel doubles: no actual SSH session or hardware required."""
import hashlib
import http.client
import json
import socket
import threading
import unittest

from src.board_detector_transport import CONTRACT, encoded
from src.board_detector_tunnel import DetectorTunnel


class FakeTransport:
    def __init__(self):
        self.active = True
        self.requests = []
        self.cancelled = []
        self.refuse = False
        self.change_port = False

    def is_active(self):
        return self.active

    def request_port_forward(self, host, port, handler):
        self.requests.append((host, port))
        if self.refuse:
            raise OSError('test_forward_refused')
        self.handler = handler
        return port+1 if self.change_port else port

    def cancel_port_forward(self, host, port):
        self.cancelled.append((host, port))


class FakeClient:
    def __init__(self):
        self.transport = FakeTransport()
        self.closes = 0

    def get_transport(self):
        return self.transport

    def close(self):
        self.closes += 1
        self.transport.active = False


class TunnelTests(unittest.TestCase):
    def setUp(self):
        self.client = FakeClient()
        self.created = 0
        self.upstream_calls = []

    def connect(self):
        self.created += 1
        return self.client

    def upstream(self, route, data):
        self.upstream_calls.append((route, data))
        return dict(model_loaded=True, fake_detector=True)

    def tunnel(self, **options):
        return DetectorTunnel(self.connect, 't'*40, upstream_request=self.upstream, **options)

    def forwarded_call(self, tunnel, route='health', token='t'*40):
        board, channel = socket.socketpair()
        board.settimeout(2)
        self.client.transport.handler(channel, ('127.0.0.1', 30000), ('127.0.0.1', tunnel.port))
        connection = http.client.HTTPConnection('127.0.0.1', tunnel.port, timeout=2)
        connection.sock = board
        value = dict(contract=CONTRACT, request_id='a'*32, route=route, payload=None)
        try:
            connection.request('POST', '/exchange', encoded(value), headers={
                'Authorization':'Bearer '+token, 'Content-Type':'application/json'})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()
            board.close()

    def test_forwarded_http_preserves_host_nonce_and_payload(self):
        tunnel = self.tunnel()
        with tunnel:
            status, result = self.forwarded_call(tunnel)
            self.assertEqual(status, 200)
            self.assertEqual(result['request_id'], 'a'*32)
            self.assertEqual(result['payload_sha256'], hashlib.sha256(encoded(result['payload'])).hexdigest())
            self.assertEqual(tunnel.board_url, 'http://127.0.0.1:'+str(tunnel.port))
        self.assertEqual(self.client.transport.requests, [('127.0.0.1', tunnel.port)])
        self.assertEqual(self.client.transport.cancelled, [('127.0.0.1', tunnel.port)])
        self.assertEqual(self.client.closes, 1)
        self.assertEqual(self.created, 1)
        with self.assertRaises(ConnectionError):
            _ = tunnel.board_url

    def test_wrong_token_and_motion_route_still_rejected_after_forwarding(self):
        with self.tunnel() as tunnel:
            self.assertEqual(self.forwarded_call(tunnel, token='w'*40)[0], 403)
            self.assertEqual(self.forwarded_call(tunnel, route='execute')[0], 400)
        self.assertEqual(self.upstream_calls, [])

    def test_ssh_loss_or_transport_replacement_never_reconnects(self):
        for replace in (False, True):
            with self.subTest(replace=replace):
                self.client = FakeClient()
                tunnel = self.tunnel()
                with tunnel:
                    if replace:
                        self.client.transport = FakeTransport()
                    else:
                        self.client.transport.active = False
                    with self.assertRaises(ConnectionError): tunnel.require_connected()
                with self.assertRaises(RuntimeError): tunnel.start()
        self.assertEqual(self.created, 2)

    def test_refused_or_changed_remote_port_leaves_no_local_listener(self):
        for flag in ('refuse', 'change_port'):
            with self.subTest(flag=flag):
                self.client = FakeClient()
                setattr(self.client.transport, flag, True)
                tunnel = self.tunnel()
                with self.assertRaises((OSError, ConnectionError)): tunnel.start()
                self.assertEqual(tunnel.state, 'closed')
                self.assertEqual(self.client.closes, 1)
                probe = socket.socket()
                try:
                    probe.bind(('127.0.0.1', tunnel.port))
                finally:
                    probe.close()

    def test_non_loopback_forwarded_address_rejected(self):
        with self.tunnel() as tunnel:
            board, channel = socket.socketpair()
            try:
                self.client.transport.handler(channel, ('127.0.0.1', 30000), ('0.0.0.0', tunnel.port))
                self.assertEqual(channel.fileno(), -1)
            finally:
                board.close()
                channel.close()
        self.assertEqual(self.upstream_calls, [])

    def test_close_drops_owned_channels_and_is_idempotent(self):
        tunnel = self.tunnel()
        tunnel.start()
        board, channel = socket.socketpair()
        self.client.transport.handler(channel, ('127.0.0.1', 30000), ('127.0.0.1', tunnel.port))
        try:
            tunnel.close()
            tunnel.close()
            self.assertEqual(channel.fileno(), -1)
            self.assertEqual(self.client.closes, 1)
            self.assertFalse(any(worker.is_alive() for worker in tunnel.workers))
        finally:
            board.close()


if __name__ == '__main__':
    unittest.main()
