import ast
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import unittest
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from board_link_monitor import BoardLinkMonitor
from board_station_links import make_control_proxy


class Board:
    def __init__(self):
        self.failure = None
        self.code = 200
        self.value = dict(compute_location='pynq_ps', board_root='native')
        self.calls = []

    def request(self, route, body=None, *args):
        self.calls.append((route, body))
        if self.failure:
            raise self.failure
        return self.code, json.dumps(self.value).encode(), {}


class MonitorTests(unittest.TestCase):
    def test_timeout_is_reported_and_next_read_only_probe_can_recover(self):
        board = Board()
        now = [10]
        monitor = BoardLinkMonitor(board, 'native', clock=lambda: now[0])
        self.assertTrue(monitor.poll()['available'])
        board.failure = ConnectionError('one SSH open timed out')
        now[0] = 15
        status = monitor.poll()
        self.assertFalse(status['available'])
        self.assertEqual(status['seconds_since_success'], 5)
        self.assertEqual(status['consecutive_failures'], 1)
        board.failure = None
        status = monitor.poll()
        self.assertTrue(status['available'])
        self.assertEqual(status['consecutive_failures'], 0)
        self.assertFalse(status['automatic_reconnect'])
        self.assertEqual(board.calls, [('/board-control', None)] * 3)

    def test_persistent_loss_or_changed_identity_never_claims_recovery_or_sends_stop(self):
        board = Board()
        monitor = BoardLinkMonitor(board, 'native')
        for failure in (ConnectionError(), TimeoutError(), EOFError()):
            board.failure = failure
            self.assertFalse(monitor.poll()['available'])
        self.assertEqual(monitor.failures, 3)
        board.failure = None
        for value in ({'compute_location': 'pynq_ps', 'board_root': 'other'},
                      {'compute_location': 'pc', 'board_root': 'native'}, []):
            board.value = value
            self.assertFalse(monitor.poll()['available'])
        board.code = 503
        self.assertFalse(monitor.poll()['available'])
        self.assertEqual(board.calls, [('/board-control', None)] * 7)

    def test_visual_requests_keep_receiving_new_frames_while_board_probe_times_out(self):
        class Visual(BaseHTTPRequestHandler):
            frames = 0
            def log_message(self, *args): pass
            def do_GET(self):
                Visual.frames += 1
                raw = json.dumps(dict(frame_id=Visual.frames, result_age_ms=25)).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(raw)))
                self.send_header('X-Frame-Id', str(Visual.frames))
                self.send_header('X-Frame-Age-Ms', '25')
                self.end_headers()
                self.wfile.write(raw)
        visual = ThreadingHTTPServer(('127.0.0.1', 0), Visual)
        board = Board()
        board.failure = ConnectionError('SSH timeout')
        proxy = make_control_proxy(0, board, 'http://127.0.0.1:' + str(visual.server_port))
        workers = []
        for server in (visual, proxy):
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            workers.append(thread)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        monitor = BoardLinkMonitor(board, 'native')
        frames = []
        try:
            for _ in range(4):
                self.assertFalse(monitor.poll()['available'])
                for route in ('/live-result', '/camera.jpg', '/health'):
                    with opener.open('http://127.0.0.1:' + str(proxy.server_port) + route, timeout=2) as response:
                        self.assertEqual(response.status, 200)
                        self.assertEqual(response.headers['X-Frame-Age-Ms'], '25')
                        frame = json.load(response)
                        self.assertEqual(response.headers['X-Frame-Id'], str(frame['frame_id']))
                        frames.append(frame['frame_id'])
            self.assertEqual(frames, list(range(1, 13)))
            self.assertEqual(board.calls, [('/board-control', None)] * 4)
        finally:
            for server in (visual, proxy):
                server.shutdown()
                server.server_close()
            for worker in workers:
                worker.join(2)

    def test_coordinator_wires_independent_control_and_nonfatal_monitor(self):
        source = (Path(__file__).parents[1] / 'serve_board_station.py').read_text(encoding='utf8')
        tree = ast.parse(source)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        pinned = next(node for node in calls if isinstance(node.func, ast.Name) and node.func.id == 'PinnedBoardHTTP')
        self.assertIn('control_client', [keyword.arg for keyword in pinned.keywords])
        self.assertIn("record['board_link_health']=monitor.poll()", source)
        self.assertNotIn("raise RuntimeError('native_board_unavailable')", source)


if __name__ == '__main__':
    unittest.main()
