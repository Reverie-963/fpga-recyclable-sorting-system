"""Browser preconnections and slow uploads must not block the local page."""
import json
from pathlib import Path
import socket
import tempfile
import threading
import unittest
import urllib.request

from src.taco_preview_server import make_server


class FakeDetector:
    labels = ['BG', 'Bottle']
    model_id = 'test-model'
    cfg = {'score_ratio_threshold': 10, 'sorting_mapping': {'Bottle': 'bottle'},
           'sorting_classes': {'bottle': '瓶子类', 'can': '罐子类', 'other': '其他类'}}


class PreviewConnectionTests(unittest.TestCase):
    def test_idle_connections_and_partial_upload_do_not_block_page(self):
        with tempfile.TemporaryDirectory() as folder:
            server = make_server(FakeDetector(), 0, Path(folder))
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            address = server.server_address
            idle = [socket.create_connection(address, timeout=2) for _ in range(8)]
            slow = socket.create_connection(address, timeout=2)
            try:
                slow.sendall(('POST /detect HTTP/1.1\r\nHost: 127.0.0.1:{}\r\n'
                              'Content-Length: 100\r\n\r\n{{').format(address[1]).encode())
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                base = 'http://127.0.0.1:{}'.format(address[1])
                with opener.open(base + '/health', timeout=2) as reply:
                    self.assertTrue(json.load(reply)['model_loaded'])
                with opener.open(base + '/', timeout=2) as reply:
                    self.assertIn('TACO 实时三类识别', reply.read().decode('utf-8'))
            finally:
                for connection in idle + [slow]:
                    connection.close()
                server.shutdown()
                server.server_close()
                worker.join(timeout=3)


if __name__ == '__main__':
    unittest.main()
