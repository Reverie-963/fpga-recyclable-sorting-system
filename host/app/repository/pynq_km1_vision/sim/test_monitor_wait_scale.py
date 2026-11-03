import unittest
from types import SimpleNamespace
from unittest.mock import patch
from src.sorting_monitor_server import make_server

class MonitorWaitScaleTests(unittest.TestCase):
    def test_default_constructed_operation_scales_http_handler_timeout(self):
        operation=SimpleNamespace(backend=SimpleNamespace(wait_time_scale=.5),close=lambda:None)
        with patch('src.sorting_monitor_server.default_operations',return_value=operation):
            server=make_server(port=0)
        try:self.assertEqual(server.RequestHandlerClass.timeout,10)
        finally:server.server_close()

    def test_default_wait_keeps_original_http_timeout(self):
        operation=SimpleNamespace(backend=SimpleNamespace(),close=lambda:None)
        with patch('src.sorting_monitor_server.default_operations',return_value=operation):
            server=make_server(port=0)
        try:self.assertEqual(server.RequestHandlerClass.timeout,20)
        finally:server.server_close()
