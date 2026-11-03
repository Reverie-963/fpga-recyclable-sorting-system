import importlib.util
import json
import io
from pathlib import Path
import threading
import unittest
import urllib.request
import urllib.error
from types import SimpleNamespace

spec=importlib.util.spec_from_file_location('station_bridge',Path(__file__).resolve().parents[1]/'station_vision_bridge.py')
bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)


class BridgeTests(unittest.TestCase):
    def test_refused_board_port_returns_service_unavailable(self):
        def refused(*args,**kwargs):raise bridge.SSHException('Connect failed')
        ssh=SimpleNamespace(is_active=lambda:True,open_channel=refused)
        transport=bridge.BoardTransport(lambda:SimpleNamespace(get_transport=lambda:ssh),'a'*32)
        server=bridge.make_server(0,transport)
        worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with self.assertRaises(urllib.error.HTTPError) as error:
                opener.open('http://127.0.0.1:'+str(server.server_port)+'/health',timeout=2)
            self.assertEqual(error.exception.code,503)
            self.assertEqual(json.load(error.exception)['error'],'board_service_unavailable')
            error.exception.close()
        finally:server.shutdown();worker.join();server.server_close()

    def test_http10_does_not_close_ssh_channel_before_reading_body(self):
        class Channel:
            def __init__(self):self.closed=False;self.file=None
            def settimeout(self,value):self.timeout=value
            def sendall(self,data):pass
            def makefile(self,*args):
                self.file=io.BytesIO(b'HTTP/1.0 200 OK\r\nContent-Length: 11\r\nContent-Type: application/json\r\nX-Baseline-Record: e30=\r\n\r\n{"ok":true}')
                return self.file
            def close(self):
                self.closed=True
                if self.file and not self.file.closed:self.file.seek(0);self.file.truncate()
        channel=Channel();ssh=SimpleNamespace(is_active=lambda:True,open_channel=lambda *a,**k:channel)
        transport=bridge.BoardTransport(lambda:SimpleNamespace(get_transport=lambda:ssh),'a'*32)
        status,raw,headers=transport.request('/health')
        self.assertEqual(status,200);self.assertTrue(json.loads(raw)['ok']);self.assertTrue(channel.closed)
        self.assertEqual(headers['X-Baseline-Record'],'e30=')

    def test_only_camera_evidence_and_baseline_routes_can_reach_board(self):
        class Fake:
            def __init__(self):self.calls=[]
            def request(self,path,body=None):
                self.calls.append((path,body));return 200,b'{"physical_motion":false}',{'Content-Type':'application/json'}
        fake=Fake();server=bridge.make_server(0,fake)
        worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
        base='http://127.0.0.1:'+str(server.server_port);opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(base+'/vision',timeout=2) as reply:self.assertFalse(json.load(reply)['physical_motion'])
            with opener.open(urllib.request.Request(base+'/baseline',data=b'{}'),timeout=2) as reply:
                self.assertEqual(reply.status,200)
            with opener.open(urllib.request.Request(base+'/regions',data=b'{}'),timeout=2) as reply:
                self.assertEqual(reply.status,200)
            with opener.open(urllib.request.Request(base+'/baseline-snapshot.png',data=b'{}'),timeout=2) as reply:
                self.assertEqual(reply.status,200)
            count=len(fake.calls)
            for path in ('/execute','/serial','/grasp','/health'):
                with self.assertRaises(urllib.error.HTTPError) as e:
                    opener.open(urllib.request.Request(base+path,data=b'{}'),timeout=2)
                self.assertEqual(e.exception.code,404);e.exception.close()
            with self.assertRaises(urllib.error.HTTPError) as e:
                opener.open(urllib.request.Request(base+'/vision',headers={'Origin':'https://example.com'}),timeout=2)
            self.assertEqual(e.exception.code,403);e.exception.close()
            self.assertEqual(len(fake.calls),count)
        finally:server.shutdown();worker.join();server.server_close()


if __name__=='__main__':unittest.main()
