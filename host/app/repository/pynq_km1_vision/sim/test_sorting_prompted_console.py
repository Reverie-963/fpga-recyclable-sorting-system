import json
import threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import unittest
from unittest.mock import patch
import urllib.request
import urllib.error
from src.sorting_prompted_console import create_console


class ConsoleTests(unittest.TestCase):
    def setUp(self):
        class Upstream(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_GET(self):
                if self.path=='/':
                    data=b"<script>fetch(path,{});fetch('/camera.jpg');const x=record.image_url;</script>"
                elif self.path=='/health':
                    data=json.dumps(dict(settings=dict(settings_sha256='settings',display_confidence=.25),
                        model_hashes={'yoloe-11m-seg.pt':'weights'},enabled=False)).encode()
                else:data=json.dumps(dict(instances=[dict(label='Drink can',official_label='Drink can',
                    material='metal',visible=True,confidence=.9)],model_sha256='weights',settings_sha256='settings',
                    generated_at=1,image_size=[640,480],overlay_base64='image')).encode()
                self.send_response(200);self.send_header('Content-Type','text/html' if self.path=='/' else 'application/json');self.end_headers();self.wfile.write(data)
            def do_POST(self):
                self.rfile.read(int(self.headers.get('Content-Length','0')))
                self.do_GET()
        self.upstream=ThreadingHTTPServer(('127.0.0.1',0),Upstream)
        threading.Thread(target=self.upstream.serve_forever,daemon=True).start()
        class Operations:
            def status(self):return dict(voice_service={},configured=True,blockers=[])
            def close(self):pass
            def command(self,*args):raise AssertionError('old-model commands must not be dispatched')
        with patch('src.sorting_prompted_console.default_operations',return_value=Operations()):
            self.server=create_console(0,'http://127.0.0.1:'+str(self.upstream.server_port))
        threading.Thread(target=self.server.serve_forever,daemon=True).start()
        self.base='http://127.0.0.1:'+str(self.server.server_port)

    def tearDown(self):
        for server in (self.server,self.upstream):server.shutdown();server.server_close()

    def request(self,path,payload=None):
        return urllib.request.urlopen(urllib.request.Request(self.base+path,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={'Content-Type':'application/json'}),timeout=4)

    def test_console_and_iframe_same_origin_routes(self):
        with self.request('/') as r:page=r.read().decode()
        self.assertIn("frame.src='/prompted/'",page)
        self.assertIn('id="operation-height"',page)
        self.assertIn('物体顶部相对桌面',page)
        with self.request('/monitor.js') as r:code=r.read().decode()
        self.assertIn("object_height_mm:Number($('operation-height').value)",code)
        self.assertIn('heightInput.dataset.material',code)
        with self.request('/prompted/') as r:page=r.read().decode()
        self.assertIn("fetch('/prompted'+path",page)
        self.assertIn("fetch('/prompted/camera.jpg')",page)
        self.assertIn("('/prompted'+record.image_url)",page)
        with self.assertRaises(urllib.error.HTTPError):self.request('/prompted/unknown')

    def test_health_and_original_labels_projected_to_console_materials(self):
        with self.request('/health') as r:h=json.load(r)
        self.assertEqual(h['runtime_id'],'settings');self.assertFalse(h['live_enabled'])
        with self.request('/detect',dict(image_base64='synthetic')) as r:obs=json.load(r)
        self.assertEqual(obs['instances'][0]['label'],'metal')
        self.assertEqual(obs['instances'][0]['official_label'],'Drink can')
        self.assertEqual(obs['model_version'],'weights')

    def test_old_motion_blocked_and_origin_checked(self):
        with self.request('/operation') as r:d=json.load(r)
        self.assertFalse(d['configured']);self.assertIn('prompted_model_preview_only',d['blockers'])
        with self.assertRaises(urllib.error.HTTPError) as e:self.request('/operation',dict(action='start'))
        self.assertEqual(e.exception.code,409)
        request=urllib.request.Request(self.base+'/prompted/health',headers={'Origin':'http://untrusted.example'})
        with self.assertRaises(urllib.error.HTTPError) as e:urllib.request.urlopen(request)
        self.assertEqual(e.exception.code,403)


if __name__=='__main__':unittest.main()
