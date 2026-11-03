"""Exercise selection failure, identity and mechanical-command separation over HTTP."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request
sys.path.insert(0, str(Path(__file__).parents[1]))
from switchable_console import ModelSelection, make_server, json_exchange, local_url, attach_to_workbench, make_redirect_server


class SwitchTests(unittest.TestCase):
    def setUp(self):
        self.calls, self.busy, self.bad_hash = [], False, False
        self.entered, self.release = threading.Event(), threading.Event()
        self.delay_result = False
        outer = self
        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_GET(self):
                candidate = self.server is outer.candidate
                digest = 'wrong' if candidate and outer.bad_hash else 'candidate' if candidate else 'bound'
                if self.path.startswith('/operation'):
                    value = dict(state='executing' if outer.busy else 'stopped', busy=outer.busy,
                                 blockers=['old_failure'], result={'task_id': 'keep-original'})
                elif self.path == '/health':
                    value = dict(model_loaded=True, model_sha256=digest, runtime_id=digest, live_enabled=True)
                elif self.path == '/live-result':
                    if outer.delay_result and not candidate:
                        outer.entered.set(); outer.release.wait(3)
                    value = dict(model_version=digest, runtime_id=digest, result_age_ms=20, instances=[])
                else: value = {}
                raw = json.dumps(value).encode()
                self.send_response(200); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(raw)))
                self.end_headers(); self.wfile.write(raw)
            def do_POST(self):
                outer.calls.append((self.path, json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
                self.do_GET()
        self.control = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
        self.candidate = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
        for server in (self.control, self.candidate):
            threading.Thread(target=server.serve_forever, daemon=True).start()
        self.folder = tempfile.TemporaryDirectory()
        self.catalog = Path(self.folder.name)/'models.json'
        self.config = dict(bound_model='bound', control_url='http://127.0.0.1:'+str(self.control.server_port), models=[
            dict(id='bound', name='Bound', kind='service', url='http://127.0.0.1:'+str(self.control.server_port), sha256='bound'),
            dict(id='candidate', name='Candidate', kind='service', url='http://127.0.0.1:'+str(self.candidate.server_port), sha256='candidate'),
            dict(id='training', name='Training', kind='unavailable', disabled_reason='训练中')])
        self.selection = ModelSelection(self.config, self.catalog)
        self.server = make_server(0, self.selection)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = 'http://127.0.0.1:'+str(self.server.server_port)

    def tearDown(self):
        self.release.set()
        self.selection.close()
        for server in (self.server, self.control, self.candidate):
            server.shutdown(); server.server_close()
        self.folder.cleanup()

    def call(self, path, payload=None, headers=None):
        req = urllib.request.Request(self.base+path, data=None if payload is None else json.dumps(payload).encode(),
            headers=headers or {'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=4) as response:
            return json.load(response)

    def choose(self, name='candidate'):
        state = self.call('/model-switch/status')
        self.call('/model-switch/select', dict(model_id=name, revision=state['revision'], token=state['token']))
        deadline = time.monotonic()+3
        while time.monotonic() < deadline:
            state = self.call('/model-switch/status')
            if not state['switching']: return state
            time.sleep(.01)
        self.fail('selection timed out')

    def test_switch_publishes_verified_model_and_preserves_pending(self):
        state = self.choose()
        self.assertEqual(state['selected'], 'candidate')
        self.assertFalse(state['motion_model_matches'])
        self.assertEqual(self.call('/health')['model_sha256'], 'candidate')
        operation = self.call('/operation/metal')
        self.assertEqual(operation['result']['task_id'], 'keep-original')
        self.assertIn('old_failure', operation['blockers'])
        self.assertIn('selected_model_recognition_only', operation['blockers'])
        self.assertEqual(self.calls, [])

    def test_alternate_blocks_start_and_preview_but_preserves_stop_heartbeat(self):
        self.choose()
        for path, payload in [('/operation/metal', {'action': 'start'}), ('/operation', {'action': 'preview'}), ('/prompted/plan', {})]:
            with self.assertRaises(urllib.error.HTTPError) as exc: self.call(path, payload)
            self.assertEqual(exc.exception.code, 409)
        self.assertEqual(self.calls, [])
        for action in ('stop', 'heartbeat'):
            self.call('/operation', {'action': action})
        self.assertEqual([data['action'] for _, data in self.calls], ['stop', 'heartbeat'])

    def test_busy_cannot_switch_and_unchanged_model(self):
        self.busy = True
        with self.assertRaises(urllib.error.HTTPError): self.choose()
        self.assertEqual(self.selection.snapshot()[0], 'bound')
        self.assertFalse(self.selection.switching)

    def test_bad_weight_identity_retains_previous_choice(self):
        self.bad_hash = True
        state = self.choose()
        self.assertEqual(state['selected'], 'bound')
        self.assertIsNotNone(state['error'])
        self.assertEqual(self.call('/health')['model_sha256'], 'bound')

    def test_same_weight_wrong_configuration_retains_previous_choice(self):
        self.selection.models['candidate']['configuration_sha256'] = 'different-confidence-or-reference-bank'
        state = self.choose()
        self.assertEqual(state['selected'], 'bound')
        self.assertIn('配置不匹配', state['error'])
        self.assertEqual(self.calls, [])

    def test_gpu_label_cannot_accept_cpu_or_missing_device(self):
        self.selection.models['candidate']['device']='cuda:0'
        state=self.choose()
        self.assertEqual(state['selected'],'bound')
        self.assertIn('计算设备',state['error'])

    def test_unknown_training_wrong_token_stale_revision_and_cross_origin(self):
        state = self.call('/model-switch/status')
        payload = dict(model_id='candidate', token=state['token'], revision=state['revision'])
        for update in ({'model_id': 'training'}, {'model_id': 'unknown'}, {'token': 'wrong'}, {'revision': 99}):
            with self.assertRaises(urllib.error.HTTPError): self.call('/model-switch/select', dict(payload, **update))
        with self.assertRaises(urllib.error.HTTPError) as exc:
            self.call('/model-switch/select', payload, {'Origin': 'http://evil.example'})
        self.assertEqual(exc.exception.code, 403)
        self.assertEqual(self.selection.snapshot()[0], 'bound')

    def test_old_inflight_result_is_discarded_after_switch(self):
        self.delay_result = True
        outcome = []
        def pending():
            try: self.call('/live-result')
            except urllib.error.HTTPError as exc: outcome.append(exc.code)
        thread = threading.Thread(target=pending); thread.start()
        self.assertTrue(self.entered.wait(1))
        self.choose()
        self.release.set(); thread.join(3)
        self.assertEqual(outcome, [503])

    def test_return_to_binding_restores_motion_route_without_changing_payload(self):
        self.choose(); self.choose('bound')
        self.call('/operation', {'action': 'preview', 'request_id': 'single'})
        self.assertEqual(self.calls, [('/operation', {'action': 'preview', 'request_id': 'single'})])

    def test_next_switch_waits_for_previous_worker_cleanup(self):
        old = object()
        self.selection.process = old
        cleaning, cleaned = threading.Event(), threading.Event()
        def stop(process):
            if process is old:
                cleaning.set(); cleaned.wait(2)
        self.selection.stop_process = stop
        state = self.call('/model-switch/status')
        self.call('/model-switch/select', dict(model_id='candidate', revision=state['revision'], token=state['token']))
        self.assertTrue(cleaning.wait(1))
        state = self.call('/model-switch/status')
        self.assertTrue(state['switching'])
        with self.assertRaises(urllib.error.HTTPError):
            self.call('/model-switch/select', dict(model_id='bound', revision=state['revision'], token=state['token']))
        cleaned.set()
        deadline = time.monotonic()+1
        while self.selection.switching and time.monotonic()<deadline: time.sleep(.01)
        self.assertFalse(self.selection.switching)

    def test_loopback_urls_only(self):
        for url in ('https://127.0.0.1', 'http://example.com', 'http://127.0.0.1/a', 'http://user:pass@localhost'):
            with self.assertRaises(ValueError): local_url(url)

    def test_board_display_failure_keeps_recognition_interface_without_control_fallback(self):
        with patch('switchable_console.exchange', return_value=(503, b'{}', {})):
            with urllib.request.urlopen(self.base+'/', timeout=2) as response:
                page=response.read().decode()
            self.assertIn('完整控制页面暂不可用', page)
            self.assertIn('/model-switch.js', page)
            self.assertNotIn('operation-start', page)
        self.assertEqual(self.calls, [])

    def integrate_native(self):
        self.server.shutdown(); self.server.server_close()
        outer=self
        self.native_unavailable=False
        class Native(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def reply(self,code,raw,headers=None):
                self.send_response(code)
                for key,value in (headers or {'Content-Type':'application/json'}).items():self.send_header(key,value)
                self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
            def forward(self):
                if self.path=='/live-result' and outer.delay_result:
                    outer.entered.set();outer.release.wait(3)
                if self.command=='POST':
                    outer.calls.append((self.path,json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
                if outer.native_unavailable:
                    return self.reply(503,b'{"error":"native_unavailable"}')
                if self.path=='/':
                    return self.reply(200,b'<html><body><button id="voice-controls">Voice</button><input id="operation-height"><section id="view-tasks"></section><section id="view-performance"></section></body></html>',{'Content-Type':'text/html'})
                if self.path=='/prompted/':
                    return self.reply(200,b'<html>native-rich-thresholds-and-empty-background</html>',{'Content-Type':'text/html'})
                if self.path.startswith('/operation'):
                    data={'state':'stopped','busy':False,'blockers':['original_pending'],'result':{'task_id':'native-original'}}
                else:data={'native_route':self.path}
                return self.reply(200,json.dumps(data).encode(),{'Content-Type':'application/json','X-Frame-Id':'native-frame'})
            def do_GET(self):self.forward()
            def do_POST(self):self.forward()
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Native)
        self.selection.config['integrated_workbench']=True
        make_server(self.server.server_port,self.selection,workbench=self.server)
        threading.Thread(target=self.server.serve_forever,daemon=True).start()
        self.base='http://127.0.0.1:'+str(self.server.server_port)

    def test_integrated_workbench_keeps_views_rich_parameters_and_native_headers(self):
        self.integrate_native()
        with urllib.request.urlopen(self.base+'/',timeout=2) as r:page=r.read().decode()
        for name in ['voice-controls','operation-height','view-tasks','view-performance']:
            self.assertIn(name,page)
        self.assertEqual(page.count('/model-switch.js'),1)
        with urllib.request.urlopen(self.base+'/prompted/',timeout=2) as r:
            self.assertIn(b'native-rich-thresholds-and-empty-background',r.read())
        with urllib.request.urlopen(self.base+'/camera.jpg',timeout=2) as r:
            self.assertEqual(r.headers['X-Frame-Id'],'native-frame')
        self.assertEqual(self.call('/workflow')['native_route'],'/workflow')
        self.assertTrue(self.call('/model-switch/status')['integrated_workbench'])

    def test_integrated_workbench_forwards_control_once_and_keeps_pending_after_switch(self):
        self.integrate_native()
        self.call('/operation/metal',{'action':'preview','object_height_mm':56})
        self.assertEqual(self.calls,[('/operation/metal',{'action':'preview','object_height_mm':56})])
        self.calls.clear();self.choose()
        state=self.call('/operation/metal')
        self.assertEqual(state['result']['task_id'],'native-original')
        self.assertEqual(state['blockers'],['original_pending','selected_model_recognition_only'])
        with self.assertRaises(urllib.error.HTTPError):self.call('/operation',{'action':'start'})
        with self.assertRaises(urllib.error.HTTPError):self.call('/prompted/configure',{'confidence':.01})
        self.assertEqual(self.calls,[])
        for action in ('stop','heartbeat'):self.call('/operation',{'action':action})
        self.assertEqual([d['action'] for _,d in self.calls],['stop','heartbeat'])
        self.choose('bound')
        with urllib.request.urlopen(self.base+'/prompted/',timeout=2) as r:
            self.assertIn(b'native-rich-thresholds-and-empty-background',r.read())

    def test_integrated_admission_reads_board_directly_instead_of_http_self_call(self):
        calls=[]
        class Board:
            def request(self,path):
                calls.append(path)
                return 200,b'{"busy":false,"state":"stopped"}',{}
        config=dict(self.config,control_url='http://127.0.0.1:'+str(self.server.server_port))
        self.catalog.write_text(json.dumps(config),encoding='utf-8')
        with patch('switchable_console.json_exchange',side_effect=AssertionError('HTTP self-call')):
            selection=attach_to_workbench(self.server,Board(),self.catalog)
            selection._idle()
            self.assertEqual(calls.count('/operation'),1)
            self.assertEqual(set(calls),{'/','/monitor.js','/monitor.css','/board-station.js','/operation'})
            selection.close()

    def test_integrated_display_failure_keeps_full_ui_but_never_cached_operation(self):
        self.integrate_native()
        with urllib.request.urlopen(self.base+'/',timeout=2) as r: original=r.read()
        self.assertEqual(self.call('/operation')['result']['task_id'],'native-original')
        self.native_unavailable=True
        with urllib.request.urlopen(self.base+'/',timeout=2) as r: cached=r.read()
        self.assertEqual(original,cached)
        self.assertEqual(cached.count(b'/model-switch.js'),1)
        with self.assertRaises(urllib.error.HTTPError) as exc:self.call('/operation')
        self.assertEqual(exc.exception.code,503)
        self.assertEqual(self.calls,[])

    def test_integrated_native_inflight_visual_response_is_discarded(self):
        self.integrate_native()
        self.delay_result=True
        results=[]
        def read():
            try:self.call('/live-result');results.append(200)
            except urllib.error.HTTPError as exc:results.append(exc.code)
        thread=threading.Thread(target=read);thread.start()
        self.assertTrue(self.entered.wait(1))
        self.choose();self.release.set();thread.join(2)
        self.assertEqual(results,[503])

    def test_retired_preview_redirects_get_and_never_replays_control_post(self):
        server=make_redirect_server(0,self.config['control_url'])
        threading.Thread(target=server.serve_forever,daemon=True).start()
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self,*args):return None
        try:
            opener=urllib.request.build_opener(NoRedirect())
            with self.assertRaises(urllib.error.HTTPError) as exc:
                opener.open('http://127.0.0.1:'+str(server.server_port)+'/',timeout=2)
            self.assertEqual(exc.exception.code,302)
            self.assertEqual(exc.exception.headers['Location'],self.config['control_url']+'/')
            with self.assertRaises(urllib.error.HTTPError) as exc:
                opener.open(urllib.request.Request('http://127.0.0.1:'+str(server.server_port)+'/operation',data=b'{"action":"start"}'),timeout=2)
            self.assertEqual(exc.exception.code,409)
            self.assertEqual(self.calls,[])
        finally:server.shutdown();server.server_close()


if __name__ == '__main__': unittest.main()
