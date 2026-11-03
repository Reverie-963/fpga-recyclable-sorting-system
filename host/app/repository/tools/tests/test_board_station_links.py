import json
import io
from pathlib import Path
import sys
import threading
import unittest
import urllib.request
import urllib.error
from types import SimpleNamespace
from paramiko import SSHException,ChannelException
sys.path.insert(0,str(Path(__file__).parents[1]))
from board_station_links import make_control_proxy,preview_display,PinnedBoardHTTP
from restart_native_board_proxy import quiescent_state


class NativeProxyTests(unittest.TestCase):
    def test_existing_page_receives_board_plan_without_local_replanning_or_success_claim(self):
        choice={'official_label':'Drink can','label':'metal','object_height_mm':56}
        plan=[{'step':'home','command':'original-PS-command','duration_ms':5000}]
        source=dict(status='preview_only',compute_location='pynq_ps',choice=choice,plan=plan,
                    original_labels_by_frame=[[dict(label='Drink can',material='metal',confidence=.7)]])
        result=preview_display(source)
        self.assertEqual(result['status'],'preview_ready')
        self.assertIs(result['candidates'][0]['steps'],plan)
        self.assertIs(result['candidates'][0]['choice'],choice)
        self.assertFalse(result['physical_motion'])
        self.assertEqual(result['decisions'][0]['official_label'],'Drink can')
        self.assertFalse(preview_display(dict(compute_location='pynq_ps',status='blocked'))['candidates'])
        with self.assertRaisesRegex(ValueError,'identity_required'):preview_display(dict(source,compute_location='pc'))

    def setUp(self):
        outer=self;self.calls=[];self.fail=False;self.failure=None
        class Board:
            def request(self,path,body=None,content_type=None):
                outer.calls.append((path,body))
                if outer.failure is not None:raise outer.failure
                if outer.fail:raise ConnectionError('lost')
                return 200,b'{"compute_location":"pynq_ps"}',{'Content-Type':'application/json'}
        self.server=make_control_proxy(0,Board(),'http://127.0.0.1:1')
        # The local-origin check must use the actual dynamically assigned port.
        self.port=self.server.server_port
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.addCleanup(self.close)
    def close(self):self.server.shutdown();self.server.server_close();self.thread.join(2)
    def call(self,path,body=None,headers=None):
        req=urllib.request.Request('http://127.0.0.1:'+str(self.port)+path,
            data=json.dumps(body).encode() if body is not None else None,
            headers=headers or {'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=2) as response:return json.load(response)
    def test_control_is_forwarded_to_board_only(self):
        self.assertEqual(self.call('/operation')['compute_location'],'pynq_ps')
        self.assertEqual(self.calls,[('/operation',None)])
    def test_board_failure_returns_unavailable_without_any_pc_control_fallback(self):
        self.fail=True
        with self.assertRaises(urllib.error.HTTPError) as error:self.call('/operation',{'action':'start'})
        self.assertEqual(error.exception.code,503)
        self.assertEqual(len(self.calls),1)
    def test_ssh_timeout_refusal_and_eof_return_json_503_without_post_retry(self):
        for failure in (SSHException('Timeout opening channel.'),ChannelException(2,'Connect failed'),EOFError()):
            with self.subTest(failure=type(failure).__name__):
                self.failure=failure;self.calls.clear()
                with self.assertRaises(urllib.error.HTTPError) as error:self.call('/operation',{'action':'heartbeat'})
                self.assertEqual(error.exception.code,503)
                self.assertEqual(json.load(error.exception)['error'],'board_control_or_visual_link_unavailable_no_fallback')
                error.exception.close();self.assertEqual(len(self.calls),1)
    def test_foreign_origin_cannot_reach_board(self):
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.call('/operation',{'action':'start'},{'Origin':'http://other.example'})
        self.assertEqual(error.exception.code,403);self.assertFalse(self.calls)
    def test_transport_identity_is_available_without_forwarding_or_exposing_token(self):
        result=self.call('/proxy-health')
        self.assertEqual(result,dict(transport_contract='shared-control-v0',automatic_retry=False,control_location='pynq_ps',control_channel_open_timeout_s=None,control_response_timeout_s=None))
        self.assertFalse(self.calls)


class PinnedTransportTests(unittest.TestCase):
    class Channel:
        def __init__(self):self.closed=False;self.timeouts=[]
        def settimeout(self,value):self.timeouts.append(value)
        def sendall(self,data):pass
        def makefile(self,*args):return io.BytesIO(b'HTTP/1.0 200 OK\r\nContent-Length: 11\r\n\r\n{"ok":true}')
        def close(self):self.closed=True
    def client(self,open_channel):
        transport=SimpleNamespace(is_active=lambda:True,open_channel=open_channel)
        return SimpleNamespace(get_transport=lambda:transport)
    def test_stalled_display_open_cannot_delay_heartbeat_or_stop(self):
        entered=threading.Event();release=threading.Event();errors=[];channels=[]
        def display(*args,**kwargs):
            entered.set();release.wait(2);raise SSHException('display stalled')
        def control(*args,**kwargs):
            self.assertEqual(kwargs['timeout'],3)
            c=self.Channel();channels.append(c);return c
        board=PinnedBoardHTTP(self.client(display),18810,'secret',control_client=self.client(control))
        def load():
            try:board.request('/station/frame.jpg')
            except ConnectionError as exc:errors.append(str(exc))
        worker=threading.Thread(target=load);worker.start()
        try:
            self.assertTrue(entered.wait(1))
            for action in ('heartbeat','stop'):
                code,raw,_=board.request('/operation',json.dumps({'action':action}).encode())
                self.assertEqual(code,200);self.assertTrue(json.loads(raw)['ok'])
                self.assertFalse(release.is_set())
            self.assertEqual(board.request('/operation')[0],200)
            self.assertTrue(all(c.closed and c.timeouts==[4] for c in channels))
        finally:release.set();worker.join(3)
        self.assertEqual(errors,['board_control_ssh_unavailable_no_retry'])
    def test_display_saturation_and_open_lock_have_bounded_wait(self):
        board=PinnedBoardHTTP(self.client(lambda *a,**k:self.Channel()),18810,'secret')
        for _ in range(3):board.display_gate.acquire()
        with self.assertRaisesRegex(ConnectionError,'capacity_busy'):board.request('/station/frame.jpg')
        for _ in range(3):board.display_gate.release()
        board.lock.acquire()
        try:
            with self.assertRaisesRegex(ConnectionError,'open_busy'):board.request('/operation',b'{}')
        finally:board.lock.release()
        self.assertEqual(board.request('/operation')[0],200)
    def test_finished_preview_readback_uses_reserved_control_connection(self):
        calls=[]
        def display(*args,**kwargs):raise AssertionError('display connection saturated')
        def control(*args,**kwargs):calls.append(1);return self.Channel()
        board=PinnedBoardHTTP(self.client(display),18810,'secret',control_client=self.client(control))
        for _ in range(3):board.display_gate.acquire()
        try:
            self.assertEqual(board.request('/board-preview-report?operation_id=finished')[0],200)
            self.assertEqual(calls,[1])
        finally:
            for _ in range(3):board.display_gate.release()
    def test_ssh_failures_do_not_reconnect_retry_or_leak_gate(self):
        calls=[]
        def fail(*args,**kwargs):calls.append(1);raise SSHException('Timeout opening channel.')
        board=PinnedBoardHTTP(self.client(fail),18810,'secret')
        for _ in range(4):
            with self.assertRaisesRegex(ConnectionError,'no_retry'):board.request('/station/frame.jpg')
        self.assertEqual(len(calls),4)
        board.transport=SimpleNamespace(is_active=lambda:True)
        with self.assertRaisesRegex(ConnectionError,'lost_no_reconnect'):board.request('/operation',b'{}')
        self.assertEqual(len(calls),4)
    def test_control_connection_must_be_independent(self):
        client=self.client(lambda *a,**k:self.Channel())
        with self.assertRaisesRegex(ValueError,'independent_control'):
            PinnedBoardHTTP(client,18810,'secret',control_client=client)


class ProxyUpdateGuardTests(unittest.TestCase):
    def test_stopped_blocked_station_can_be_updated_without_review_or_clearing(self):
        state=dict(compute_location='pynq_ps',busy=False,state='stopped',reason='operator_stop',
                   board_ownership={'status':'blocked','reason':'operator_stop'},operation_id='old',board_root='native12')
        preserved=quiescent_state(state)
        self.assertEqual(preserved['board_ownership'],state['board_ownership'])
        self.assertEqual(preserved['state'],'stopped')
        self.assertEqual(state['board_ownership']['status'],'blocked')
    def test_live_unknown_or_pc_state_cannot_be_updated(self):
        state=dict(compute_location='pynq_ps',busy=False,board_ownership={'status':'blocked'})
        for change in ({'busy':True},{'compute_location':'pc'},{'board_ownership':{'status':'active'}},{'board_ownership':{}}):
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,'quiescent'):
                quiescent_state(dict(state,**change))


if __name__=='__main__':unittest.main()
