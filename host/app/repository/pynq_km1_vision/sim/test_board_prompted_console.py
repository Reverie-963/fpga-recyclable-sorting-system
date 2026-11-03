import json
import threading
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

from src.board_prompted_console import create_board_console,BoardPreviewBackend


class Board:
    def __init__(self):self.calls=[];self.fail=False
    def health(self):return dict(compute_location='pynq_ps',execution_enabled=False)
    def plan(self,payload):
        self.calls.append(payload)
        if self.fail:raise ConnectionError('disconnected')
        return dict(status='preview_ready',compute_location='pynq_ps',physical_motion_attempted=False)


class ConsoleTests(unittest.TestCase):
    def setUp(self):
        class Operations:
            backend=None
            def close(self):pass
            def status(self):return dict(voice_service={},configured=False)
        self.board=Board()
        self.server=create_board_console(0,'http://127.0.0.1:1',self.board,operations=Operations())
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.addCleanup(self.close)
        self.url='http://127.0.0.1:'+str(self.server.server_port)
    def close(self):self.server.shutdown();self.server.server_close();self.thread.join(2)
    def call(self,path,data=None,headers=None):
        request=urllib.request.Request(self.url+path,data=json.dumps(data).encode() if data is not None else None,
            headers=headers or {'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=2) as response:return response.read()
    def test_existing_route_plan_reaches_board_and_not_pc_upstream(self):
        result=json.loads(self.call('/prompted/plan',{}))
        self.assertEqual(result['compute_location'],'pynq_ps')
        self.assertEqual(self.board.calls,[{}])
    def test_disconnected_board_never_falls_back_to_pc_planner(self):
        self.board.fail=True
        with self.assertRaises(urllib.error.HTTPError) as error:self.call('/prompted/plan',{})
        self.assertEqual(error.exception.code,503)
        self.assertEqual(len(self.board.calls),1)
    def test_status_note_and_origin_protection(self):
        self.assertIn(b'board-control-note',self.call('/'))
        self.assertTrue(json.loads(self.call('/board-control'))['connected'])
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.call('/prompted/plan',{},headers={'Origin':'http://outside.example'})
        self.assertEqual(error.exception.code,403)
        self.assertFalse(self.board.calls)

    def test_motion_window_preview_uses_board_but_execution_preserves_original_backend(self):
        class Original:
            def __init__(self):self.calls=[];self.journal_dir='original-ledger'
            def cycle(self,material,execute):self.calls.append((material,execute));return {'existing_motion_path':True}
        original=Original();backend=BoardPreviewBackend(original,self.board)
        result=backend.cycle('METAL',False)
        self.assertEqual(result['status'],'preview_only')
        self.assertEqual(self.board.calls,[dict(material='METAL',client='motion-window-preview')])
        self.assertEqual(original.calls,[])
        self.assertEqual(backend.journal_dir,'original-ledger')
        self.assertEqual(backend.cycle('METAL',True),{'existing_motion_path':True})
        self.assertEqual(original.calls,[('METAL',True)])

    def test_cutover_disables_pc_motion_and_home_while_board_preview_remains_available(self):
        class Original:
            def check(self):return []
            def cycle(self,*args):raise AssertionError('must not invoke PC actuator')
            def home(self):raise AssertionError('must not invoke PC actuator')
        backend=BoardPreviewBackend(Original(),self.board,disable_pc_motion=True)
        self.assertIn('board_executor_not_connected',backend.check())
        self.assertEqual(backend.cycle('METAL',False)['status'],'preview_only')
        with self.assertRaisesRegex(ValueError,'board_executor_not_connected'):backend.cycle('METAL',True)
        with self.assertRaisesRegex(ValueError,'board_executor_not_connected'):backend.home()


if __name__=='__main__':unittest.main()
