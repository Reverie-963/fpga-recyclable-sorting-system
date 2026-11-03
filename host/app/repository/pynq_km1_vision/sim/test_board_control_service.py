import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

from src.board_control_service import BoardTasks,make_server


class Runtime:
    def __init__(self):self.entered=threading.Event();self.release=threading.Event();self.calls=[]
    def preview(self,expected,material,check,task_id):
        self.calls.append(material);self.entered.set()
        if not self.release.wait(3):raise ValueError('test timed out')
        check()
        return dict(status='preview_ready',task_id=task_id,physical_motion_attempted=False,execution_enabled=False)


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.runtime=Runtime()
        self.tasks=BoardTasks(self.runtime,Path(self.temp.name)/'own.db',Path(self.temp.name)/'results')
        self.server=make_server(self.tasks,'t'*40,0)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.addCleanup(self.close)
        self.url='http://127.0.0.1:'+str(self.server.server_port)
        self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def close(self):
        self.runtime.release.set();self.server.shutdown();self.server.server_close();self.thread.join(2);self.tasks.close()

    def call(self,path,data=None,token='t'*40):
        request=urllib.request.Request(self.url+path,data=json.dumps(data).encode() if data is not None else None,
            headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
        with self.opener.open(request,timeout=2) as response:return json.load(response)

    def submit(self,owner='web',request_id='first'):
        return self.call('/tasks',dict(owner=owner,request_id=request_id,material='AUTO',expected={}))

    def finished(self,task):
        deadline=time.monotonic()+2
        while time.monotonic()<deadline:
            result=self.call('/tasks/'+task)
            if result['status']=='finished':return result
            time.sleep(.01)
        self.fail('preview did not finish')

    def test_web_voice_cli_share_one_real_admission_endpoint(self):
        accepted=self.submit();self.assertTrue(self.runtime.entered.wait(1))
        for owner in ('voice','cli'):
            with self.assertRaises(urllib.error.HTTPError) as error:self.submit(owner,owner)
            self.assertEqual(error.exception.code,409)
        self.call('/heartbeat',dict(lease=accepted['lease']))
        self.runtime.release.set()
        result=self.finished(accepted['task_id'])
        self.assertEqual(result['report']['status'],'preview_ready')
        self.assertFalse(result['report']['execution_enabled'])
        self.assertEqual(self.tasks.status()['ownership']['status'],'idle')
        with self.assertRaises(urllib.error.HTTPError):self.submit('voice','first')

    def test_stop_latches_and_worker_cannot_clear_it(self):
        accepted=self.submit();self.assertTrue(self.runtime.entered.wait(1))
        self.call('/stop',dict(reason='simulated preview stop'))
        self.runtime.release.set()
        result=self.finished(accepted['task_id'])
        self.assertEqual(result['report']['status'],'blocked')
        self.assertEqual(self.tasks.status()['ownership']['status'],'blocked')
        with self.assertRaises(urllib.error.HTTPError):self.submit('cli','new')

    def test_wrong_token_and_motion_routes_do_not_reach_runtime(self):
        with self.assertRaises(urllib.error.HTTPError) as error:self.call('/health',token='w'*40)
        self.assertEqual(error.exception.code,403)
        for route in ('/execute','/home','/reset','/review'):
            with self.assertRaises(urllib.error.HTTPError) as error:self.call(route,{})
            self.assertEqual(error.exception.code,404)
        self.assertEqual(self.runtime.calls,[])


if __name__=='__main__':unittest.main()
