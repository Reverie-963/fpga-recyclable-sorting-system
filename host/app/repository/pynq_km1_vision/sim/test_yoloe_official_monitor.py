import json
import threading
import time
import unittest
import urllib.error
import urllib.request
from types import SimpleNamespace

from src.yoloe_official_monitor import Playground, make_server, parse_classes, validate_box


class FakeDetector:
    execution = {'device': 'test'}
    hashes = {}
    vocabulary = ['person', 'bottle']
    embeddings = {}

    def prepare(self, payload):
        if payload.get('mode') == 'invalid':
            raise ValueError('invalid mode')
        return dict(mode=payload.get('mode', 'free'), confidence=.25, classes=[], vocabulary_count=2)


class OfficialMonitorTests(unittest.TestCase):
    def test_classes_preserve_phrases_and_remove_duplicates(self):
        self.assertEqual(parse_classes(' person,shoe box，person\nwater bottle '), ['person','shoe box','water bottle'])

    def test_classes_reject_empty_excessive_or_invalid_values(self):
        for value in ['', ['x']*81, [None], ['x'*101], 7]:
            with self.subTest(value=str(value)[:20]), self.assertRaises(ValueError):
                parse_classes(value)

    def test_reference_uses_original_image_pixels(self):
        self.assertEqual(validate_box([10,20,80,90], (100,200,3)), [10.,20.,80.,90.])

    def test_reference_rejects_nonfinite_reversed_and_outside_box(self):
        for box in [[0,0,500,50], [40,0,20,20], [0,0,1,1], [0,0,float('nan'),40], [False,0,10,10]]:
            with self.subTest(box=box), self.assertRaises(ValueError):
                validate_box(box,(100,200,3))

    def test_camera_source_is_local_only(self):
        for url in ['https://example.com/frame.jpg','http://127.0.0.1/private','http://user@localhost/frame.jpg','http://localhost/frame.jpg?x=1']:
            with self.subTest(url=url), self.assertRaises(ValueError):
                Playground(FakeDetector(),url)

    def test_reconfiguration_clears_old_result_and_advances_generation(self):
        p=Playground(FakeDetector(),'http://127.0.0.1:8780/frame.jpg')
        p.result={'old':True}
        p.completed.append(time.monotonic())
        p.configure(dict(mode='text',enabled=False))
        self.assertIsNone(p.result)
        self.assertEqual(p.generation,1)
        self.assertFalse(p.enabled)
        self.assertEqual(list(p.completed),[])

    def test_failed_configuration_preserves_active_settings(self):
        p=Playground(FakeDetector(),'http://localhost/frame.jpg')
        before=dict(p.settings)
        with self.assertRaises(ValueError):
            p.configure(dict(mode='invalid'))
        self.assertEqual(p.settings,before)
        self.assertEqual(p.generation,0)

    def test_upstream_age_is_included_in_expiry(self):
        p=Playground(FakeDetector(),'http://localhost/frame.jpg')
        self.assertFalse(p.fresh(dict(received=time.monotonic()-3,upstream_age_ms=2100)))
        self.assertTrue(p.fresh(dict(received=time.monotonic()-.1,upstream_age_ms=100)))

    def test_expired_or_paused_result_is_not_published(self):
        p=Playground(FakeDetector(),'http://localhost/frame.jpg')
        p.frame=dict(received=time.monotonic(),upstream_age_ms=10)
        p.result=dict(received=time.monotonic()-6,upstream_age_ms=10,data={})
        self.assertIsNone(p.latest())
        p.result=dict(received=time.monotonic(),upstream_age_ms=10,data={'mode':'free'})
        self.assertIsNotNone(p.latest())
        p.enabled=False
        self.assertIsNone(p.latest())

    def test_http_page_origin_and_malformed_input(self):
        p=Playground(FakeDetector(),'http://localhost/frame.jpg')
        server=make_server(p,0)
        thread=threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        base='http://127.0.0.1:'+str(server.server_port)
        try:
            with urllib.request.urlopen(base) as r:
                self.assertIn('referenceCanvas',r.read().decode())
            with urllib.request.urlopen(base+'/health') as r:
                self.assertFalse(json.load(r)['automatic_motion_allowed'])
            req=urllib.request.Request(base+'/configure',data=b'[]',headers={'Content-Type':'application/json'})
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(req)
            self.assertEqual(error.exception.code,400)
            req=urllib.request.Request(base+'/health',headers={'Origin':'https://example.com'})
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(req)
            self.assertEqual(error.exception.code,403)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)


if __name__=='__main__':
    unittest.main()
