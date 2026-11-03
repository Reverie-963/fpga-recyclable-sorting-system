import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
import threading
import urllib.request
import urllib.error
import numpy as np
from src.empty_background_records import EmptyBackgroundRecords


class EmptyBackgroundTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.store=EmptyBackgroundRecords(Path(self.tmp.name)/'records')
        self.packet=dict(jpeg=b'original-camera-jpeg',bgr=np.zeros((20,30,3),np.uint8),
            identity=('camera-stream','123'),received=time.monotonic(),upstream_age_ms=10)

    def test_confirmation_and_freshness_required_without_writing(self):
        for confirmed in (False,1,'true',None):
            with self.assertRaises(ValueError):self.store.save(self.packet,confirmed_empty=confirmed,settings_sha256='a'*64)
        for packet in (None,dict(self.packet,received=time.monotonic()-3)):
            with self.assertRaises(ValueError):self.store.save(packet,confirmed_empty=True,settings_sha256='a'*64)
        self.assertFalse(self.store.root.exists())

    def test_exact_image_identity_persistence_and_no_acceptance_claim(self):
        a=self.store.save(self.packet,confirmed_empty=True,settings_sha256='a'*64)
        b=self.store.save(self.packet,confirmed_empty=True,settings_sha256='a'*64)
        self.assertNotEqual(a['id'],b['id'])
        restarted=EmptyBackgroundRecords(self.store.root)
        self.assertEqual(restarted.image(a['id']),self.packet['jpeg'])
        self.assertEqual(a['image_sha256'],hashlib.sha256(self.packet['jpeg']).hexdigest())
        self.assertEqual(a['source_frame_id'],'123')
        self.assertEqual(restarted.list()[0]['id'],b['id'])
        for key in ('automatically_verified_empty','model_trained','board_baseline_changed'):
            self.assertFalse(a[key])

    def test_tampering_and_traversal_fail(self):
        record=self.store.save(self.packet,confirmed_empty=True,settings_sha256='a'*64)
        (self.store.root/(record['id']+'.jpg')).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'校验'):self.store.image(record['id'])
        for record_id in ('../settings','../../secret','x'*32):
            with self.assertRaises(ValueError):self.store.image(record_id)

    def test_unpublished_orphan_photo_not_listed(self):
        self.store.root.mkdir()
        (self.store.root/('a'*32+'.jpg')).write_bytes(b'partial')
        self.assertEqual(self.store.list(),[])

    def test_http_capture_list_and_photo_with_synthetic_frame(self):
        from src.yoloe_official_monitor import make_server
        from src.yoloe_recyclable_monitor import RecyclablePlayground
        from src.prompted_recyclables import read_policy,settings_for
        class Detector:
            policy=read_policy();execution={};hashes={};embeddings={};vocabulary=[]
            def prepare(self,payload):return settings_for(self.policy,payload)
        playground=RecyclablePlayground(Detector(),'http://127.0.0.1:8780/frame.jpg',
            Path(self.tmp.name)/'settings.json',8811)
        playground.frame=self.packet
        server=make_server(playground,0,'yoloe_recyclables.html')
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base='http://127.0.0.1:'+str(server.server_port)
        def request(path,payload=None):
            data=None if payload is None else json.dumps(payload).encode()
            return urllib.request.urlopen(urllib.request.Request(base+path,data=data,
                headers={'Content-Type':'application/json'}),timeout=3)
        try:
            with self.assertRaises(urllib.error.HTTPError) as error:request('/backgrounds',{})
            self.assertEqual(error.exception.code,400)
            with request('/backgrounds',dict(confirmed_empty=True)) as r:record=json.load(r)
            with request('/backgrounds') as r:self.assertEqual(json.load(r)['records'][0]['id'],record['id'])
            with request(record['image_url']) as r:self.assertEqual(r.read(),self.packet['jpeg'])
            with self.assertRaises(urllib.error.HTTPError):request('/background/not-an-id.jpg')
            self.assertFalse(playground.enabled)
        finally:
            server.shutdown();server.server_close();thread.join(timeout=3)


if __name__=='__main__':unittest.main()
