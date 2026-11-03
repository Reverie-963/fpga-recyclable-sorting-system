"""Current ledger receipts and review evidence stay distinct from auto success."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np

from src.monitor_tasks import TaskTelemetry,project,current_workflow,recorded_frame


def save(path,value):
    path.write_text(json.dumps(value),encoding='utf-8')


class CurrentTaskTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name);self.task='a'*32;self.path=self.folder/(self.task+'.json')
        command='{#005P1300T1000!}';encoded=command.encode().hex()
        self.record=dict(task_id=self.task,mode='known_physical',status='needs_review',config_sha256='config',
            choice=dict(label='plastic',bin_id='bin_plastic',pick_pixel=[3,4],
                geometry=dict(status=1,m00=4,m10=8,m01=12,min_col=1,max_col=3,min_row=2,max_row=4)),
            plan=[dict(step='close',command=command,duration_ms=1000)],
            steps=[dict(step='close',status='sent_and_waited_unverified',transport_receipt=dict(
                sent_hex=encoded,received_hex=encoded,echo_matched=True,completed_movement_wait=True))],
            statistics=dict(backend='pl_board_local',geometry_backend='pc_numpy',assets={'bit':'hash'},config_sha256='c',stream_id='s'),
            motion_baseline=dict(baseline_frame_id=2,baseline_bgr_sha256='base',stream_id='s',config_sha256='c',evidence={}),
            reason='semantic_failed',elapsed_s=36.5,evidence=[])
        image=np.full((8,10,3),80,np.uint8);ok,buffer=cv2.imencode('.png',image);self.assertTrue(ok)
        raw=buffer.tobytes();digest=hashlib.sha256(raw).hexdigest();name=self.task+'.before'
        (self.folder/(name+'.png')).write_bytes(raw)
        save(self.folder/(name+'.observation.json'),dict(png_sha256=digest,decoded_bgr_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
            observation=dict(model_version='model',runtime_id='runtime',source_frame_id=7,image_size=[10,8])))
        self.record['evidence']=[dict(image=name+'.png',observation=name+'.observation.json',png_sha256=digest)]
        save(self.path,self.record)

    def project(self):return project(self.path,self.record)

    def test_legacy_dimensions_derive_only_from_hash_verified_original_pixels(self):
        item=self.record['evidence'][0];path=self.folder/item['observation']
        meta=json.loads(path.read_text(encoding='utf-8'))
        meta['observation'].pop('image_size');meta['observation']['source_stream_id']='original-stream'
        save(path,meta);original=path.read_bytes()
        _,observation=recorded_frame(self.folder,item,pixels=True)
        self.assertEqual(observation['image_size'],[10,8])
        self.assertEqual(observation['image_size_source'],'hash_verified_decoded_legacy_record')
        self.assertEqual(path.read_bytes(),original)
        meta['decoded_bgr_sha256']='0'*64;save(path,meta)
        with self.assertRaisesRegex(ValueError,'record_pixels_mismatch'):
            recorded_frame(self.folder,item,pixels=True)

    def test_explicit_wrong_dimensions_are_never_repaired(self):
        item=self.record['evidence'][0];path=self.folder/item['observation']
        for size in ([8,10],None,[]):
            meta=json.loads(path.read_text(encoding='utf-8'));meta['observation']['image_size']=size;save(path,meta)
            with self.assertRaisesRegex(ValueError,'record_image_size_mismatch'):
                recorded_frame(self.folder,item,pixels=True)

    def test_missing_dimensions_without_frame_identity_are_rejected(self):
        item=self.record['evidence'][0];path=self.folder/item['observation']
        meta=json.loads(path.read_text(encoding='utf-8'));meta['observation'].pop('image_size');save(path,meta)
        with self.assertRaisesRegex(ValueError,'record_image_size_mismatch'):
            recorded_frame(self.folder,item,pixels=True)

    def test_legacy_decimal_frame_header_retains_original_identity(self):
        item=self.record['evidence'][0];path=self.folder/item['observation']
        meta=json.loads(path.read_text(encoding='utf-8'));meta['observation'].pop('image_size')
        meta['observation'].update(source_frame_id='7',source_stream_id='original',station_vision={'frame_id':7})
        save(path,meta)
        _,obs=recorded_frame(self.folder,item,pixels=True)
        self.assertEqual(obs['source_frame_id'],'7');self.assertEqual(obs['image_size'],[10,8])
        for invalid in ('07','-1','7.0',True):
            meta['observation']['source_frame_id']=invalid;save(path,meta)
            with self.assertRaisesRegex(ValueError,'record_image_size_mismatch'):
                recorded_frame(self.folder,item,pixels=True)
        meta['observation']['source_frame_id']='8';save(path,meta)
        with self.assertRaisesRegex(ValueError,'record_image_size_mismatch'):
            recorded_frame(self.folder,item,pixels=True)

    def test_new_receipts_count_only_matching_sent_bytes_and_echoes(self):
        d=self.project();self.assertEqual(d['echoed_commands'],1);self.assertEqual(d['waited_commands'],1)
        self.assertEqual(d['geometry']['backend'],'pc_numpy');self.assertEqual(d['raw_status'],'needs_review')
        self.assertFalse(d['physical_success_verified']);self.assertFalse(d['automatic_motion_allowed'])
        receipt=self.record['steps'][0]['transport_receipt'];receipt['received_hex']='00'
        self.assertEqual(self.project()['echoed_commands'],0)
        receipt['sent_hex']=b'{#005P1500T1000!}'.hex();receipt['received_hex']=receipt['sent_hex']
        d=self.project();self.assertEqual(d['commands'][0]['state'],'identity_mismatch')
        self.assertEqual(d['echoed_commands'],0);self.assertEqual(d['waited_commands'],0)

    def test_intent_and_preview_are_never_counted_as_actual_sends(self):
        self.record['steps'][0].pop('transport_receipt')
        self.assertEqual(self.project()['written_commands'],0)
        self.record['mode']='preview';self.assertEqual(self.project()['record_kind'],'preview')

    def test_review_requires_the_matching_saved_task_snapshot(self):
        value=dict(task_id=self.task,pending=copy.deepcopy(self.record),observation='Operator confirmed stable placement',outcome='operator_resolved_not_automatic_success')
        save(self.folder/(self.task+'.review.json'),value)
        d=self.project();self.assertIsNotNone(d['operator_review']);self.assertEqual(d['raw_status'],'needs_review')
        value['pending']['steps']=[];save(self.folder/(self.task+'.review.json'),value)
        self.assertIsNone(self.project()['operator_review'])

    def test_supplement_is_separate_and_requires_three_hash_bound_images(self):
        items=[]
        for i in range(3):
            original=self.record['evidence'][0];item=dict(image='extra-'+str(i)+'.png',observation='extra-'+str(i)+'.observation.json',png_sha256=original['png_sha256'])
            (self.folder/item['image']).write_bytes((self.folder/original['image']).read_bytes())
            meta=json.loads((self.folder/original['observation']).read_text(encoding='utf-8'))
            meta['observation'].update(source_frame_id=20+i,source_stream_id='s')
            save(self.folder/item['observation'],meta);items.append(item)
        value=dict(original_task_id=self.task,physical_motion=False,original_outcome_unchanged=True,
            status='supplemental_visual_evidence_pass',min_confidence=.7,frames=items,
            semantic=dict(status='visual_evidence_pass'),board_rgb=dict(status='board_rgb_evidence_pass'))
        save(self.folder/'recheck-extra.json',value);d=self.project()
        self.assertEqual(len(d['supplemental']),1);self.assertEqual(d['raw_status'],'needs_review')
        self.assertFalse(d['physical_success_verified'])
        value['frames']=[items[0]]*3;save(self.folder/'recheck-extra.json',value)
        self.assertEqual(self.project()['supplemental'],[])
        value['frames']=items;save(self.folder/'recheck-extra.json',value)
        (self.folder/items[0]['image']).write_bytes(b'changed')
        self.assertEqual(self.project()['supplemental'],[])

    def test_board_result_requires_assets_stream_and_exact_baseline(self):
        value=dict(status='board_rgb_evidence_pass',provenance=self.record['statistics'],
            baseline={k:v for k,v in self.record['motion_baseline'].items() if k!='evidence'},frames=[{}]*3)
        path=self.folder/(self.task+'.board-vision.json');save(path,value)
        self.assertEqual(self.project()['board_verification']['frames'],3)
        value['baseline']['baseline_frame_id']=999;save(path,value)
        self.assertIsNone(self.project()['board_verification'])

    def test_exact_original_pixels_and_selected_point_required(self):
        reader=TaskTelemetry(self.folder,refresh_s=0);d=reader.grasp_frame(dict(model_sha256='model'))
        self.assertTrue(d['available']);self.assertEqual(d['source_frame_id'],7)
        self.assertFalse(d['current_live_frame']);self.assertFalse(d['control_authorized'])
        (self.folder/(self.task+'.before.png')).write_bytes(b'changed')
        self.assertFalse(reader.grasp_frame({})['available'])

    def test_legacy_and_ancillary_records_cannot_replace_current_task(self):
        legacy=copy.deepcopy(self.record);legacy['task_id']='b'*32;legacy['statistics']['backend']='pl_http'
        save(self.folder/(legacy['task_id']+'.json'),legacy)
        save(self.folder/'recheck-not-task.json',dict(original_task_id=self.task,status='pass'))
        d=TaskTelemetry(self.folder,refresh_s=0).status({})
        self.assertEqual(d['task_id'],self.task);self.assertEqual(len(d['records']),1)

    def test_incomplete_newest_task_does_not_show_an_older_success(self):
        import os
        pending=self.folder/'pending.json';pending.write_text('{',encoding='utf-8')
        os.utime(pending,(self.path.stat().st_mtime+1,self.path.stat().st_mtime+1))
        reader=TaskTelemetry(self.folder,refresh_s=0)
        self.assertFalse(reader.status({})['available'])
        self.assertTrue(reader.status({},self.task)['available'])

    def test_workflow_exposes_only_allowlisted_fields(self):
        c=dict(statistics=dict(backend='pl_board_local',token='private'),min_confidence=.7,
            known=dict(motion_profile='compact'),secret='private')
        path=self.folder/'config.json';save(path,c);d=current_workflow(path)
        self.assertTrue(d['available']);self.assertNotIn('secret',d);self.assertNotIn('statistics',d)
        c['min_confidence']=float('nan');save(path,c);self.assertFalse(current_workflow(path)['available'])


if __name__=='__main__':unittest.main()
