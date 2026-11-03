"""Collection safety/data-integrity checks; no camera, serial, GPU or network."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest

import cv2
import numpy as np

SPEC = importlib.util.spec_from_file_location('final_collection', Path(__file__).parents[1] / 'collect_recyclables_final.py')
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


class FinalCollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        registry = dict(objects=[dict(object_id=m+'-old', label=m, legacy_model_class=m,
                    reserved_split='train') for m in mod.NAMES])
        # Latest ownership deliberately differs from the older registry.
        registry['objects'].append(dict(object_id='can-dark', label='dark can', legacy_model_class='can', reserved_split='train'))
        previous = dict(images=[dict(object_ids=[m+'-old'], split='train') for m in mod.NAMES]+
                              [dict(object_ids=['can-dark'], split='val')])
        self.registry = registry
        self.previous = previous
        for name, data in [('registry.json',registry), ('previous.json',previous)]:
            (self.base/name).write_text(json.dumps(data), encoding='utf-8')
        self.args = argparse.Namespace(output=self.base/'dataset', registry=self.base/'registry.json',
                   previous_manifest=self.base/'previous.json', session='train-session-1', split='train',
                   workcell='fixed-1', camera_url='http://127.0.0.1:8772', seconds=8)
        self.c = mod.Collection(self.args)

    def frames(self, seed=1, motion=False):
        now = time.monotonic()
        self.c.frames.clear()
        rng = np.random.default_rng(seed)
        base = rng.integers(30,220,(480,640),dtype=np.uint8)
        for i in range(3):
            gray = (np.clip(base.astype(np.int16)+i,0,255)).astype(np.uint8)
            if motion:
                gray = np.roll(gray, i*20, axis=1)
            ok, jpeg = cv2.imencode('.jpg',gray)
            self.assertTrue(ok)
            gray = cv2.imdecode(jpeg,cv2.IMREAD_GRAYSCALE)
            self.c.frames.append(dict(jpeg=jpeg.tobytes(), gray=gray, received=now-.7+.35*i,
                      received_sequence_ns=100+i, received_at='2026-10-28T12:00:00Z', source_age=.1))
        self.c.error = None
        self.c.armed_at = now-2

    def test_latest_actual_ownership_wins_and_test_is_not_created(self):
        registry = mod.make_registry(self.registry,self.previous)
        self.assertEqual(next(o for o in registry if o['object_id']=='can-dark')['reserved_split'],'val')
        self.assertTrue(all('can-dark' not in r['object_ids'] for r in self.c.plan))
        self.assertTrue(all(r['split']!='test' for r in self.c.rows))

    def test_historical_cross_split_rejected(self):
        self.previous['images'].append(dict(object_ids=['can-old'],split='val'))
        with self.assertRaises(ValueError):
            mod.make_registry(self.registry,self.previous)

    def test_source_bin_complete_and_wrong_bin_semantics(self):
        plan = self.c.plan
        for obj in self.c.objects:
            if obj['reserved_split']!='train':
                continue
            for scene in ('source','bin'):
                rows=[r for r in plan if r['object_ids']==[obj['object_id']] and r['scene_type']==scene]
                self.assertEqual(len(rows),12)
                self.assertEqual(len({r['pose_id'] for r in rows}),12)
        classes = {o['object_id']:o['material'] for o in self.c.objects}
        for row in plan:
            if row['scene_type']=='wrong_bin':
                self.assertNotEqual(classes[row['object_ids'][0]],row['bin_material'])

    def test_one_full_workcell_empty_frame_without_artificial_quota(self):
        self.assertEqual(sum(r['scene_type']=='empty' for r in self.c.plan),1)
        self.frames()
        self.assertIsNone(self.c.capture(self.c.plan[0]))
        self.assertEqual(self.c.rows[0]['quality']['checked_region'],[0,0,640,480])
        self.frames()
        # Genuine lighting change confined to the paper-bin area must count;
        # the previous source-only comparison incorrectly discarded it.
        for frame in self.c.frames:
            gray=frame['gray'].copy()
            gray[270:410,420:600]=145
            _, jpeg=cv2.imencode('.jpg',gray)
            frame['jpeg']=jpeg.tobytes()
            frame['gray']=cv2.imdecode(jpeg,cv2.IMREAD_GRAYSCALE)
        self.assertIsNone(self.c.capture(self.c.plan[0]))
        self.assertEqual(len(self.c.rows),2)

    def test_empty_station_can_recur_in_a_separate_real_session(self):
        self.frames()
        self.c.capture(self.c.plan[0])
        self.args.session='another-actual-session'
        self.frames()
        for frame in self.c.frames:
            frame['jpeg']+=b'\n'
        self.assertIsNone(self.c.capture(self.c.plan[0]))

    def test_stale_frames_cannot_arm(self):
        self.frames()
        self.c.frames[-1]['received']=time.monotonic()-5
        with self.assertRaises(ValueError):
            self.c.command('capture')

    def test_motion_and_pre_request_frames_do_not_save(self):
        self.frames(motion=True)
        self.assertIn('动',self.c.capture(self.c.plan[0]))
        self.assertEqual(self.c.rows,[])
        self.frames()
        self.c.armed_at=time.monotonic()
        self.assertIn('新帧',self.c.capture(self.c.plan[0]))
        self.assertEqual(self.c.rows,[])

    def test_one_candidate_pending_masks_and_restart_dedup(self):
        self.frames()
        self.assertIsNone(self.c.capture(self.c.plan[0]))
        self.assertEqual(len(self.c.rows),1)
        row=self.c.rows[0]
        self.assertEqual(row['annotation_status'],'pending')
        self.assertFalse(row['source_board_timestamp_available'])
        self.assertFalse((self.args.output/'labels').exists())
        restored=mod.Collection(self.args)
        self.assertEqual(restored.index(),1)
        self.assertEqual(len(restored.rows),1)
        # Different bytes of effectively identical content must not inflate coverage.
        old=self.c
        self.c=restored
        self.frames()
        for frame in self.c.frames:
            frame['jpeg']+=b'\n'
        reason=self.c.capture(self.c.plan[0])
        self.assertIn('近似重复',reason)
        self.assertEqual(len(self.c.rows),1)
        self.c=old

    def test_undo_preserves_raw_and_restores_step(self):
        self.frames()
        self.c.capture(self.c.plan[0])
        original=self.args.output/self.c.rows[0]['image']
        self.c.command('undo')
        self.assertEqual(self.c.index(),0)
        self.assertEqual(self.c.rows,[])
        self.assertFalse(original.exists())
        self.assertEqual(len(list((self.c.session_dir/'rejected').glob('*.jpg'))),1)
        self.assertEqual(mod.Collection(self.args).index(),0)

    def test_session_cannot_change_split(self):
        self.args.split='val'
        with self.assertRaises(ValueError):
            mod.Collection(self.args)

    def test_earlier_gap_preserves_completed_progress_and_next_pending(self):
        self.c.events=[dict(step_id=step['id'],status='skipped') for step in self.c.plan[1:]]
        state=self.c.state()
        self.assertEqual(state['index'],0)
        self.assertEqual(state['completed_steps'],len(self.c.plan)-1)
        self.assertIn('无需重拍',state['next'])

    def prepared(self, count=1):
        objects = mod.make_registry(self.registry, self.previous)
        objects += [dict(object_id='new-plastic-'+str(i), label='new bottle '+str(i), material='plastic',
                        reserved_split='train', previously_used_for_development=False, requires_reference=True,
                        reference_image=None, reference_sha256=None) for i in range(count)]
        path = self.base/'prepared.json'
        path.write_text(json.dumps(dict(schema_version='recyclable-prepared-registry-v1',objects=objects)),encoding='utf-8')
        self.args.registry=path
        self.args.output=self.base/'prepared-dataset'
        self.c=mod.Collection(self.args)

    def test_new_references_resume_and_undo_without_entering_training(self):
        self.prepared()
        self.assertEqual(self.c.plan[0]['scene_type'],'reference')
        self.frames()
        with self.assertRaisesRegex(ValueError,'逐件'):
            self.c.command('auto')
        with self.assertRaisesRegex(ValueError,'不能跳过'):
            self.c.command('skip')
        self.assertIsNone(self.c.capture(self.c.plan[0]))
        self.assertEqual(self.c.state()['registered'],1)
        self.assertEqual(self.c.state()['saved'],0)
        self.assertEqual(self.c.rows,[])
        resumed=mod.Collection(self.args)
        self.assertEqual(resumed.plan[resumed.index()]['scene_type'],'empty')
        self.assertTrue(resumed.objects[-1]['reference_image'])
        resumed.command('undo')
        self.assertIsNone(resumed.objects[-1]['reference_image'])
        self.assertEqual(resumed.index(),0)

    def test_registration_is_retained_when_switching_to_validation(self):
        self.prepared()
        self.frames()
        self.c.capture(self.c.plan[0])
        self.args.session='separate-validation-session'
        self.args.split='val'
        validation=mod.Collection(self.args)
        self.assertEqual(validation.state()['registered'],1)
        self.assertEqual(validation.state()['registration_total'],1)
        self.assertEqual(validation.state()['saved'],0)
        self.assertFalse(any(r['scene_type']=='reference' for r in validation.plan))

    def test_duplicate_reference_does_not_register_two_objects(self):
        self.prepared(2)
        self.frames()
        self.assertIsNone(self.c.capture(self.c.plan[0]))
        self.frames()
        self.assertIn('近似重复',self.c.capture(self.c.plan[1]))
        self.assertEqual(self.c.state()['registered'],1)

    def test_prepared_registry_rejects_reassigned_history_and_old_test_objects(self):
        objects=mod.make_registry(self.registry,self.previous)
        objects[0]['reserved_split']='test'
        objects[0]['previously_used_for_development']=False
        with self.assertRaises(ValueError):
            mod.make_registry(dict(schema_version='recyclable-prepared-registry-v1',objects=objects),self.previous)


if __name__=='__main__':
    unittest.main()
