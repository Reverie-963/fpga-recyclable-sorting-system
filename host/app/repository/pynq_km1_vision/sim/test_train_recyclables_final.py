"""Contract tests only: no model training, model downloads or device access."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.train_recyclables_final import (DEFAULT_RECIPE, check_final_dataset,
    describe_counts, evaluate_and_package, match_instances, new_counts,
    quality_gate, read_json, sha256)


class FinalDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.recipe = read_json(DEFAULT_RECIPE)
        self.recipe['image_size'] = [16, 16]
        self.rows, self.objects = [], {}
        self.rng = np.random.default_rng(19)
        for split in ('train', 'val', 'test'):
            self.recipe['dataset_minimums'][split] = dict(objects_per_class=1,
                source_views_per_object=1, bin_views_per_object=1,
                empty_images=1, wrong_bin_images=3, multi_images=3)
            for folder in ('images', 'labels'):
                (self.root / folder / split).mkdir(parents=True)
            ids = [split + '-' + name for name in self.recipe['class_names']]
            self.objects.update(zip(ids, self.recipe['class_names']))
            for cid, object_id in enumerate(ids):
                for scene in ('source', 'bin', 'wrong_bin'):
                    self.add_row(split, scene, [object_id], [cid])
            self.add_row(split, 'empty', [], [])
            for pair in ((0, 1), (0, 2), (1, 2)):
                self.add_row(split, 'multi', [ids[x] for x in pair], list(pair))
        self.save()

    def add_row(self, split, scene, ids, cids):
        name = 'image-%03d' % len(self.rows)
        cv2.imwrite(str(self.root / 'images' / split / (name+'.png')),
                    self.rng.integers(0, 256, (16, 16, 3), dtype=np.uint8))
        (self.root / 'labels' / split / (name+'.txt')).write_text(
            '\n'.join(str(cid)+' 0.1 0.1 0.8 0.1 0.8 0.8 0.1 0.8' for cid in cids), encoding='utf-8')
        row = dict(image=f'images/{split}/{name}.png', split=split, object_ids=ids,
                   session_id=split+'-segment', capture_session_id=split+'-actual-session',
                   workcell_id='fixture', scene_type=scene, annotation_status='reviewed', pose_id=name)
        if scene in ('bin', 'wrong_bin'):
            row['bin_material'] = self.recipe['class_names'][(cids[0]+int(scene=='wrong_bin')) % 3]
        self.rows.append(row)

    def save(self):
        (self.root / 'split_manifest.json').write_text(json.dumps(dict(
            schema_version='recyclable-final-dataset-v1', class_names=self.recipe['class_names'],
            object_classes=self.objects, images=self.rows)), encoding='utf-8')

    def test_complete_contract_reports_true_objects_scenes_and_hashes(self):
        result = check_final_dataset(self.root, self.recipe)
        self.assertEqual(result['images'], {'train':13, 'val':13, 'test':13})
        self.assertEqual(result['object_counts']['test:can'], 1)
        self.assertEqual(result['scene_counts']['val']['wrong_bin'], 3)
        self.assertEqual(len(set(result['split_sha256'].values())), 3)

    def test_real_capture_session_cannot_be_disguised_as_different_segments(self):
        self.rows[-1]['capture_session_id'] = 'train-actual-session'
        self.save()
        with self.assertRaisesRegex(ValueError, 'Capture session crosses'):
            check_final_dataset(self.root, self.recipe)

    def test_unreviewed_contours_and_unknown_objects_rejected(self):
        self.rows[0]['annotation_status'] = 'draft'
        self.save()
        with self.assertRaisesRegex(ValueError, 'reviewed'):
            check_final_dataset(self.root, self.recipe)
        self.rows[0]['annotation_status'] = 'reviewed'
        self.objects.pop('train-can')
        self.save()
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            check_final_dataset(self.root, self.recipe)

    def test_small_or_incomplete_data_cannot_start_final_training(self):
        recipe = read_json(DEFAULT_RECIPE)
        recipe['image_size'] = [16, 16]
        with self.assertRaisesRegex(ValueError, 'Need 3 different'):
            check_final_dataset(self.root, recipe)

    def test_per_material_minimums_are_enforced_individually(self):
        self.recipe['dataset_minimums']['train']['objects_per_class'] = dict(can=1,plastic=1,cardboard=1)
        self.assertEqual(check_final_dataset(self.root,self.recipe)['object_counts']['train:plastic'],1)
        self.recipe['dataset_minimums']['train']['objects_per_class']['plastic']=2
        with self.assertRaisesRegex(ValueError,'Need 2 different plastic'):
            check_final_dataset(self.root,self.recipe)

    def test_incomplete_material_minimum_map_is_rejected(self):
        self.recipe['dataset_minimums']['train']['objects_per_class'] = dict(can=1,plastic=1)
        with self.assertRaisesRegex(ValueError,'all three materials'):
            check_final_dataset(self.root,self.recipe)

    def test_duplicate_training_bytes_not_counted_as_more_data(self):
        first, second = self.root/self.rows[0]['image'], self.root/self.rows[1]['image']
        second.write_bytes(first.read_bytes())
        with self.assertRaisesRegex(ValueError, 'Duplicate image bytes'):
            check_final_dataset(self.root, self.recipe)

    def test_material_labels_not_replaced_by_destination_bin(self):
        row = next(r for r in self.rows if r['scene_type']=='wrong_bin')
        row['bin_material'] = self.objects[row['object_ids'][0]]
        self.save()
        with self.assertRaisesRegex(ValueError, 'Material/bin'):
            check_final_dataset(self.root, self.recipe)

    def test_object_polygon_disagreement_rejected(self):
        self.objects['train-can'] = 'plastic'
        self.save()
        with self.assertRaisesRegex(ValueError, 'classes/counts disagree'):
            check_final_dataset(self.root, self.recipe)


class FinalMetricsTests(unittest.TestCase):
    def test_final_recipe_has_16_objects_507_images_and_complete_training_parameters(self):
        recipe = read_json(DEFAULT_RECIPE)
        totals = {split:sum(p['objects_per_class'].values())*(p['source_views_per_object']+p['bin_views_per_object'])+
                  p['empty_images']+p['wrong_bin_images']+p['multi_images']
                  for split,p in recipe['dataset_minimums'].items()}
        self.assertEqual(totals,{'train':265,'val':121,'test':121})
        self.assertEqual({m:sum(p['objects_per_class'][m] for p in recipe['dataset_minimums'].values())
                          for m in recipe['class_names']}, {'can':7,'plastic':5,'cardboard':4})
        self.assertEqual(recipe['training']['epochs'],100)
        self.assertEqual(recipe['training']['patience'],25)
        self.assertFalse(recipe['training']['amp'])

    def test_duplicate_masks_and_wrong_classes_count_as_false_positives(self):
        mask = np.ones((4, 4), bool)
        truth = [dict(class_id=0, mask=mask)]
        predictions = [dict(class_id=2, confidence=.99, mask=mask),
                       dict(class_id=0, confidence=.9, mask=mask),
                       dict(class_id=0, confidence=.8, mask=mask)]
        result = match_instances(truth, predictions, .5)
        self.assertEqual(result['tp'], [1,0,0])
        self.assertEqual(result['fp'], [1,0,1])
        self.assertEqual(result['fn'], [0,0,0])

    def test_empty_false_alarm_and_missed_target_are_counted(self):
        mask = np.ones((4,4), bool)
        self.assertEqual(match_instances([], [dict(class_id=1,confidence=.9,mask=mask)], .5)
                         ['empty_false_positive_images'], 1)
        self.assertEqual(match_instances([dict(class_id=1,mask=mask)], [], .5)['fn'], [0,1,0])

    def test_good_overall_does_not_hide_a_failed_bin_material(self):
        recipe = read_json(DEFAULT_RECIPE)
        counts = new_counts()
        counts['tp'] = [10,10,10]
        counts['images'] = 10
        good = describe_counts(counts, recipe['class_names'])
        empty = describe_counts(new_counts(), recipe['class_names'])
        empty.update(empty_images=12, images=12)
        metrics = dict(overall=good, scenes={scene:copy.deepcopy(good)
            for scene in ('source','bin','wrong_bin','multi')})
        metrics['scenes']['empty'] = empty
        self.assertTrue(quality_gate(metrics, recipe)['passed'])
        metrics['scenes']['bin']['per_class']['plastic']['recall'] = .5
        self.assertIn('bin:plastic:recall_below_target', quality_gate(metrics, recipe)['reasons'])


class FinalPackagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.best = self.root/'best.pt'
        self.best.write_bytes(b'unit-test-checkpoint-no-model')
        self.output = self.root/'output'
        self.output.mkdir()
        self.recipe = read_json(DEFAULT_RECIPE)
        self.dataset_report = dict(dataset_sha256='frozen', split_sha256={'test':'heldout'})

    def run_package(self):
        return evaluate_and_package(None,self.best,self.root,self.root/'data.yaml',
            self.recipe,'cpu',self.output,self.dataset_report)

    @patch('src.train_recyclables_final.evaluate_split')
    def test_failed_validation_leaves_test_unscored(self, evaluate):
        evaluate.return_value = dict(gate=dict(passed=False,reasons=['bin:plastic']))
        result = self.run_package()
        self.assertEqual(result['status'],'validation_failed')
        self.assertFalse(result['test_evaluated'])
        self.assertEqual(evaluate.call_count,1)
        self.assertFalse((self.output/'test-reservation.json').exists())
        self.assertFalse((self.output/'package').exists())

    @patch('src.train_recyclables_final.check_final_dataset')
    @patch('src.train_recyclables_final.evaluate_split')
    def test_one_frozen_model_is_tested_then_packaged(self, evaluate, check):
        evaluate.return_value = dict(gate=dict(passed=True,reasons=[]))
        check.return_value = self.dataset_report
        result = self.run_package()
        self.assertEqual(result['status'],'quality_targets_met')
        self.assertEqual([c.args[3] for c in evaluate.call_args_list],['val','test'])
        self.assertEqual(sha256(self.output/'package/model.pt'),sha256(self.best))
        self.assertFalse(result['automatic_motion_allowed'])
        with self.assertRaises(FileExistsError):
            self.run_package()

    @patch('src.train_recyclables_final.check_final_dataset')
    @patch('src.train_recyclables_final.evaluate_split')
    def test_failed_or_interrupted_test_cannot_publish(self, evaluate, check):
        check.return_value = self.dataset_report
        evaluate.side_effect = [dict(gate=dict(passed=True,reasons=[])),
                                dict(gate=dict(passed=False,reasons=['can']))]
        result = self.run_package()
        self.assertEqual(result['status'],'test_failed')
        self.assertTrue((self.output/'test-reservation.json').exists())
        self.assertFalse((self.output/'package').exists())

    @patch('src.train_recyclables_final.check_final_dataset')
    @patch('src.train_recyclables_final.evaluate_split')
    def test_interrupted_test_keeps_reservation_and_no_package(self, evaluate, check):
        check.return_value = self.dataset_report
        evaluate.side_effect = [dict(gate=dict(passed=True,reasons=[])), RuntimeError('inference interrupted')]
        with self.assertRaisesRegex(RuntimeError,'interrupted'):
            self.run_package()
        self.assertTrue((self.output/'test-reservation.json').exists())
        self.assertEqual(read_json(self.output/'final-report.json')['status'],
                         'test_interrupted_do_not_retry_automatically')
        self.assertFalse((self.output/'package').exists())

    @patch('src.train_recyclables_final.check_final_dataset')
    @patch('src.train_recyclables_final.evaluate_split')
    def test_data_change_does_not_touch_heldout_test(self, evaluate, check):
        evaluate.return_value = dict(gate=dict(passed=True,reasons=[]))
        check.return_value = dict(self.dataset_report,dataset_sha256='changed')
        with self.assertRaisesRegex(ValueError,'Dataset changed'):
            self.run_package()
        self.assertEqual(evaluate.call_count,1)
        self.assertFalse((self.output/'test-reservation.json').exists())

    @patch('src.train_recyclables_final.check_final_dataset')
    @patch('src.train_recyclables_final.evaluate_split')
    def test_new_output_cannot_rescore_a_previously_used_dataset(self, evaluate, check):
        evaluate.return_value = dict(gate=dict(passed=True, reasons=[]))
        check.return_value = self.dataset_report
        (self.root/'test-use.json').write_text('{"status":"already_scored"}', encoding='utf-8')
        with self.assertRaisesRegex(FileExistsError, 'new independent test data'):
            self.run_package()
        self.assertEqual([c.args[3] for c in evaluate.call_args_list], ['val'])
        self.assertFalse((self.output/'test-reservation.json').exists())
        self.assertFalse((self.output/'package').exists())
        self.assertEqual(read_json(self.output/'final-report.json')['status'],
                         'heldout_previously_used_do_not_repeat')

    @patch('src.train_recyclables_final.evaluate_split')
    def test_failed_validation_does_not_claim_previously_used_test_is_fresh(self, evaluate):
        evaluate.return_value = dict(gate=dict(passed=False, reasons=['plastic']))
        (self.root/'test-use.json').write_text('{}', encoding='utf-8')
        result = self.run_package()
        self.assertTrue(result['test_previously_reserved'])
        self.assertFalse(result['test_evaluated'])
        self.assertEqual(evaluate.call_count, 1)


if __name__ == '__main__':
    unittest.main()
