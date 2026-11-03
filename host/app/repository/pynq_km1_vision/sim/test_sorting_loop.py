import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.sorting_demo import build_demo, demo_config, encode, run_demo
from src.sorting_loop import Controller, Journal, acceptance_fingerprint, readiness, validate_config, resolve_material_duplicates
from src.sorting_run import preflight


class SortingLoopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.controller, self.source, self.motion, self.journal = build_demo(self.root)

    def cycle(self):
        with self.journal:
            return self.controller.cycle()

    def mutate_frame(self, mutate, index=1):
        original = self.source.next
        def next_frame(timeout):
            row = original(timeout)
            if self.source.frame_id == index:
                mutate(row)
            return row
        self.source.next = next_frame

    def assert_rejected(self, reason):
        result = self.cycle()
        self.assertEqual(result['status'], 'rejected', result)
        self.assertIn(reason, result['reason'])
        self.assertEqual(self.motion.sent, [])
        self.assertIsNone(self.journal.pending())

    def test_three_classes_replan_and_empty_scene(self):
        result = run_demo(self.root / 'all')
        self.assertEqual(result['completed'], 3)
        self.assertEqual(set(result['labels']), {'metal', 'paper', 'plastic'})
        self.assertEqual(result['statuses'][-1], 'empty_scene')
        self.assertFalse(result['hardware_access'])

    def test_preview_never_opens_or_sends_motion(self):
        self.controller.mode = 'preview'
        result = self.cycle()
        self.assertEqual(result['status'], 'preview_only', result)
        self.assertEqual(len(result['plan']), 12)
        self.assertFalse(self.motion.opened)
        self.assertEqual(self.motion.sent, [])
        self.assertTrue(result['execution_blockers'])

    def test_partial_write_stops_and_survives_restart(self):
        self.motion.fail_step = 'descend'
        result = self.cycle()
        self.assertEqual(result['status'], 'needs_review', result)
        self.assertEqual(self.motion.sent[-1], 'descend')
        self.assertFalse(self.motion.opened)
        other, _, _, _ = build_demo(self.root)
        with self.assertRaisesRegex(ValueError, 'unresolved_previous_task'):
            other.cycle()

    def test_wrong_bin_never_completes(self):
        self.motion.wrong_bin = True
        result = self.cycle()
        self.assertEqual(result['status'], 'needs_review', result)
        self.assertIn('no_unique_new_target_in_correct_bin', result['reason'])
        self.assertIsNotNone(self.journal.pending())

    def test_review_is_bound_to_pending_id_and_never_marks_success(self):
        self.motion.fail_step = 'close'
        result = self.cycle()
        with self.journal:
            with self.assertRaisesRegex(ValueError, 'wrong_pending_task'):
                self.journal.resolve('wrong', 'Checked station')
            with self.assertRaisesRegex(ValueError, 'operator_observation_required'):
                self.journal.resolve(result['task_id'], '')
            self.journal.resolve(result['task_id'], 'Power off, object removed, restored checked idle pose')
        self.assertIsNone(self.journal.pending())
        review = json.loads((self.root / (result['task_id']+'.review.json')).read_text())
        self.assertEqual(review['outcome'], 'operator_resolved_not_automatic_success')

    def test_exclusive_station_lock(self):
        with self.journal:
            with self.assertRaises(FileExistsError):
                with Journal(self.root):
                    pass

    def test_pending_written_before_serial_open(self):
        original = self.motion.open
        def opened(plan):
            self.assertEqual(self.journal.pending()['status'], 'motion_intent_reserved')
            original(plan)
        self.motion.open = opened
        self.assertEqual(self.cycle()['status'], 'simulated_visual_pass')

    def test_intent_persisted_before_each_send(self):
        original = self.motion.send
        def send(data, ms, guard=None):
            pending = self.journal.pending()
            self.assertEqual(pending['steps'][-1]['status'], 'may_be_in_flight')
            original(data, ms, guard=guard)
        self.motion.send = send
        self.assertEqual(self.cycle()['status'], 'simulated_visual_pass')

    def test_immutable_images_and_observations_are_retained(self):
        result = self.cycle()
        self.assertEqual(result['status'], 'simulated_visual_pass', result)
        self.assertGreaterEqual(len(result['evidence']), 8)
        for entry in result['evidence']:
            self.assertTrue((self.root / entry['image']).is_file())
            self.assertTrue((self.root / entry['observation']).is_file())

    def test_low_confidence(self):
        self.mutate_frame(lambda r: r['observation']['instances'][0].update(confidence=.2))
        self.assert_rejected('uncertain_source_object')

    def test_changed_model_is_rejected(self):
        self.mutate_frame(lambda r: r['observation'].update(model_version='new'))
        self.assert_rejected('observation_identity_mismatch:model_version')

    def test_changed_prompt_runtime_or_camera_is_rejected(self):
        for key in ('runtime_id', 'prompt_mode', 'camera_id'):
            with self.subTest(key=key):
                original = self.source.next
                self.mutate_frame(lambda r: r['observation'].update({key:'changed'}), self.source.frame_id+1)
                self.assert_rejected('observation_identity_mismatch:'+key)
                self.source.next = original

    def test_mask_must_match_original_image_size_and_bbox(self):
        self.mutate_frame(lambda r: r['observation']['instances'][0].update(bbox_xyxy=[0,0,1,1]))
        self.assert_rejected('mask_bbox_mismatch')

    def test_image_binding(self):
        self.mutate_frame(lambda r: r['observation'].update(image_sha256='wrong'))
        self.assert_rejected('mask_image_binding_mismatch')

    def test_mutating_decoded_pixels(self):
        self.mutate_frame(lambda r: r['image'].__setitem__((119,1), 0))
        self.assert_rejected('decoded_image_changed')

    def test_stale_frame(self):
        self.mutate_frame(lambda r: r['observation'].update(capture_started_ms=0))
        self.assert_rejected('stale_or_future_frame')

    def test_future_frame(self):
        self.mutate_frame(lambda r: r['observation'].update(capture_started_ms=1e12))
        self.assert_rejected('stale_or_future_frame')

    def test_duplicate_frame(self):
        self.mutate_frame(lambda r: r['observation'].update(frame_id=1), 2)
        self.assert_rejected('repeated_or_old_frame')

    def test_stream_restart(self):
        self.mutate_frame(lambda r: r['observation'].update(source_stream_id='restart'), 2)
        self.assert_rejected('camera_stream_restarted')

    def test_unknown_material(self):
        self.mutate_frame(lambda r: r['observation']['instances'][0].update(label='can'))
        self.assert_rejected('invalid_material_instance')

    def test_scene_changes_before_motion(self):
        self.mutate_frame(lambda r: r['observation']['instances'].pop(), 4)
        self.assert_rejected('source_scene_changed')

    def test_recheck_after_usb_open_stops_without_first_write(self):
        self.mutate_frame(lambda r: r['observation']['instances'].pop(), 5)
        result = self.cycle()
        self.assertEqual(result['status'], 'needs_review', result)
        self.assertEqual(self.motion.sent, [])

    def test_scene_changes_before_descent(self):
        self.mutate_frame(lambda r: r['observation']['instances'].pop(), 7)
        result = self.cycle()
        self.assertEqual(result['status'], 'needs_review', result)
        self.assertNotIn('descend', self.motion.sent)

    def test_unverified_orientation(self):
        self.controller.c['profiles']['metal']['axis_deg'] = 0
        self.assert_rejected('unverified_grasp_orientation')

    def test_missing_calibration(self):
        self.controller.c['profiles']['metal']['image_to_grip_mm'] = None
        self.assert_rejected('missing_profile_calibration:metal')

    def test_silhouette_must_fit_measured_region(self):
        self.controller.c['profiles']['metal']['verified_polygon_px'] = [[0,0],[5,0],[5,5],[0,5]]
        self.assert_rejected('silhouette_outside_verified_region')

    def test_joint_limit_on_later_waypoint_blocks_whole_plan(self):
        self.controller.c['arm']['joint_limits_pwm'][5] = [1000, 1200]
        self.assert_rejected('Joint locked or PWM outside measured limits')

    def test_width_is_class_specific(self):
        self.controller.c['profiles']['metal']['max_width_mm'] = 2
        self.assert_rejected('jaw_width_outside_profile')

    def test_geometry_fault_does_not_open_serial(self):
        def geometry(mask):
            raise TimeoutError('PL stalled')
        self.controller.stats.geometry = geometry
        self.assert_rejected('PL stalled')

    def test_post_motion_inference_failure_latches_station(self):
        original = self.source.next
        def next_frame(timeout):
            if self.source.barrier > 0:
                raise TimeoutError('inference failed after retreat')
            return original(timeout)
        self.source.next = next_frame
        result = self.cycle()
        self.assertEqual(result['status'], 'needs_review')
        self.assertIsNotNone(self.journal.pending())

    def test_source_remaining_after_release_cannot_pass(self):
        original = self.motion.send
        def send(data, ms, guard=None):
            name = self.motion.names[0]
            saved = copy.deepcopy(self.source.items)
            original(data, ms, guard=guard)
            if name == 'release':
                self.source.items = saved
        self.motion.send = send
        result = self.cycle()
        self.assertEqual(result['status'], 'needs_review', result)
        self.assertIn('selected_object_still_in_source', result['reason'])

    def test_no_rgb_bin_change_cannot_pass(self):
        original = self.source.next
        saved = None
        def next_frame(timeout):
            nonlocal saved
            row = original(timeout)
            if self.source.barrier < 0:
                saved = row['image'].copy()
            else:
                # Bind a valid unchanged image to a semantic-only invented gain.
                import hashlib
                row['image'] = saved.copy()
                row['image'][119,0] = self.source.frame_id
                sha = hashlib.sha256(row['image'].tobytes()).hexdigest()
                row.update(jpeg_sha256=sha, bgr_sha256=sha)
                row['observation']['image_sha256'] = sha
            return row
        self.source.next = next_frame
        result = self.cycle()
        self.assertEqual(result['status'], 'needs_review', result)
        self.assertIn('no_semantic_and_rgb_bin_gain', result['reason'])

    def test_synthetic_data_cannot_be_used_in_physical_mode(self):
        with self.assertRaisesRegex(ValueError, 'synthetic_config_forbidden'):
            Controller(self.controller.c, self.source, self.controller.stats, self.motion,
                       self.journal, self.source.reference, mode='physical')

    def test_example_preflight_is_readonly_and_lists_missing_classes(self):
        path = Path(__file__).resolve().parents[1] / 'config/sorting_station.example.json'
        c = json.loads(path.read_text())
        result = preflight(c, path.parent)
        self.assertEqual(result['status'], 'not_ready_for_motion')
        for label in ('plastic','metal','paper'):
            self.assertIn(label+'.image_to_grip_mm', result['execution_blockers'])
        self.assertFalse(result['physical_motion'])

    def test_overlapping_source_and_bins_are_invalid(self):
        c = demo_config()
        c['profiles']['metal']['bin_roi'] = c['source_roi']
        with self.assertRaisesRegex(ValueError, 'overlapping_station_rois'):
            validate_config(c)

    def test_acceptance_is_invalidated_by_behavior_changes(self):
        c = demo_config()
        expected = acceptance_fingerprint(c)
        c['station_commissioned'] = True
        c['profiles']['metal']['commissioned'] = True
        self.assertEqual(expected, acceptance_fingerprint(c))
        for mutation in ('timing', 'threshold', 'calibration', 'bin'):
            changed = copy.deepcopy(c)
            if mutation == 'timing':
                changed['arm']['move_ms'] = 1000
            elif mutation == 'threshold':
                changed['min_scene_iou'] = .5
            elif mutation == 'calibration':
                changed['profiles']['metal']['image_to_grip_mm'][0][2] += 10
            else:
                changed['profiles']['paper']['bin_xyz_mm'][0] += 10
            self.assertNotEqual(expected, acceptance_fingerprint(changed))

    def test_acceptance_record_hash_and_config_are_checked(self):
        import hashlib
        c = demo_config()
        c['layout_reference'] = dict(path='not-present', sha256='x')
        accepted = dict(kind='sorting-profile-acceptance-v1', label='metal',
                        station_id=c['station_id'], automatic_trial_approved=True,
                        configuration_sha256=acceptance_fingerprint(c))
        path = self.root / 'acceptance.json'
        path.write_text(json.dumps(accepted))
        c['profiles']['metal']['acceptance_record'] = dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertFalse(any(v.startswith('metal.acceptance_record:') for v in preflight(c,self.root)['execution_blockers']))
        c['min_bin_gain_px'] += 1
        self.assertTrue(any('configuration_not_accepted' in v for v in preflight(c,self.root)['execution_blockers']))
        path.write_text('{}')
        self.assertTrue(any('hash_changed' in v for v in preflight(c,self.root)['execution_blockers']))

    def test_postretreat_stale_frame_blocks(self):
        def mutate(row):
            row['observation']['capture_started_ms'] -= 300
        self.mutate_frame(mutate, 8)
        result = self.cycle()
        self.assertEqual(result['status'], 'needs_review', result)
        self.assertIn('frame_precedes_retreat', result['reason'])

    def test_no_remaining_commands_after_keyboard_interrupt(self):
        original = self.motion.send
        def send(data, ms, guard=None):
            if self.motion.names[0] == 'close':
                raise KeyboardInterrupt()
            original(data, ms, guard=guard)
        self.motion.send = send
        result = self.cycle()
        self.assertEqual(result['status'], 'needs_review', result)
        self.assertNotIn('lift', self.motion.sent)
        self.assertIsNotNone(self.journal.pending())

    def test_bin_object_from_previous_cycle_cannot_be_reused_as_new_deposit(self):
        # First metal deposit is kept in the bin; second paper task is then
        # falsely reported as another metal mask at the old location.
        self.assertEqual(self.cycle()['status'], 'simulated_visual_pass')
        original = self.motion.send
        def send(data, ms, guard=None):
            name = self.motion.names[0]
            original(data, ms, guard=guard)
            if name == 'release':
                metal = next(i for i in self.source.items if i['label'] == 'metal')
                paper = next(i for i in self.source.items if i['label'] == 'paper')
                paper['mask'] = metal['mask'].copy()
        self.motion.send = send
        result = self.cycle()
        self.assertEqual(result['status'], 'needs_review', result)

    def test_slow_durable_log_cannot_send_with_expired_guard_frame(self):
        original = self.journal.save
        def save(name, value):
            original(name, value)
            if name == 'pending.json' and value.get('steps') and value['steps'][-1]['status'] == 'may_be_in_flight':
                self.controller.sleep(3)
        self.journal.save = save
        result = self.cycle()
        self.assertEqual(result['status'], 'needs_review', result)
        self.assertIn('motion_guard_frame_expired', result['reason'])
        self.assertEqual(self.motion.sent, [])


class MaterialDuplicateTests(unittest.TestCase):
    def setUp(self):
        self.mask=np.zeros((20,20),bool);self.mask[5:15,5:15]=True
        self.policy=dict(min_mask_iou=.9,min_score_margin=.2)

    def test_same_shape_clear_class_margin_selects_one_and_records_other(self):
        objects=[dict(label='paper',confidence=.82),dict(label='plastic',confidence=.4)]
        selected,reports=resolve_material_duplicates(objects,[self.mask,self.mask.copy()],self.policy)
        self.assertEqual(len(selected),1)
        self.assertEqual(selected[0][0]['label'],'paper')
        self.assertEqual(reports[0]['duplicate_candidates'][0]['label'],'plastic')

    def test_close_competing_classes_reject(self):
        with self.assertRaisesRegex(ValueError,'ambiguous_material_on_same_instance'):
            resolve_material_duplicates([dict(label='paper',confidence=.82),dict(label='plastic',confidence=.8)],
                                        [self.mask,self.mask.copy()],self.policy)

    def test_separate_low_confidence_object_is_preserved(self):
        other=np.roll(self.mask,9,axis=1)
        selected,reports=resolve_material_duplicates([dict(label='paper',confidence=.82),dict(label='plastic',confidence=.4)],
                                                      [self.mask,other],self.policy)
        self.assertEqual(len(selected),2)
        self.assertEqual(reports,[])

    def test_same_label_duplicate_does_not_create_second_object(self):
        selected,_=resolve_material_duplicates([dict(label='paper',confidence=.82),dict(label='paper',confidence=.8)],
                                               [self.mask,self.mask.copy()],self.policy)
        self.assertEqual(len(selected),1)


if __name__ == '__main__':
    unittest.main()
