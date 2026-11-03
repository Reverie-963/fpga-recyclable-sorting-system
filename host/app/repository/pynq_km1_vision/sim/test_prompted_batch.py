"""Bounded batch orchestration, single final capture and failure latches."""
import unittest
from unittest.mock import patch
from src.prompted_batch import batch_settings,run_registered_batch,PromptedBatchController
from src.sorting_unified import UnifiedController
from sim import test_sorting_unified as unified_fixtures
from sim.test_bin_occupancy import OPTIONS


class PromptedBatchTests(unittest.TestCase):
    def fixture(self):
        f=unified_fixtures.UnifiedBatchTests();f.setUp();self.addCleanup(f.doCleanups)
        f.policy.update(batch_verification_mode='end_of_batch',skip_per_item_visual_verification=True,
            batch_final_operator_statement='explicit test request',placement_verification=OPTIONS)
        return f

    def run_batch(self,f,owner_check=lambda:None):
        with f.journal:
            return run_registered_batch(f.factory,f.journal,{'max_items':6,'max_seconds':1200},owner_check,f.clock.monotonic)

    def test_all_routes_then_exactly_one_final_frame_without_per_item_verdicts(self):
        f=self.fixture();calls=[];original=UnifiedController.verification_frames
        def capture(controller,*a,**kw):
            calls.append(len(f.motions));return original(controller,*a,**kw)
        with patch.object(UnifiedController,'verify_within_window',side_effect=AssertionError('per-item verification forbidden')),patch.object(UnifiedController,'verification_frames',capture):
            result=self.run_batch(f)
        self.assertEqual(result['status'],'completed_visual_evidence',result)
        self.assertEqual(result['motion_completed_count'],3)
        self.assertEqual(calls,[3])
        self.assertEqual(len(result['final_verification']['frames']),1)
        self.assertEqual(len(result['final_verification']['bins']),3)
        self.assertFalse(result['final_verification']['physical_success_verified'])
        self.assertTrue(result['steps']);self.assertEqual(len(result['steps']),len(result['plan']))
        self.assertFalse((f.journal.root/'batch-pending.json').exists())

    def test_serial_failure_stops_batch_without_final_check_or_retry(self):
        f=self.fixture();f.fail_step='close'
        with patch.object(UnifiedController,'verification_frames',side_effect=AssertionError('no verdict after uncertain motion')):
            result=self.run_batch(f)
        self.assertEqual(result['status'],'needs_review',result)
        self.assertEqual(f.factory_calls,1)
        self.assertIsNotNone(f.journal.pending())
        self.assertTrue((f.journal.root/'batch-pending.json').exists())

    def test_stop_between_items_keeps_partial_batch_and_does_not_verify(self):
        f=self.fixture()
        def check():
            if any(m.sent for m in f.motions):raise ValueError('operator_stop')
        with patch.object(UnifiedController,'verification_frames',side_effect=AssertionError('no final after stop')):
            result=self.run_batch(f,check)
        self.assertEqual(result['status'],'needs_review',result)
        self.assertEqual(f.factory_calls,1)
        self.assertEqual(result['motion_completed_count'],1)
        self.assertTrue((f.journal.root/'batch-pending.json').exists())

    def test_empty_final_bin_is_not_success_and_keeps_batch_pending(self):
        f=self.fixture()
        with patch.object(UnifiedController,'verify_bin_occupied',side_effect=ValueError('unified_target_bin_empty')):
            result=self.run_batch(f)
        self.assertEqual(result['status'],'needs_review',result)
        self.assertEqual(result['completed_count'],0)
        self.assertEqual(result['motion_completed_count'],3)
        self.assertTrue((f.journal.root/'batch-pending.json').exists())

    def test_initial_unplannable_target_rejects_before_action(self):
        from types import SimpleNamespace
        controller=object.__new__(PromptedBatchController);controller.tracker=SimpleNamespace(initialized=False)
        with patch('src.sorting_prompted_motion.PromptedMotionController.select',return_value=('selected',[{'reason':'unreachable'}])):
            with self.assertRaisesRegex(ValueError,'initial_route_rejected'):controller.select({})

    def test_all_planning_precedes_first_write_and_no_mid_batch_reselection(self):
        f=self.fixture();events=[];select=UnifiedController.select;candidate=UnifiedController.candidate
        def choose(controller,row):
            events.append(('select',sum(bool(m.sent) for m in f.motions)))
            return select(controller,row)
        def plan(controller,*args):
            events.append(('plan',sum(bool(m.sent) for m in f.motions)))
            return candidate(controller,*args)
        with patch.object(UnifiedController,'select',choose),patch.object(UnifiedController,'candidate',plan):
            result=self.run_batch(f)
        self.assertEqual(result['status'],'completed_visual_evidence',result)
        self.assertEqual(events,[('select',0),('plan',0),('plan',0),('plan',0)])
        self.assertEqual(result['planned_count'],3)
        self.assertTrue(result['planning_completed_before_motion'])
        self.assertEqual([p['choice'] for p in result['frozen_plan']],[p['choice'] for p in result['items']])

    def test_last_target_planning_failure_prevents_every_write(self):
        f=self.fixture();original=UnifiedController.candidate;calls=[]
        def candidate(controller,*args):
            calls.append(1)
            if len(calls)==3:raise ValueError('last_target_unreachable')
            return original(controller,*args)
        with patch.object(UnifiedController,'candidate',candidate):result=self.run_batch(f)
        self.assertEqual(result['status'],'rejected',result)
        self.assertEqual(result['motion_completed_count'],0)
        self.assertFalse(any(m.sent for m in f.motions))
        self.assertIn('last_target_unreachable',result['reason'])

    def test_scene_larger_than_limit_rejects_whole_batch_before_motion(self):
        f=self.fixture()
        with f.journal:
            result=run_registered_batch(f.factory,f.journal,{'max_items':2,'max_seconds':1200},lambda:None,f.clock.monotonic)
        self.assertEqual(result['status'],'rejected',result)
        self.assertIn('initial_scene_exceeds_limit',result['reason'])
        self.assertFalse(any(m.sent for m in f.motions))

    def test_same_class_slot_conflict_rejects_before_first_grasp(self):
        import copy
        import numpy as np
        f=self.fixture();f.source.items=[i for i in f.source.items if i['label']=='paper']
        second=copy.deepcopy(f.source.items[0]);second['mask']=np.roll(second['mask'],-24,axis=1)
        f.source.items.append(second)
        result=self.run_batch(f)
        self.assertEqual(result['status'],'rejected',result)
        self.assertFalse(any(m.sent for m in f.motions))
        self.assertIn('no_free_verified_bin_slot',result['reason'])

    def physical_continuations(self,f,wrong_stream=False,jpeg=False):
        from types import SimpleNamespace
        factory=f.factory;events=[]
        def build(index):
            controller=factory(index)
            if index:
                controller.mode='known_physical'
                controller.source=SimpleNamespace(source=SimpleNamespace(c=controller.c,detector=SimpleNamespace(
                    verify_identity=lambda:events.append('identity'))),after=f.source.after)
                def raw(timeout):
                    events.append('raw')
                    row=f.source.next(timeout);row['source_encoding']='jpeg' if jpeg else 'png'
                    row['observation'].pop('instances')
                    if wrong_stream:row['observation']['source_stream_id']='other-camera'
                    return row
                controller.source.source.raw=raw
                def marker_only(timeout):raise AssertionError('JPEG motion camera cannot provide lossless admission')
                controller.motion.camera=SimpleNamespace(after=f.source.after,raw=marker_only)
                controller.motion.last_receipt=dict(echo_matched=True,completed_movement_wait=True)
                # The final verdict still captures fresh semantic evidence.
                controller.verification_frames=lambda *a,**kw:UnifiedController.verification_frames(
                    controller,*a,**kw)
                controller.source.next=f.source.next
            return controller
        f.factory=build
        return events

    def test_explicitly_disabled_occupancy_can_share_existing_fixed_destination(self):
        import copy
        import numpy as np
        f=self.fixture();f.source.items=[i for i in f.source.items if i['label']=='paper']
        second=copy.deepcopy(f.source.items[0]);second['mask']=np.roll(second['mask'],-24,axis=1)
        f.source.items.append(second)
        f.source.items.sort(key=lambda i:int(np.nonzero(i['mask'])[1].min()))
        original=UnifiedController.candidate
        def candidate(controller,*args):
            choice,plan=original(controller,*args)
            choice['destination_empty_check_enabled']=False
            return choice,plan
        with patch.object(UnifiedController,'candidate',candidate):result=self.run_batch(f)
        self.assertEqual(result['status'],'completed_visual_evidence',result)
        self.assertEqual(result['planned_count'],2)
        self.assertEqual(result['motion_completed_count'],2)
        self.assertEqual(len(result['final_verification']['bins']),1)

    def test_physical_continuation_checks_raw_identity_without_semantic_replanning(self):
        f=self.fixture();events=self.physical_continuations(f)
        result=self.run_batch(f)
        self.assertEqual(result['status'],'completed_visual_evidence',result)
        self.assertEqual(events,['identity','raw','identity','identity','raw','identity'])

    def test_camera_change_between_frozen_routes_stops_without_second_write_or_final_verdict(self):
        f=self.fixture();self.physical_continuations(f,wrong_stream=True)
        with patch.object(UnifiedController,'verification_frames',side_effect=AssertionError('no final after camera change')):
            result=self.run_batch(f)
        self.assertEqual(result['status'],'needs_review',result)
        self.assertEqual(result['motion_completed_count'],1)
        self.assertFalse(f.motions[1].sent)
        self.assertIn('motion_camera_stream_changed',result['reason'])
        self.assertIsNotNone(f.journal.pending())

    def test_jpeg_continuation_frame_still_rejects_before_second_route(self):
        f=self.fixture();self.physical_continuations(f,jpeg=True)
        result=self.run_batch(f)
        self.assertEqual(result['status'],'needs_review')
        self.assertIn('requires_lossless_frame',result['reason']);self.assertFalse(f.motions[1].sent)

    def test_batch_is_explicit_bounded_and_disabled_legacy_stays_available(self):
        self.assertIsNone(batch_settings({}))
        good=dict(automatic_batch_enabled=True,batch_final_operator_statement='one bounded initial scene')
        self.assertEqual(batch_settings(good)['max_items'],6)
        for update in ({'batch_max_items':7},{'batch_max_seconds':1201},{'batch_final_operator_statement':''}):
            with self.assertRaises(ValueError):batch_settings(dict(good,**update))

    def test_shared_destination_is_checked_once_and_stale_final_frame_is_rejected(self):
        from types import SimpleNamespace
        from src.sorting_batch_verification import verify_batch_final
        f=self.fixture();calls=[]
        row={'observation':{'source_frame_id':'final','capture_started_ms':1000}}
        context=dict(track_id=1,before={},target=({'label':'plastic'},None),choice={'bin_id':'bin_plastic'})
        def capture(barrier,rows):calls.append('capture');rows.append(row)
        def verify(*args):calls.append('bin');return {'status':'bin_occupied_pass'}
        controller=SimpleNamespace(batch_context=context,placement_verification=OPTIONS,
            policy={'batch_final_operator_statement':'explicit','skip_per_item_visual_verification':True,'verification_max_age_ms':5000},
            verification_frames=capture,verify_bin_occupied=verify,clock=lambda:1)
        second=SimpleNamespace(**vars(controller));second.batch_context=dict(context,track_id=2)
        journal=SimpleNamespace(capture=lambda *a:{},save=f.journal.save)
        result=verify_batch_final([controller,second],journal,'shared-bin')
        self.assertEqual(result['status'],'batch_visual_evidence_pass')
        self.assertEqual(calls,['capture','bin'])
        row['observation']['capture_started_ms']=-10000
        result=verify_batch_final([controller],journal,'stale-final')
        self.assertEqual(result['status'],'batch_final_verification_failed')


if __name__=='__main__':unittest.main()
