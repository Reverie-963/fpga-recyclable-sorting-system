import threading
import time
import unittest
from importlib.util import find_spec
from src.monitor_operation import Operations

# The experimental native sorting foundation is deployed locally but is not
# fully integrated into main yet. Never count an absent foundation as a pass.
INTEGRATION_AVAILABLE=(find_spec('sim.test_sorting_unified') is not None and
                       find_spec('src.sorting_loop') is not None)


class DeferredPreflightTests(unittest.TestCase):
    def scenario(self, blockers=None):
        class Backend:
            def __init__(self):
                self.entered=threading.Event();self.release=threading.Event();self.calls=[]
            def check(self):
                self.entered.set();self.release.wait(3);return blockers or []
            def cycle(self,*args):
                self.calls.append(args);return dict(status='preview_only',task_id='test')
            def stop(self):pass
        backend=Backend();op=Operations(backend,defer_preflight=True)
        self.addCleanup(op.close);self.addCleanup(backend.release.set)
        state=op.command(dict(action='start',ready=True,owner='test-page',request_id='request-first',token=op.token))
        self.assertTrue(state['busy']);self.assertTrue(backend.entered.wait(1))
        return backend,op

    def test_slow_preflight_does_not_block_status_heartbeat_or_stop(self):
        backend,op=self.scenario()
        start=time.monotonic();self.assertTrue(op.status()['busy'])
        ack=op.command(dict(action='heartbeat',owner='test-page',operation_id=op.operation_id,token=op.token))
        self.assertTrue(ack['busy']);self.assertLess(time.monotonic()-start,.3)
        op.command(dict(action='stop',owner='test-page',token=op.token))
        backend.release.set();op.worker.join(1)
        self.assertFalse(backend.calls);self.assertEqual(op.state,'stopped')

    def test_fresh_preflight_rejection_never_enters_executor(self):
        backend,op=self.scenario(['actual_guard_failed'])
        backend.release.set();op.worker.join(1)
        self.assertFalse(backend.calls)
        self.assertIn('actual_guard_failed',op.result['reason'])

    def test_expired_page_cannot_execute_after_slow_preflight(self):
        backend,op=self.scenario()
        op.last_heartbeat=time.monotonic()-13
        backend.release.set();op.worker.join(1)
        self.assertFalse(backend.calls);self.assertEqual(op.reason,'page_disconnected')


@unittest.skipUnless(INTEGRATION_AVAILABLE,'requires the existing native sorting integration and fixtures')
class LiveObjectTests(unittest.TestCase):
    def fixture(self):
        from sim.test_sorting_unified import UnifiedBatchTests
        f=UnifiedBatchTests();f.setUp();self.addCleanup(f.doCleanups)
        f.source.items=[i for i in f.source.items if i['label']=='plastic']
        f.policy['live_object_admission']=dict(kind='operator-live-geometry-v1',
            operator_statement='Admit all new categories using live geometry.',physical_validation_completed=False)
        controller=f.factory(0);row=controller.acquire();target=(row['observation']['instances'][0],row['masks'][0])
        return f,controller,row,target

    def test_old_sample_area_and_size_do_not_reject_new_object(self):
        f,c,row,target=self.fixture()
        p=c.c['profiles']['plastic'];p.update(min_area_px=1,max_area_px=2,width_bounds_mm=[1,2],length_bounds_mm=[1,2])
        choice,plan=c.candidate(row,target,'new-object')
        self.assertGreater(len(plan),5);self.assertEqual(choice['specimen_admission']['mode'],'live_object')
        self.assertFalse(choice['specimen_admission']['sample_size_template_required'])
        self.assertFalse(choice['workspace_physically_qualified'])

    def test_gripper_opening_still_blocks_oversized_object(self):
        f,c,row,target=self.fixture();c.policy['jaw_open_span_mm']=2.01
        with self.assertRaisesRegex(ValueError,'exceeds_gripper'):c.candidate(row,target,'wide')

    def test_old_sample_area_does_not_reject_verified_landing(self):
        f,c,row,target=self.fixture()
        c.c['profiles']['plastic'].update(min_area_px=1,max_area_px=2)
        result=c.cycle()
        self.assertEqual(result['status'],'simulated_visual_pass',result.get('reason'))
        self.assertTrue(result['verification']['frames'])


    def test_round_new_object_uses_neutral_axis_instead_of_sample_aspect_ratio(self):
        import numpy as np
        f,c,row,target=self.fixture();old=target[1];ys,xs=np.nonzero(old);yy,xx=np.indices(old.shape)
        mask=(xx-int(xs.mean()))**2+(yy-int(ys.mean()))**2<=16
        row['masks'][0]=mask
        center_px=np.array([int(xs.mean()),int(ys.mean())]);center_mm=c.geometry.map('plastic',[center_px])[0]
        c.geometry.map=lambda label,points:(np.asarray(points)-center_px)*.5+center_mm
        choice,plan=c.candidate(row,(target[0],mask),'round-object')
        self.assertFalse(choice['specimen_admission']['principal_axis_used'])
        self.assertEqual(choice['axis_deg'],0)
        self.assertTrue(plan)

    def test_live_mode_requires_explicit_trial_statement(self):
        f,c,row,target=self.fixture();c.policy['live_object_admission']['operator_statement']=''
        with self.assertRaisesRegex(ValueError,'invalid_live_object'):c.candidate(row,target,'unapproved')




@unittest.skipUnless(INTEGRATION_AVAILABLE,'requires the existing native sorting integration and fixtures')
class SelectedTargetStabilityTests(unittest.TestCase):
    def test_new_background_label_count_does_not_reenable_whole_scene_matching(self):
        import numpy as np
        from src.sorting_prompted_motion import PromptedMotionController
        c=object.__new__(PromptedMotionController)
        c.motion_options={'full_scene_instance_stability_check_enabled':False}
        c.trial_labels=['Drink can','Other carton']
        c.c=dict(min_scene_iou=.85,image_size=[10,10],source_roi=[0,0,10,10],profiles={},max_scene_changed_fraction=.05)
        c.changed_pixels=lambda a,b:np.zeros((10,10),bool)
        mask=np.zeros((10,10),bool);mask[2:6,2:6]=True
        target=dict(label='metal',official_label='Drink can',zone='source',semantic_gate_passed=True)
        background=dict(label='paper',official_label='Other carton',zone='source',semantic_gate_passed=True)
        c.batch_context={'target':(target,mask)}
        before=dict(masks=[mask,~mask],observation={'instances':[target,background]})
        after=dict(masks=[mask],observation={'instances':[target]})
        c.unchanged(before,after)
        self.assertEqual(after['target_stability']['tracked_targets'],1)
        self.assertTrue(after['target_stability']['all_latest_masks_retained_for_obstacle_checks'])
        missing=dict(masks=[~mask],observation={'instances':[background]})
        with self.assertRaisesRegex(ValueError,'selected_target_missing'):c.unchanged(before,missing)


if __name__=='__main__':unittest.main()
