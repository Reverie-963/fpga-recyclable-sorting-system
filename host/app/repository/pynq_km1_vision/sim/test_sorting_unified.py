"""All devices and commissioning records in these tests are synthetic."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from sim.test_sorting_known import config
from src.sorting_demo import DemoClock,DemoSource,DemoMotion
from src.sorting_instances import associate,InstanceLedger
from src.sorting_io import Statistics
from src.sorting_loop import Journal,mapped
from src.sorting_scene import check_motion_layout
from src.sorting_unified import UnifiedController
from src.sorting_unified_plan import compile_route,validate_policy,slot_rect
from src.sorting_unified_run import BatchLease,run_batch,target_region_runtime,configure_target_region,clear_batch_pending


def setup_policy(c):
    models={}
    for label,p in c['profiles'].items():
        paper=label=='paper'
        models[label]=dict(kind='unified-position-model-v1',object_profile_id=p['object_profile_id'],
            object_height_mm=p['object_height_mm'],grasp_z_mm=p['grasp_z_mm'],
            provenance='synthetic_test_not_physical_calibration',base_center_pwm=1500,
            base_pwm_per_degree=7.66,wrist_pwm_per_degree=600/90,radial_bias_mm=10 if paper else 0,
            xy_mode='constant_grip' if paper else 'two_height_maps',source_chains=(
                [[[130,-45],[90,-60],[70,-65],[50,-65]]] if paper else
                [[[130,-45],[70,-75],[40,-75],[10,-75]],[[130,-45],[90,-60],[40,-65],[10,-65]]]))
    return dict(kind='supervised-unified-sorting-v1',operator_authorized=True,operator_statement='synthetic test',
        workspace_physically_qualified=False,unattended_accepted=False,max_objects=6,
        source_card_polygon_mm=[[-110,-60],[110,-60],[110,100],[-110,100]],
        pick_center_card_polygon_mm=[[-90,-30],[90,-30],[90,60],[-90,60]],
        jaw_open_span_mm=20,jaw_depth_mm=5,obstacle_margin_mm=1,bin_padding_px=1,models=models,
        bin_slots={label:[dict(slot_id=label+'-original',origin='legacy_qualified_profile',roi_fraction=[0,0,1,1])]
                   for label in c['profiles']})


def test_config():
    c=config();c['max_cycle_s']=200
    for label,p in c['profiles'].items():
        p['object_height_mm']={'paper':65,'metal':56,'plastic':53}[label]
        p['image_to_card_mm']=[[1,0,-43],[0,-.5,37.5],[0,0,1]]
    c['profiles']['paper'].update(grasp_z_mm=50,source_mid_z_mm=60,hover_z_mm=70)
    return c


class AssociationTests(unittest.TestCase):
    def masks(self):
        a=np.zeros((30,40),bool);b=a.copy();a[3:9,3:9]=1;b[15:21,25:31]=1
        return [({'label':'paper'},a),({'label':'paper'},b)]
    def test_same_class_order_reversal_does_not_swap_identity(self):
        old=self.masks();self.assertEqual(associate(old,list(reversed(old)),.95),[1,0])
    def test_duplicate_masks_are_ambiguous(self):
        a=self.masks()[0]
        with self.assertRaisesRegex(ValueError,'ambiguous'):associate([a,a],[a,a],.95)
    def test_missing_instance_or_merge_is_not_empty_completion(self):
        old=self.masks()
        with self.assertRaises(ValueError):associate(old,old[:1],.95)
    def test_tracker_retires_only_selected_and_rejects_late_missing_item(self):
        old=self.masks();tracker=InstanceLedger(.95);tracker.observe(old)
        tracker.retire(1,old[1:]);self.assertEqual(tracker.live[0][0],2)
        with self.assertRaises(ValueError):tracker.observe([])
    def test_existing_bin_matches_allow_exactly_separate_new_instance(self):
        old=self.masks();self.assertEqual(associate(old[:1],list(reversed(old)),.95,allow_extra=True),[1])


class UnifiedPlanTests(unittest.TestCase):
    def setUp(self):self.c=test_config();self.policy=setup_policy(self.c)
    def test_three_class_full_angular_sweep(self):
        for label in self.c['profiles']:
            for angle in range(0,360,15):
                r=compile_route(self.c,label,[0,5],angle,self.policy,self.policy['bin_slots'][label][0])
                self.assertEqual(r['object_height_mm'],self.c['profiles'][label]['object_height_mm'])
                self.assertEqual(r['nominal_interpolation'][-1]['target_pwm'],self.c['known']['initial_pwm'])
                self.assertFalse(r['collision_free_verified'])
    def test_angle_symmetry_and_position_change(self):
        slot=self.policy['bin_slots']['paper'][0]
        a=compile_route(self.c,'paper',[-20,5],0,self.policy,slot)
        b=compile_route(self.c,'paper',[-20,5],180,self.policy,slot)
        c=compile_route(self.c,'paper',[20,5],0,self.policy,slot)
        self.assertEqual(a['wrist_pwm'],b['wrist_pwm']);self.assertNotEqual(a['base_pwm'],c['base_pwm'])
    def test_profile_height_cannot_be_borrowed_from_another_class(self):
        p=copy.deepcopy(self.policy);p['models']['metal']['object_height_mm']=65
        with self.assertRaises(ValueError):validate_policy(p,self.c)
    def test_outside_workspace_and_unreachable_target_rejected(self):
        slot=self.policy['bin_slots']['paper'][0]
        for point in ([150,5],[0,-200],[float('nan'),0]):
            with self.assertRaises(ValueError):compile_route(self.c,'paper',point,90,self.policy,slot)
    def test_original_bin_slot_cannot_be_changed_without_new_evidence(self):
        p=copy.deepcopy(self.policy);p['bin_slots']['paper'][0]['transfer_pwm']=[[900,1500,1500,1500]]
        with self.assertRaises(ValueError):validate_policy(p,self.c)
    def test_overlapping_slots_rejected(self):
        p=copy.deepcopy(self.policy);p['bin_slots']['paper'].append(dict(p['bin_slots']['paper'][0],slot_id='copy'))
        with self.assertRaises(ValueError):validate_policy(p,self.c)


class TargetRegionTests(unittest.TestCase):
    def setUp(self):
        self.c=test_config()
        regions=dict(source=self.c['source_roi'],static_a=self.c['layout']['static_rois'][0],
                     static_b=self.c['layout']['static_rois'][1])
        regions.update({p['bin_id']:p['bin_roi'] for p in self.c['profiles'].values()})
        self.c['statistics'].update(backend='pl_board_local',url='http://127.0.0.1:8780',regions=regions,
            image_size=self.c['image_size'],threshold=45,max_gap_s=.6,window_frames=3)
        self.geometry=SimpleNamespace(runtime=self.c,signature='f'*64)

    def test_rebind_changes_only_source_region_and_keeps_all_release_routes(self):
        rect=[10,35,30,80];runtime=target_region_runtime(self.geometry,rect)
        self.assertEqual(runtime['source_roi'],rect)
        self.assertEqual(runtime['profiles'],self.c['profiles'])
        for name,r in self.c['statistics']['regions'].items():
            if name!='source':self.assertEqual(runtime['statistics']['regions'][name],r)
        self.assertNotEqual(runtime['source_roi'],self.c['source_roi'])

    def test_target_rgb_region_cannot_include_any_box_or_static_patch(self):
        for rect in ([90,25,130,50],[0,0,10,10],[-1,35,20,60]):
            with self.assertRaises(ValueError):target_region_runtime(self.geometry,rect)

    def test_rebind_replaces_all_camera_statistics_before_later_baseline(self):
        old=SimpleNamespace();new=SimpleNamespace();owner=SimpleNamespace(check=lambda:None)
        controller=SimpleNamespace(geometry=self.geometry,c=self.c,stats=old,policy=setup_policy(self.c),
            source=SimpleNamespace(source=SimpleNamespace(c=self.c,statistics=old)),
            motion=SimpleNamespace(c=self.c,camera=SimpleNamespace(c=self.c,statistics=old)))
        with (patch('src.sorting_unified_run.activate_regions',return_value={'baseline_invalidated':True}) as activate,
              patch('src.sorting_unified_run.StationStatistics',return_value=new)):
            result=configure_target_region(controller,dict(source_evidence_roi=[10,35,30,80]),owner)
        self.assertTrue(result['baseline_invalidated']);activate.assert_called_once_with(old,self.geometry,owner)
        self.assertIs(controller.stats,new)
        self.assertIs(controller.source.source.statistics,new)
        self.assertIs(controller.motion.camera.statistics,new)
        self.assertEqual(controller.geometry.runtime['profiles'],self.c['profiles'])

    def test_target_crop_keeps_batch_pickup_region_when_a_neighbour_mask_crosses_crop_edge(self):
        old=SimpleNamespace();new=SimpleNamespace();owner=SimpleNamespace(check=lambda:None)
        full=copy.deepcopy(self.c['source_roi']);config=copy.deepcopy(self.c);config['pickup_source_roi']=full
        controller=SimpleNamespace(geometry=self.geometry,c=config,stats=old,policy=setup_policy(self.c),
            source=SimpleNamespace(source=SimpleNamespace(c=config,statistics=old)),
            motion=SimpleNamespace(c=config,camera=SimpleNamespace(c=config,statistics=old)))
        with patch('src.sorting_unified_run.activate_regions',return_value={}),patch('src.sorting_unified_run.StationStatistics',return_value=new):
            configure_target_region(controller,dict(source_evidence_roi=[10,35,30,80]),owner)
        self.assertEqual(controller.c['pickup_source_roi'],full)
        self.assertEqual(self.geometry.runtime['pickup_source_roi'],full)
        scope=UnifiedController.__new__(UnifiedController);scope.c=controller.c;scope.policy={'detection_window_px':full}
        scope.geometry=SimpleNamespace(original=self.c)
        a=np.zeros((self.c['image_size'][1],self.c['image_size'][0]),bool);a[40:60,12:22]=True
        for left in (31,29):
            b=np.zeros_like(a);b[40:60,left:left+8]=True
            objects=[dict(label='metal',bbox_xyxy=[12,40,22,60]),dict(label='plastic',bbox_xyxy=[left,40,left+8,60])]
            row=dict(image=np.zeros((*a.shape,3),np.uint8),observation={'instances':objects},masks=[a,b])
            self.assertEqual(len(scope.source_objects(row)),2)
        self.assertIs(controller.source.source.statistics,new)
        self.assertIs(controller.motion.camera.statistics,new)
        self.assertEqual(controller.geometry.runtime['profiles'],self.c['profiles'])


class UnifiedBatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.c=test_config();self.policy=setup_policy(self.c);self.clock=DemoClock();self.source=DemoSource(self.c,self.clock)
        get=self.source.next
        def row(timeout):
            r=get(timeout);r['observation']['source_frame_id']=str(r['observation']['frame_id']);return r
        self.source.next=row
        layout=dict(self.c['layout'],image_size=self.c['image_size'])
        self.geometry=SimpleNamespace(runtime=self.c,reference=self.source.reference,signature='f'*64,
            map=lambda label,points:mapped(self.c['profiles'][label]['image_to_card_mm'],points),
            check=lambda image:check_motion_layout(self.source.reference,image,layout))
        self.tracker=InstanceLedger(.95);self.motions=[];self.factory_calls=0;self.fail_step=None;self.motion_class=DemoMotion
        self.journal=Journal(self.root/'ledger')
    def factory(self,index):
        self.factory_calls+=1;self.source.after(self.clock.monotonic()*1000)
        motion=self.motion_class(self.source,self.clock);motion.fail_step=self.fail_step;self.motions.append(motion)
        return UnifiedController(self.c,self.source,Statistics(),motion,self.journal,self.geometry,self.source.reference,
            self.root,policy=self.policy,tracker=self.tracker,mode='simulation',clock=self.clock.monotonic,sleep=self.clock.sleep)
    def run_batch(self,limit=6):
        with self.journal:return run_batch(self.factory,self.journal,limit,1200,clock=self.clock.monotonic)
    def test_batch_cleanup_retries_sharing_conflict_without_repeating_controller(self):
        self.source.items=[]
        original=Path.unlink;calls=[]
        def sharing(path,*args,**kwargs):
            if path.name=='batch-pending.json':
                calls.append(path)
                if len(calls)<3:
                    error=PermissionError('temporary sharing conflict');error.winerror=32;raise error
            return original(path,*args,**kwargs)
        with patch.object(Path,'unlink',sharing),patch('src.sorting_unified_run.time.sleep') as sleep:
            result=self.run_batch()
        self.assertEqual(result['status'],'empty_scene');self.assertEqual(self.factory_calls,1)
        self.assertEqual(len(calls),3);self.assertEqual(sleep.call_count,2)
        self.assertFalse((self.journal.root/'batch-pending.json').exists())
        self.assertFalse(any(m.sent for m in self.motions))
    def test_batch_cleanup_persistent_or_nonsharing_error_stays_latched(self):
        for code,count in [(32,10),(5,1)]:
            self.journal.save('batch-pending.json',{'batch_id':'do-not-clear'})
            error=PermissionError('blocked');error.winerror=code
            with patch.object(Path,'unlink',side_effect=error) as unlink,patch('src.sorting_unified_run.time.sleep'):
                with self.assertRaises(PermissionError):clear_batch_pending(self.journal)
            self.assertEqual(unlink.call_count,count)
            self.assertTrue((self.journal.root/'batch-pending.json').exists())
    def test_three_materials_processed_one_at_a_time_and_reobserved(self):
        result=self.run_batch()
        self.assertEqual(result['status'],'completed',result)
        self.assertEqual(result['completed_count'],3);self.assertEqual(result['remaining'],[])
        self.assertEqual(self.factory_calls,3);self.assertIsNone(self.journal.pending())
        self.assertTrue(all(i['status']=='simulated_visual_pass' for i in result['items']))

    def test_end_batch_continues_after_visual_failure_then_checks_all_items(self):
        self.policy['batch_verification_mode']='end_of_batch'
        factory=self.factory
        def local_failure(index):
            controller=factory(index)
            if index==0:
                controller.verify_within_window=lambda *args:(_ for _ in ()).throw(ValueError('unified_wrong_bin_change'))
            return controller
        self.factory=local_failure
        result=self.run_batch()
        self.assertEqual(result['status'],'completed',result)
        self.assertEqual(result['motion_completed_count'],3)
        self.assertEqual(result['completed_count'],3)
        self.assertEqual(len(result['per_item_visual_failures']),1)
        self.assertEqual(result['final_verification']['status'],'batch_visual_evidence_pass')
        self.assertEqual(len(result['final_verification']['frames']),3)
        first=result['items'][0]['task_id']
        import json
        record=json.loads((self.journal.root/(first+'.json')).read_text())
        self.assertEqual(record['verification']['status'],'per_item_visual_failed')
        self.assertEqual(record['status'],'motion_completed_awaiting_batch_verification')
        self.assertIsNone(self.journal.pending())
        self.assertFalse((self.journal.root/'batch-pending.json').exists())

    def test_end_batch_still_stops_on_uncertain_serial_write(self):
        self.policy['batch_verification_mode']='end_of_batch';self.fail_step='close'
        result=self.run_batch()
        self.assertEqual(result['status'],'needs_review');self.assertEqual(self.factory_calls,1)
        self.assertEqual(result['motion_completed_count'],0);self.assertIsNotNone(self.journal.pending())

    def test_skip_item_visual_checks_still_reobserves_and_final_checks_three_items(self):
        self.policy.update(batch_verification_mode='end_of_batch',skip_per_item_visual_verification=True)
        factory=self.factory
        def no_local_check(index):
            controller=factory(index)
            controller.verify_within_window=lambda *a:(_ for _ in ()).throw(AssertionError('local check must not run'))
            return controller
        self.factory=no_local_check
        result=self.run_batch()
        self.assertEqual(result['status'],'completed',result)
        self.assertEqual(result['motion_completed_count'],3)
        self.assertEqual(result['completed_count'],3)
        self.assertEqual(self.factory_calls,3)
        self.assertEqual(len(result['final_verification']['frames']),3)
        self.assertEqual(result['per_item_visual_failures'],[])
        for item in result['items']:
            import json
            r=json.loads((self.journal.root/(item['task_id']+'.json')).read_text())
            self.assertEqual(r['verification']['status'],'deferred_to_batch_end')
            self.assertEqual(r['verification']['frames'],[])

    def test_skip_local_visual_check_never_skips_next_scene_or_serial_fault(self):
        self.policy.update(batch_verification_mode='end_of_batch',skip_per_item_visual_verification=True)
        factory=self.factory
        def bad_second(index):
            controller=factory(index)
            if index==1:controller.observe=lambda:(_ for _ in ()).throw(TimeoutError('new scene timeout'))
            return controller
        self.factory=bad_second
        result=self.run_batch()
        self.assertEqual(result['status'],'blocked',result)
        self.assertEqual(result['motion_completed_count'],1)
        self.assertEqual(len(self.motions[1].sent),0)
        self.assertIsNone(self.journal.pending())

    def test_skip_local_visual_check_preserves_final_failure(self):
        self.policy.update(batch_verification_mode='end_of_batch',skip_per_item_visual_verification=True)
        factory=self.factory
        def wrong_last(index):
            controller=factory(index)
            if index==2:controller.motion.wrong_bin=True
            return controller
        self.factory=wrong_last
        result=self.run_batch()
        self.assertEqual(result['motion_completed_count'],3,result)
        self.assertEqual(result['status'],'needs_review',result)
        self.assertEqual(result['completed_count'],0)

    def test_skip_local_visual_check_still_latches_a_motion_failure(self):
        self.policy.update(batch_verification_mode='end_of_batch',skip_per_item_visual_verification=True)
        self.fail_step='close';result=self.run_batch()
        self.assertEqual(result['status'],'needs_review')
        self.assertEqual(self.factory_calls,1)
        self.assertEqual(result['motion_completed_count'],0)
        self.assertIsNotNone(self.journal.pending())

    def test_skip_item_check_requires_batch_final_and_half_waits_require_record(self):
        from src.sorting_unified_plan import validate_policy
        p=copy.deepcopy(self.policy);p['skip_per_item_visual_verification']=True
        with self.assertRaisesRegex(ValueError,'requires_batch_final'):validate_policy(p,self.c)
        p=copy.deepcopy(self.policy);p['wait_time_scale']=.5
        with self.assertRaisesRegex(ValueError,'operator_statement'):validate_policy(p,self.c)
        p['wait_operator_statement']='采图、通信超时上限也减半'
        validate_policy(p,self.c)

    def test_end_batch_does_not_regrasp_target_left_at_source(self):
        self.policy['batch_verification_mode']='end_of_batch';factory=self.factory
        def missed(index):
            controller=factory(index)
            if index==0:
                send=controller.motion.send
                def no_release(data,duration,guard=None):
                    if controller.motion.names[0]=='release':
                        name=controller.motion.names.pop(0);controller.motion.sent.append(name)
                    else:send(data,duration,guard)
                controller.motion.send=no_release
                controller.verify_within_window=lambda *args:(_ for _ in ()).throw(ValueError('unified_selected_not_removed'))
            return controller
        self.factory=missed
        result=self.run_batch()
        # Tracking may reject rather than guess, but never repeats the first pick.
        self.assertEqual(sum('close' in m.sent for m in self.motions),len(self.tracker.attempted))
        ids=[i['choice']['track_id'] for i in result['items'] if i.get('choice')]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(result['status'],'needs_review',result)

    def test_deferred_tracker_does_not_accept_unexplained_new_object(self):
        pairs=AssociationTests().masks();tracker=InstanceLedger(.95)
        tracker.observe(pairs);tracker.defer(1)
        extra=np.zeros_like(pairs[0][1]);extra[2:6,30:35]=True
        with self.assertRaisesRegex(ValueError,'unexplained_instance'):
            tracker.observe([pairs[1],({'label':'paper'},extra)])

    def test_end_batch_wrong_final_class_is_not_rewritten_as_success(self):
        self.policy['batch_verification_mode']='end_of_batch';factory=self.factory
        def wrong_last(index):
            controller=factory(index)
            if index==2:controller.motion.wrong_bin=True
            return controller
        self.factory=wrong_last
        result=self.run_batch()
        self.assertEqual(result['motion_completed_count'],3,result)
        self.assertEqual(result['status'],'needs_review',result)
        self.assertEqual(result['completed_count'],0)
        self.assertTrue((self.journal.root/'batch-pending.json').exists())

    def test_relaxed_unified_thresholds_do_not_modify_original_motion_guards(self):
        self.policy['decision_thresholds']={'min_confidence':.4,'min_scene_iou':.9,
            'max_scene_changed_fraction':.05,'max_other_bin_changed_fraction':.05}
        self.policy['threshold_operator_statement']='captain requests 40 percent'
        controller=self.factory(0)
        self.assertEqual(controller.c['min_confidence'],.4)
        self.assertNotEqual(self.c['min_confidence'],.4)
        self.assertEqual(controller.c['layout'],self.c['layout'])
        self.assertEqual(controller.c['arm'],self.c['arm'])
        self.assertEqual(controller.c['max_age_ms'],self.c['max_age_ms'])

    def test_twenty_percent_empty_box_threshold_is_separate_from_scene_checks(self):
        self.policy['empty_bin_max_changed_fraction']=.2
        controller=self.factory(0);row=controller.observe()[-1]
        p=controller.c['profiles']['metal'];x0,y0,x1,y1=p['bin_roi']
        # Change ten percent of the empty slot's pixels; no semantic object.
        row['image'][y0:y0+3,x0:x1]=255
        slots=controller.free_slots(row,'metal')
        self.assertTrue(slots);self.assertEqual(controller.slot_checks[-1]['max_changed_fraction'],.2)
        self.assertLess(controller.c['max_scene_changed_fraction'],.2)
        # An actual detected object still occupies the slot regardless of RGB allowance.
        mask=np.zeros(row['image'].shape[:2],bool);mask[y0:y0+3,x0:x0+3]=True
        row['masks'].append(mask);row['observation']['instances'].append({'label':'metal'})
        self.assertEqual(controller.free_slots(row,'metal'),[])
        self.assertEqual(controller.slot_checks[-1]['reason'],'slot_object_detected')

    def test_forty_percent_is_used_for_candidate_and_final_confidence(self):
        self.policy.update(batch_verification_mode='end_of_batch',
            decision_thresholds={'min_confidence':.4},threshold_operator_statement='captain 40 percent')
        get=self.source.next
        def lower_confidence(timeout):
            row=get(timeout)
            for obj in row['observation']['instances']:obj['confidence']=.45
            return row
        self.source.next=lower_confidence
        result=self.run_batch()
        self.assertEqual(result['status'],'completed',result)
        self.assertTrue(all(placed['confidence']==.45 for frame in result['final_verification']['frames']
                            for placed in frame['placements']))

    def test_deferred_visual_policy_does_not_continue_after_camera_layout_error(self):
        self.policy['batch_verification_mode']='end_of_batch';factory=self.factory
        def layout_error(index):
            controller=factory(index)
            controller.verify_within_window=lambda *args:(_ for _ in ()).throw(ValueError('camera_not_stable'))
            return controller
        self.factory=layout_error
        result=self.run_batch()
        self.assertEqual(result['status'],'needs_review');self.assertEqual(self.factory_calls,1)
        self.assertIsNotNone(self.journal.pending())
    def test_detector_order_can_reverse_between_frames(self):
        get=self.source.next
        def reversed_row(timeout):
            r=get(timeout)
            if r['observation']['frame_id']%2:r['observation']['instances'].reverse()
            return r
        self.source.next=reversed_row
        result=self.run_batch();self.assertEqual(result['completed_count'],3,result)
    def test_uncertain_write_stops_whole_batch_and_keeps_pending(self):
        self.fail_step='close';result=self.run_batch()
        self.assertEqual(result['status'],'needs_review',result)
        self.assertEqual(self.factory_calls,1);self.assertIsNotNone(self.journal.pending())
        self.assertTrue((self.journal.root/'batch-pending.json').exists())
    def test_limit_is_partial_completion_not_all_done(self):
        result=self.run_batch(limit=1)
        self.assertEqual(result['status'],'item_limit_reached',result)
        self.assertEqual(result['completed_count'],1);self.assertEqual(len(result['remaining']),2)
    def test_neighbor_occupying_open_jaw_is_advisory_and_does_not_reject_plan(self):
        self.policy['jaw_open_span_mm']=150
        controller=self.factory(0)
        candidate,rejected=controller.select(controller.observe()[-1])
        self.assertIsNotNone(candidate,rejected)
        choice=candidate[2]
        self.assertFalse(choice['nominal_tool_footprint_clear'])
        self.assertFalse(choice['nominal_tool_footprint_check_enforced'])
        self.assertEqual(choice['nominal_neighbor_clearance_mm'],0)
        self.assertFalse(choice['full_link_collision_check'])
        self.assertTrue(choice['complete_route_compiled_before_motion'])
        self.assertFalse(any(m.sent for m in self.motions))
    def test_same_class_occupied_original_slot_is_not_reused(self):
        self.source.items=[i for i in self.source.items if i['label']=='paper']
        second=copy.deepcopy(self.source.items[0]);second['mask']=np.roll(second['mask'],-24,axis=1)
        self.source.items.append(second)
        self.source.items.sort(key=lambda i:int(np.nonzero(i['mask'])[1].min()))
        result=self.run_batch()
        self.assertEqual(result['completed_count'],1,result);self.assertEqual(result['status'],'blocked',result)
        self.assertIn('no_free_verified_bin_slot',result['reason'])
    def test_wrong_same_class_instance_removed_is_rejected(self):
        self.source.items=[i for i in self.source.items if i['label']=='paper']
        second=copy.deepcopy(self.source.items[0]);second['mask']=np.roll(second['mask'],-24,axis=1)
        self.source.items.append(second)  # Synthetic actuator deliberately moves the other instance.
        result=self.run_batch()
        self.assertEqual(result['status'],'needs_review',result);self.assertEqual(result['completed_count'],0)
    def test_same_class_two_explicit_slots_preserve_the_first_deposit(self):
        self.source.items=[i for i in self.source.items if i['label']=='paper']
        second=copy.deepcopy(self.source.items[0]);second['mask']=np.roll(second['mask'],-24,axis=1)
        self.source.items.append(second);self.source.items.sort(key=lambda i:int(np.nonzero(i['mask'])[1].min()))
        self.c['profiles']['paper']['bin_roi']=[90,80,158,118]
        slots=[dict(slot_id='paper-upper',origin='candidate',roi_fraction=[0,0,1,.5]),
               dict(slot_id='paper-lower',origin='candidate',roi_fraction=[0,.5,1,1])]
        self.policy['bin_slots']['paper']=slots
        class LandingMotion(DemoMotion):
            def open(self,plan):
                super().open(plan);self.landing_index=sum(i['placed'] for i in self.source.items)
            def send(self,data,duration,guard=None):
                name=self.names[0];super().send(data,duration,guard)
                if name=='release':
                    x,y,_,_=slot_rect(self.source.c['profiles']['paper'],slots[self.landing_index])
                    self.item['mask'][:]=False;self.item['mask'][y+4:y+14,x+5:x+45]=True
        self.motion_class=LandingMotion
        result=self.run_batch()
        self.assertEqual(result['status'],'completed',result);self.assertEqual(result['completed_count'],2)
        self.assertEqual([i['choice']['slot_id'] for i in result['items']],['paper-upper','paper-lower'])
    def test_candidate_landing_slots_are_not_physical_permissions(self):
        self.policy['bin_slots']['paper']=[dict(slot_id='unmeasured',origin='candidate',roi_fraction=[0,0,1,1])]
        controller=self.factory(0);controller.mode='known_physical';row=controller.acquire()
        self.assertEqual(controller.free_slots(row,'paper'),[])
    def test_empty_scene_never_opens_transport(self):
        self.source.items=[];result=self.run_batch()
        self.assertEqual(result['status'],'empty_scene',result);self.assertFalse(self.motions[0].opened)
    def test_material_filter_leaves_other_materials_as_tracked_obstacles(self):
        class PaperMotion(DemoMotion):
            def open(self,plan):
                super().open(plan)
                self.item=next(i for i in self.source.items if i['label']=='paper' and not i['placed'])
        self.motion_class=PaperMotion
        factory=self.factory
        def filtered(index):
            controller=factory(index);controller.allowed_labels=frozenset(('paper',));return controller
        self.factory=filtered
        result=self.run_batch()
        self.assertEqual(result['status'],'selection_complete',result)
        self.assertEqual(result['completed_count'],1)
        self.assertEqual({i['label'] for i in result['remaining']},{'metal','plastic'})

    def full_frame_filtered(self,label):
        self.policy['detection_window_px']=[0,0,*self.c['image_size']]
        class FilteredMotion(DemoMotion):
            def open(self,plan):
                super().open(plan);self.item=next(i for i in self.source.items if i['label']==label and not i['placed'])
        self.motion_class=FilteredMotion;factory=self.factory
        def filtered(index):
            controller=factory(index);controller.allowed_labels=frozenset((label,));return controller
        self.factory=filtered

    def test_full_frame_paper_interface_only_moves_paper(self):
        self.full_frame_filtered('paper');result=self.run_batch()
        self.assertEqual(result['status'],'selection_complete',result)
        self.assertEqual([i['choice']['label'] for i in result['items'] if i.get('choice')],['paper'])
        self.assertEqual({i['label'] for i in result['remaining']},{'metal','plastic'})

    def test_full_frame_metal_interface_only_moves_metal(self):
        self.full_frame_filtered('metal');result=self.run_batch()
        self.assertEqual(result['status'],'selection_complete',result)
        self.assertEqual([i['choice']['label'] for i in result['items'] if i.get('choice')],['metal'])

    def test_full_frame_plastic_interface_only_moves_plastic(self):
        self.full_frame_filtered('plastic');result=self.run_batch()
        self.assertEqual(result['status'],'selection_complete',result)
        self.assertEqual([i['choice']['label'] for i in result['items'] if i.get('choice')],['plastic'])

    def test_full_frame_combined_interface_still_sorts_all_three(self):
        self.policy['detection_window_px']=[0,0,*self.c['image_size']]
        result=self.run_batch();self.assertEqual(result['status'],'completed',result)
        self.assertEqual(result['completed_count'],3)

    def test_other_material_outside_old_window_or_low_confidence_is_not_selected(self):
        self.full_frame_filtered('paper')
        get=self.source.next
        def low_confidence(timeout):
            row=get(timeout)
            for obj in row['observation']['instances']:
                if obj['label']!='paper':obj['confidence']=.4
            return row
        self.source.next=low_confidence
        self.source.items[0]['mask']=np.roll(self.source.items[0]['mask'],-22,axis=0)
        result=self.run_batch()
        self.assertEqual(result['status'],'selection_complete',result)
        self.assertEqual(result['completed_count'],1)

    def test_full_frame_observes_target_beyond_old_source_without_changing_bin_routes(self):
        self.policy['detection_window_px']=[0,0,*self.c['image_size']]
        controller=self.factory(0);before=copy.deepcopy(self.c['profiles']['paper']['transfer_pwm'])
        controller.allowed_labels=frozenset(('paper',))
        controller.c['source_roi']=[28,45,59,83]  # Artificially tiny legacy crop.
        row=controller.acquire();selected,_=controller.select(row)
        self.assertIsNotNone(selected)
        self.assertEqual(selected[2]['label'],'paper')
        self.assertGreater(selected[2]['source_evidence_roi'][2],controller.c['source_roi'][2])
        self.assertEqual(len(controller.source_objects(row)),3)
        self.assertEqual(self.c['profiles']['paper']['transfer_pwm'],before)

    def test_existing_bin_object_is_excluded_from_full_frame_pick_pool(self):
        self.policy['detection_window_px']=[0,0,*self.c['image_size']]
        self.source.items=[i for i in self.source.items if i['label']=='paper']
        self.source.items[0]['mask'][:]=False;self.source.items[0]['mask'][90:100,100:120]=True
        self.source.items[0]['placed']=True
        result=self.run_batch();self.assertEqual(result['status'],'empty_scene',result)
        self.assertFalse(any(m.sent for m in self.motions))

    def test_target_region_failure_never_opens_or_moves_arm(self):
        self.policy['detection_window_px']=[0,0,*self.c['image_size']]
        factory=self.factory
        def guarded(index):
            controller=factory(index)
            def rejected(*_):raise ValueError('region_identity_changed')
            controller.target_region_setup=rejected;return controller
        self.factory=guarded;result=self.run_batch()
        self.assertEqual(result['status'],'blocked',result);self.assertIsNone(self.journal.pending())
        self.assertFalse(any(m.sent for m in self.motions))

    def test_tiny_mask_island_is_geometry_noise_but_raw_mask_stays_in_verification(self):
        self.policy['detection_window_px']=[0,0,*self.c['image_size']]
        self.source.items=[i for i in self.source.items if i['label']=='paper']
        self.source.items[0]['mask'][43,75]=True
        result=self.run_batch();self.assertEqual(result['status'],'completed',result)
        choice=result['items'][0]['choice']
        self.assertEqual(choice['mask_geometry_cleanup']['removed_island_px'],1)
        self.assertTrue(choice['mask_geometry_cleanup']['raw_mask_preserved_for_verification'])

    def test_large_disconnected_specimen_still_blocks_before_motion(self):
        self.policy['detection_window_px']=[0,0,*self.c['image_size']]
        self.source.items=[i for i in self.source.items if i['label']=='paper']
        self.source.items[0]['mask'][35:40,70:79]=True
        result=self.run_batch();self.assertEqual(result['status'],'blocked',result)
        self.assertIn('fragmented_specimen',result['reason']);self.assertFalse(any(m.sent for m in self.motions))

    def test_minor_bin_rectangle_overlap_is_source_not_already_deposited(self):
        self.policy['detection_window_px']=[0,0,*self.c['image_size']]
        self.source.items=[i for i in self.source.items if i['label']=='metal']
        self.c['source_roi']=[5,30,46,110]
        self.c['profiles']['metal']['bin_roi']=[47,82,89,118]
        class SmallLanding(DemoMotion):
            def send(self,data,duration,guard=None):
                name=self.names[0];super().send(data,duration,guard)
                if name=='release':
                    self.item['mask'][:]=False;self.item['mask'][87:97,52:82]=True
        self.motion_class=SmallLanding
        result=self.run_batch();self.assertEqual(result['status'],'completed',result)
        choice=result['items'][0]['choice'];self.assertGreater(choice['source_bin_overlap_fraction'],0)
        self.assertLessEqual(choice['source_bin_overlap_fraction'],.05)
        x0,y0,x1,y1=choice['source_evidence_roi']
        self.assertFalse(max(x0,47)<min(x1,89) and max(y0,82)<min(y1,118))

    def test_per_bin_reference_does_not_replace_other_bins_or_relax_change_limit(self):
        controller=self.factory(0);row=controller.acquire()
        changed=self.source.reference.copy();x0,y0,x1,y1=self.c['profiles']['metal']['bin_roi']
        changed[y0:y1,x0:x1]=255-self.source.reference[y0:y1,x0:x1]
        controller.empty_reference=changed
        self.assertEqual(controller.free_slots(row,'metal'),[])
        controller.empty_by_label={'metal':self.source.reference}
        self.assertEqual(len(controller.free_slots(row,'metal')),1)
        self.assertIs(controller.empty_reference,changed)
        self.assertEqual(controller.c['max_scene_changed_fraction'],.02)

    def test_wider_post_verification_timeout_keeps_prewrite_and_frame_age_limits(self):
        self.policy['verification_timeout_s']=5
        self.source.items=[i for i in self.source.items if i['label']=='metal']
        phase=[0];timeouts=[];after=self.source.after;get=self.source.next
        def barrier(stamp):phase[0]+=1;return after(stamp)
        def captured(timeout):timeouts.append((phase[0],timeout));return get(timeout)
        self.source.after=barrier;self.source.next=captured
        result=self.run_batch();self.assertEqual(result['status'],'completed',result)
        self.assertTrue(all(t==2 for p,t in timeouts if p==1))
        self.assertEqual([t for p,t in timeouts if p==2],[5,5,5])
        self.assertEqual(self.c['max_age_ms'],2000);self.assertEqual(self.c['frame_timeout_s'],2)

    def test_verification_waits_for_three_consecutive_valid_frames_without_repeating_motion(self):
        self.policy['verification_timeout_s']=5
        self.source.items=[i for i in self.source.items if i['label']=='metal']
        after=self.source.after;get=self.source.next;phase=[0];post=[0]
        def barrier(stamp):phase[0]+=1;return after(stamp)
        def captured(timeout):
            row=get(timeout)
            if phase[0]==2:
                post[0]+=1
                if post[0]==1:row['observation']['instances'][0]['confidence']=.4
            return row
        self.source.after=barrier;self.source.next=captured
        result=self.run_batch();self.assertEqual(result['status'],'completed',result)
        self.assertEqual(post[0],4);self.assertEqual(self.motions[0].sent.count('close'),1)

    def test_five_second_post_age_accepts_delayed_inference_after_retreat_only(self):
        self.policy.update(verification_timeout_s=5,verification_max_age_ms=5000)
        self.source.items=[i for i in self.source.items if i['label']=='metal']
        after=self.source.after;get=self.source.next;phase=[0];post=[0]
        def barrier(stamp):phase[0]+=1;return after(stamp)
        def captured(timeout):
            row=get(timeout)
            if phase[0]==2:
                post[0]+=1
                if post[0]==1:self.clock.sleep(3)
            return row
        self.source.after=barrier;self.source.next=captured
        result=self.run_batch();self.assertEqual(result['status'],'completed',result)
        self.assertEqual(self.c['max_age_ms'],2000)

    def test_slow_pre_motion_frame_still_rejected_at_two_seconds(self):
        self.policy.update(verification_timeout_s=5,verification_max_age_ms=5000)
        get=self.source.next
        def captured(timeout):
            row=get(timeout);self.clock.sleep(3);return row
        self.source.next=captured
        result=self.run_batch();self.assertEqual(result['status'],'blocked',result)
        self.assertFalse(any(m.sent for m in self.motions))
    def test_batch_lease_survives_all_inter_item_gaps(self):
        lease=self.root/'device.lock'
        with self.journal:
            with BatchLease(lease,[],self.clock.monotonic()+1200,clock=self.clock.monotonic) as owner:
                def factory(index):
                    self.assertTrue(lease.exists());return self.factory(index)
                result=run_batch(factory,self.journal,6,1200,owner.check,clock=self.clock.monotonic)
                self.assertEqual(result['status'],'completed',result);self.assertTrue(lease.exists())
        self.assertFalse(lease.exists())
    def test_existing_lease_is_not_removed_or_overwritten(self):
        lease=self.root/'device.lock';lease.write_text('other-owner')
        with self.assertRaises(FileExistsError):
            with BatchLease(lease,[],self.clock.monotonic()+10,clock=self.clock.monotonic):pass
        self.assertEqual(lease.read_text(),'other-owner')
    def test_stop_race_after_acquiring_lease_releases_only_own_lease(self):
        lease=self.root/'device.lock';stop=self.root/'STOP'
        class StopRace(BatchLease):
            def check(owner):
                if owner.owned:stop.write_text('stop')
                super().check()
        with self.assertRaisesRegex(ValueError,'operator_stop'):
            with StopRace(lease,[stop],self.clock.monotonic()+10,clock=self.clock.monotonic):pass
        self.assertFalse(lease.exists());self.assertTrue(stop.exists())

if __name__=='__main__':unittest.main()
