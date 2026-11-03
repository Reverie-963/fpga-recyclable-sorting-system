"""Known-item scope and failure tests. Every image, device and record is synthetic."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.sorting_demo import DemoClock, DemoSource, DemoMotion, demo_config
from src.sorting_io import Statistics, Motion
from src.sorting_known import (GeometryGuardedMotion, KnownController, SCOPE, file_hash,
                              known_readiness, profile_fingerprint, validate_known_config)
from src.sorting_loop import Controller, Journal
from src.sorting_scene import adjust_exposure_reference, verify_setup_reanchor
from src.sorting_known_run import capture_empty_reference


def config():
    c = demo_config()
    c.update(model_family='yolov8s', prompt_mode='supervised', model_version='a'*64,
             runtime_id='b'*64, max_objects=1, max_cycle_s=180, min_scene_iou=.95, min_interior_px=3,
             camera_url='http://127.0.0.1:8780/frame.jpg')
    c['layout'].update(min_patch_correlation=.98, min_motion_patch_dynamic_range=80)
    c['arm'].update(move_ms=5000, settle_s=.5, step_settle_s={'release':1},
                    pick_bounds_mm=[-20,20,145,165], usb_identity={'vid':0x1a86,'pid':0x7523})
    c['known'] = dict(scope=SCOPE, experiment_authorized=False, initial_pwm=[1500]*5+[1000],
        prepare_pwm=[1500,1632,1987,928], grip_move_ms=2000,
        device_lease='device.lock', stop_files=['STOP'])
    for i, (label, p) in enumerate(c['profiles'].items()):
        p.update(image_to_card_mm=[[1,0,-(19+i*24)], [0,-.5,37.5], [0,0,1]],
            card_to_grip_mm=[[1,0,0],[0,1.25,145],[0,0,1]],
            card_to_hover_mm=[[1,0,0],[0,.75,152],[0,0,1]],
            pick_hulls_card_mm=[[[-20,0],[20,0],[0,12]],[[-20,0],[20,0],[0,10]]],
            width_bounds_mm=[5,15], length_bounds_mm=[15,30], max_grasp_offset_mm=10,
            grasp_z_mm=10, source_mid_z_mm=40, hover_z_mm=70, alpha_deg=-75,
            wrist_pwm=1500, open_pwm=1000, close_pwm=1300, axis_tolerance_deg=15,
            transfer_pwm=[[1500,1632,1987,928],[1400+i*100,1632,1987,928]],
            retreat_pwm=[[1400+i*100,1632,1987,928],[1500]*4], qualification_record=None)
    return c


class KnownCycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.build('plastic')

    def build(self, label):
        self.c = config(); self.clock = DemoClock()
        self.source = DemoSource(self.c, self.clock)
        self.source.items = [i for i in self.source.items if i['label'] == label]
        self.motion = DemoMotion(self.source, self.clock)
        self.journal = Journal(self.root / label)
        self.controller = KnownController(self.c,self.source,Statistics(),self.motion,self.journal,
            self.source.reference,self.source.reference,self.root,mode='simulation',
            clock=self.clock.monotonic,sleep=self.clock.sleep)

    def cycle(self):
        with self.journal:return self.controller.cycle()

    def test_each_material_routes_one_item_then_empty(self):
        for i, label in enumerate(('plastic','metal','paper')):
            with self.subTest(label=label):
                self.build(label)
                result = self.cycle()
                self.assertEqual(result['status'],'simulated_visual_pass',result)
                self.assertEqual(result['choice']['label'],label)
                self.assertIn('#000P%04d' % (1400+i*100), ''.join(p['command'] for p in result['plan']))
                self.assertFalse(result['production_quality_claimed'])
                self.assertEqual(self.cycle()['status'],'empty_scene')

    def test_preview_compiles_whole_recipe_without_usb(self):
        self.controller.mode='preview'
        result=self.cycle()
        self.assertEqual(result['status'],'preview_only',result)
        self.assertEqual(result['plan'][-1]['step'],'return')
        self.assertEqual(self.motion.sent,[])
        self.assertTrue(result['execution_blockers'])

    def test_operator_selected_point_seven_accepts_source_and_bin_above_threshold(self):
        self.controller.c['min_confidence']=.7
        original=self.source.next
        def observed(timeout):
            row=original(timeout)
            for obj in row['observation']['instances']:obj['confidence']=.72
            return row
        self.source.next=observed
        self.assertEqual(self.cycle()['status'],'simulated_visual_pass')

    def test_point_seven_rejects_weak_source_before_motion(self):
        self.controller.c['min_confidence']=.7
        original=self.source.next
        def observed(timeout):
            row=original(timeout)
            for obj in row['observation']['instances']:obj['confidence']=.69
            return row
        self.source.next=observed
        result=self.cycle()
        self.assertEqual(result['status'],'rejected')
        self.assertEqual(self.motion.sent,[])

    def test_point_seven_does_not_waive_a_weak_post_place_frame(self):
        self.controller.c['min_confidence']=.7
        original=self.source.next
        def observed(timeout):
            row=original(timeout)
            for obj in row['observation']['instances']:
                obj['confidence']=.69 if self.motion.sent and self.motion.sent[-1]=='return' else .72
            return row
        self.source.next=observed
        result=self.cycle()
        self.assertEqual(result['status'],'needs_review')
        self.assertIn('no_unique_new_target_in_correct_bin',result['reason'])

    def test_expected_arm_occlusion_does_not_fabricate_in_grip_inference(self):
        original=self.source.next
        def next_frame(timeout):
            if self.motion.opened and self.motion.sent and self.motion.sent[-1]!='return':
                raise AssertionError('Semantic source inference requested while synthetic arm occludes it')
            return original(timeout)
        self.source.next=next_frame
        self.assertEqual(self.cycle()['status'],'simulated_visual_pass')

    def test_partial_write_latches_and_restart_never_replays(self):
        self.motion.fail_step='source_grip'
        result=self.cycle()
        self.assertEqual(result['status'],'needs_review',result)
        self.assertEqual(self.motion.sent[-1],'source_grip')
        with self.assertRaisesRegex(ValueError,'unresolved_previous_task'):self.cycle()

    def test_wrong_bin_stops_for_review(self):
        self.motion.wrong_bin=True
        result=self.cycle()
        self.assertEqual(result['status'],'needs_review',result)
        self.assertIn('no_unique_new_target_in_correct_bin',result['reason'])

    def test_unchecked_grasp_domain_rejects_before_usb(self):
        self.controller.c['profiles']['plastic']['pick_hulls_card_mm']=[[[100,100],[120,100],[110,110]]]*2
        result=self.cycle()
        self.assertEqual(result['status'],'rejected',result)
        self.assertIn('no_internal_grasp_in_checked_domain',result['reason'])
        self.assertEqual(self.motion.sent,[])

    def test_unreachable_later_pose_prevents_first_command(self):
        self.controller.c['profiles']['plastic']['source_mid_z_mm']=400
        result=self.cycle()
        self.assertEqual(result['status'],'rejected',result)
        self.assertEqual(self.motion.sent,[])

    def test_multiple_source_objects_reject_in_single_cycle_scope(self):
        full=DemoSource(self.c,self.clock)
        self.source.items=full.items
        result=self.cycle()
        self.assertEqual(result['status'],'rejected',result)
        self.assertIn('too_many_source_objects',result['reason'])
        self.assertEqual(self.motion.sent,[])

    def test_unrecognized_rgb_object_in_destination_is_not_empty(self):
        original=self.source.next
        def occupied(timeout):
            row=original(timeout);x,y,_,_=self.c['profiles']['plastic']['bin_roi']
            row['image'][y+1:y+20,x+1:x+20]=255
            h=hashlib.sha256(row['image'].tobytes()).hexdigest()
            row.update(jpeg_sha256=h,bgr_sha256=h);row['observation']['image_sha256']=h
            return row
        self.source.next=occupied
        result=self.cycle()
        self.assertEqual(result['status'],'rejected',result)
        self.assertIn('known_destination_rgb_not_empty',result['reason'])
        self.assertEqual(self.motion.sent,[])


class KnownReadinessTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);self.c=config()
        self.c.update(synthetic_fixture=False,statistics=dict(backend='pl_http',assets={'vision.bit':'c'*64,'vision.hwh':'d'*64}))
        self.c['known']['experiment_authorized']=True
        source=DemoSource(self.c,DemoClock());cv2.imwrite(str(self.base/'reference.png'),source.reference)
        ref=dict(path='reference.png',sha256=file_hash(self.base/'reference.png'))
        self.c['layout_reference']=ref;self.c['empty_reference']=ref
        for i in range(2):(self.base/('synthetic-evidence%d.json'%i)).write_text('{"synthetic_test_only":true}',encoding='utf-8')
        for label,p in self.c['profiles'].items():
            q=dict(kind='known-item-geometry-observation-v1',label=label,station_id=self.c['station_id'],scope=SCOPE,
                operator_confirmed=True,completed_supervised_trial=True,profile_sha256=profile_fingerprint(self.c,label),
                evidence=[dict(path='synthetic-evidence%d.json'%i,sha256=file_hash(self.base/('synthetic-evidence%d.json'%i))) for i in range(2)])
            f=self.base/(label+'.json');f.write_text(json.dumps(q),encoding='utf-8')
            p['qualification_record']=dict(path=f.name,sha256=file_hash(f))

    def test_bound_known_scope_does_not_enable_production(self):
        self.assertEqual(known_readiness(self.c,self.base),[])
        self.assertFalse(self.c['quality_gate_passed'])
        with self.assertRaisesRegex(ValueError,'station_not_ready'):
            Controller(self.c,None,None,None,None,None,mode='physical')

    def test_changed_profile_or_evidence_cannot_reuse_qualification(self):
        self.c['profiles']['plastic']['close_pwm']=1350
        self.assertTrue(any('plastic.qualification' in b for b in known_readiness(self.c,self.base)))
        self.c['profiles']['plastic']['close_pwm']=1300
        (self.base/'synthetic-evidence0.json').write_text('{}')
        self.assertTrue(any('.qualification' in b for b in known_readiness(self.c,self.base)))

    def test_synthetic_fixture_and_missing_setup_remain_blocked(self):
        self.c['synthetic_fixture']=True;self.c['known']['experiment_authorized']=False
        self.c['empty_reference']=None
        blockers=known_readiness(self.c,self.base)
        self.assertIn('synthetic_fixture',blockers)
        self.assertIn('known.experiment_authorized',blockers)
        self.assertTrue(any(b.startswith('empty_reference:') for b in blockers))

    def test_batch_scope_and_fast_motion_are_rejected(self):
        self.c['max_objects']=3
        with self.assertRaisesRegex(ValueError,'single_known_item_scope_required'):validate_known_config(self.c)
        self.c['max_objects']=1;self.c['arm']['move_ms']=1500
        with self.assertRaisesRegex(ValueError,'known_route_low_speed_required'):validate_known_config(self.c)

    def test_return_pose_must_be_neutral(self):
        self.c['profiles']['metal']['retreat_pwm'][-1][0]=1700
        with self.assertRaisesRegex(ValueError,'route_must_end_at_neutral'):validate_known_config(self.c)

    def test_decision_guards_cannot_be_silently_relaxed(self):
        for key,value in [('min_confidence',.3),('min_scene_iou',.5),('max_age_ms',10000),
                          ('rgb_difference_threshold',500),('max_scene_changed_fraction',.5)]:
            with self.subTest(key=key):
                changed=copy.deepcopy(self.c);changed[key]=value
                with self.assertRaisesRegex(ValueError,'known_decision_guards_relaxed'):validate_known_config(changed)

    def test_other_usb_device_cannot_reuse_the_same_profile_record(self):
        self.c['arm']['port']='SYNTHETIC_OTHER_DEVICE'
        self.assertTrue(any('.qualification' in b for b in known_readiness(self.c,self.base)))

    def test_stop_file_blocks_readiness_without_clearing_it(self):
        stop=self.base/'STOP';stop.write_text('synthetic stop')
        self.assertIn('operator_stop_requested',known_readiness(self.c,self.base))
        self.assertEqual(stop.read_text(),'synthetic stop')

    def test_read_only_empty_capture_binds_exact_png_and_preserves_production_gates(self):
        clock=DemoClock();source=DemoSource(self.c,clock);source.items=source.items[:1]
        path=self.base/'station.json';path.write_text(json.dumps(self.c),encoding='utf-8')
        with Journal(self.base/'capture') as journal:
            controller=KnownController(self.c,source,Statistics(),DemoMotion(source,clock),journal,
                source.reference,None,self.base,mode='preview',clock=clock.monotonic,sleep=clock.sleep)
            with patch('src.sorting_known_run.time',SimpleNamespace(monotonic=clock.monotonic)):
                result=capture_empty_reference(self.c,self.base,controller,journal,path)
        self.assertEqual(result['status'],'ready_for_one_supervised_known_item',result)
        updated=json.loads(path.read_text(encoding='utf-8'))
        self.assertFalse(updated['quality_gate_passed'])
        self.assertEqual(file_hash((self.base/updated['empty_reference']['path']).resolve()),updated['empty_reference']['sha256'])
        self.assertTrue(result['production_execution_blockers'])

    def test_empty_capture_rejects_an_occupied_other_box(self):
        clock=DemoClock();source=DemoSource(self.c,clock);first=source.items[0];old=source.items[1]
        x,y,_,_=self.c['profiles']['metal']['bin_roi'];old['mask'][:]=False;old['mask'][y+2:y+12,x+2:x+42]=True;old['placed']=True
        source.items=[first,old]
        path=self.base/'station.json';original=json.dumps(self.c);path.write_text(original,encoding='utf-8')
        with Journal(self.base/'occupied-capture') as journal:
            controller=KnownController(self.c,source,Statistics(),DemoMotion(source,clock),journal,
                source.reference,None,self.base,mode='preview',clock=clock.monotonic,sleep=clock.sleep)
            with self.assertRaisesRegex(ValueError,'all_three_bins_must_be_empty'):
                capture_empty_reference(self.c,self.base,controller,journal,path)
        self.assertEqual(path.read_text(encoding='utf-8'),original)


class MotionGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.c=config();self.clock=DemoClock();self.source=DemoSource(self.c,self.clock)
        self.journal=Journal(self.root/'journal');self.lease=self.root/'device.lock'
        self.motion=GeometryGuardedMotion(self.c,self.journal,self.source.reference,self.lease,[self.root/'STOP'])
        self.devices=patch('serial.tools.list_ports.comports',return_value=[SimpleNamespace(device=self.c['arm']['port'],vid=0x1a86,pid=0x7523)])
        self.devices.start();self.addCleanup(self.devices.stop)

    def test_other_controller_lease_is_never_removed(self):
        self.lease.write_text('other')
        with self.assertRaises(FileExistsError):self.motion.open([])
        self.motion.close();self.assertEqual(self.lease.read_text(),'other')

    def test_usb_open_failure_releases_owned_lease(self):
        with patch.object(Motion,'open',side_effect=IOError('synthetic USB failure')):
            with self.assertRaises(IOError):self.motion.open([])
        self.assertFalse(self.lease.exists())

    def test_raw_guard_precedes_every_packet_and_preserves_png(self):
        with self.journal:
            self.journal.save('pending.json',dict(task_id='synthetic'))
            self.motion.expected_stream=lambda:self.source.stream
            self.motion.camera=SimpleNamespace(raw=self.source.next)
            with patch.object(Motion,'send',side_effect=lambda data,duration,guard:guard()):
                self.motion.send(b'one',5000);self.motion.send(b'two',5000)
            self.assertEqual(self.motion.guard_index,2)
            self.assertEqual(len(list(self.journal.root.glob('synthetic.motion_*.png'))),2)


class PhotometryTests(unittest.TestCase):
    def setUp(self):
        self.c=config();self.source=DemoSource(self.c,DemoClock())

    def test_bounded_brightness_change_does_not_warp_real_object_gain(self):
        a=self.source.reference
        b=np.clip(a.astype(float)*.9+12,0,255).astype(np.uint8)
        b[50:60,40:50]=255
        ref,evidence=adjust_exposure_reference(a,b,dict(self.c['layout'],image_size=self.c['image_size']))
        self.assertIn('no_geometric_warp',evidence['method'])
        self.assertTrue((np.abs(b[50:60,40:50].astype(int)-ref[50:60,40:50].astype(int)).sum(axis=2)>90).all())

    def test_geometric_drift_is_not_exposure_compensated(self):
        a=self.source.reference;b=np.roll(a,3,axis=0)
        with self.assertRaises(ValueError):adjust_exposure_reference(a,b,dict(self.c['layout'],image_size=self.c['image_size']))


class SetupReanchorTests(unittest.TestCase):
    def setUp(self):
        self.a=np.full((480,640,3),130,np.uint8)
        self.policy=dict(image_size=[640,480],static_rois=[[84,25,143,90],[490,20,550,85]],
            search_margin_px=5,max_layout_shift_px=1,min_patch_std=8,min_patch_corners=4,
            min_patch_peak_margin=.01,min_patch_correlation=.98,min_motion_patch_dynamic_range=80)
        for x0,y0,x1,y1 in self.policy['static_rois']:
            patch=self.a[y0:y1,x0:x1];patch[:]=230
            cv2.rectangle(patch,(4,4),(patch.shape[1]-5,patch.shape[0]-5),(20,20,20),2)
            cv2.line(patch,(10,32),(45,32),(20,20,20),3)
            cv2.line(patch,(29,12),(29,52),(20,20,20),3)
            for xy in [(12,14),(44,16),(13,49),(43,48)]:cv2.rectangle(patch,xy,(xy[0]+3,xy[1]+3),(20,20,20),-1)

    def test_operator_setup_accepts_uneven_lighting_only_with_subpixel_tracks(self):
        b=self.a.copy()
        for i,(x0,y0,x1,y1) in enumerate(self.policy['static_rois']):
            gain=[1.1,1.05,.95] if i==0 else [.8,.95,1.0]
            bias=[0,5,10] if i==0 else [55,30,20]
            b[y0:y1,x0:x1]=np.clip(b[y0:y1,x0:x1].astype(float)*gain+bias,0,255).astype(np.uint8)
        report=verify_setup_reanchor(self.a,b,self.policy)
        self.assertFalse(report['motion_policy_relaxed'])
        self.assertFalse(report['robot_mapping_changed'])
        self.assertTrue(all(m['tracked_corners']>=8 and m['max_displacement_px']<=1 for m in report['markers']))

    def test_setup_cannot_reanchor_a_moved_camera(self):
        with self.assertRaises(ValueError):verify_setup_reanchor(self.a,np.roll(self.a,3,axis=0),self.policy)

    def test_setup_cannot_learn_blackout_or_occluded_marker(self):
        b=self.a.copy();b[20:85,490:550]=0
        with self.assertRaises(ValueError):verify_setup_reanchor(self.a,b,self.policy)


if __name__=='__main__':unittest.main()
