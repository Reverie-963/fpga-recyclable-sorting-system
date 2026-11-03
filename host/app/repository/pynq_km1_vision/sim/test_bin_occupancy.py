"""Occupancy semantics and optional full local sorting integration."""
import copy
from importlib.util import find_spec
import unittest
from unittest.mock import patch
import numpy as np
from src.bin_occupancy import bin_occupancy,verification_options

OPTIONS=dict(mode='bin_occupied',rgb_difference_threshold=45,min_component_px=64,min_detection_confidence=.2)


class OccupancyTests(unittest.TestCase):
    def setUp(self):
        self.empty=np.full((60,80,3),100,np.uint8)
        self.image=self.empty.copy();self.roi=[20,10,70,50]

    def check(self):return bin_occupancy(self.image,self.empty,self.roi,OPTIONS)

    def test_empty_bin_fails(self):self.assertFalse(self.check()['occupied'])

    def test_visible_object_without_any_detector_input_passes(self):
        self.image[20:28,30:38]=[200,100,100]
        self.assertTrue(self.check()['occupied']);self.assertEqual(self.check()['largest_component_px'],64)

    def test_object_in_other_bin_does_not_pass(self):
        self.image[:10,:10]=255
        self.assertFalse(self.check()['occupied'])

    def test_scattered_noise_does_not_add_up_to_an_object(self):
        self.image[10:50:2,20:70:2]=255
        result=self.check();self.assertGreater(result['changed_px'],64)
        self.assertFalse(result['occupied']);self.assertEqual(result['largest_component_px'],1)

    def test_low_pixel_noise_is_ignored(self):
        self.image[:]=110
        self.assertFalse(self.check()['occupied'])

    def test_dark_objects_do_not_wrap_unsigned_subtraction(self):
        self.image[20:28,30:38]=0
        self.assertTrue(self.check()['occupied'])

    def test_existing_object_counts_as_occupied(self):
        self.image[20:30,30:40]=255
        self.assertTrue(self.check()['occupied'])

    def test_roi_outside_frame_is_rejected(self):
        for roi in ([0,0,81,60],[10,10,10,20],[-1,0,10,10]):
            with self.subTest(roi=roi),self.assertRaises(ValueError):
                bin_occupancy(self.image,self.empty,roi,OPTIONS)

    def test_bad_image_or_reference_is_rejected(self):
        for ref in (None,self.empty[:20],self.empty.astype(float)):
            with self.subTest(),self.assertRaises(ValueError):bin_occupancy(self.image,ref,self.roi,OPTIONS)

    def test_legacy_default_remains_semantic(self):
        self.assertEqual(verification_options({}),{'mode':'semantic_rgb'})

    def test_invalid_thresholds_fail_closed(self):
        for key,value in (('mode','typo'),('min_component_px',0),('min_component_px',True),
                          ('rgb_difference_threshold',0),('rgb_difference_threshold',float('nan'))):
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                verification_options({'placement_verification':dict(OPTIONS,**{key:value})})

    def test_current_console_policy_enables_simple_mode(self):
        from src.prompted_motion_policy import read_motion_options
        self.assertEqual(verification_options(read_motion_options()),OPTIONS)


AVAILABLE=find_spec('sim.test_sorting_unified') is not None


@unittest.skipUnless(AVAILABLE,'requires the existing local native sorting integration and fixtures')
class OccupancyIntegrationTests(unittest.TestCase):
    def fixture(self):
        from sim.test_sorting_unified import UnifiedBatchTests
        f=UnifiedBatchTests();f.setUp();self.addCleanup(f.doCleanups)
        f.source.items=[i for i in f.source.items if i['label']=='metal']
        f.policy['placement_verification']=copy.deepcopy(OPTIONS)
        return f

    def test_complete_cycle_with_no_post_motion_model_detections(self):
        f=self.fixture();get=f.source.next
        def without_detections(timeout):
            row=get(timeout)
            if f.source.items[0]['placed']:row['observation']['instances']=[]
            return row
        f.source.next=without_detections
        c=f.factory(0)
        with f.journal:result=c.cycle()
        self.assertEqual(result['status'],'simulated_visual_pass',result)
        self.assertEqual(result['verification']['status'],'bin_occupied_pass')
        self.assertEqual(result['verification_window']['stable_frames'],1)
        self.assertFalse(result['verification']['new_item_verified'])
        self.assertFalse(result['verification']['physical_success_verified'])
        self.assertEqual(len(c.tracker.attempted),1)
        self.assertIsNone(f.journal.pending())

    def test_wrong_bin_keeps_pending_and_does_not_retry_motion(self):
        f=self.fixture();c=f.factory(0);c.motion.wrong_bin=True
        with f.journal:result=c.cycle()
        self.assertEqual(result['status'],'needs_review',result)
        self.assertEqual(c.motion.sent.count('close'),1)
        self.assertIsNotNone(f.journal.pending())

    def test_serial_fault_never_becomes_occupancy_success(self):
        f=self.fixture();f.fail_step='close';c=f.factory(0)
        with f.journal:result=c.cycle()
        self.assertEqual(result['status'],'needs_review',result)
        self.assertNotIn('verification',result)
        self.assertIsNotNone(f.journal.pending())

    def test_old_empty_reference_mismatch_cannot_pass_unchanged_bin(self):
        f=self.fixture();c=f.factory(0);row=c.acquire()
        p=c.c['profiles']['metal'];c.empty_reference[:]=0
        with self.assertRaisesRegex(ValueError,'target_bin_empty'):
            c.verify_bin_occupied(row,'metal',dict(bin_id=p['bin_id']),row)

    def test_no_new_instance_or_source_removal_requirement(self):
        f=self.fixture();c=f.factory(0);row=c.acquire()
        before=copy.deepcopy(row)
        p=c.c['profiles']['metal'];x,y,_,_=p['bin_roi'];row['image'][y+5:y+15,x+5:x+15]=255
        result=c.verify_selected(before,(row['observation']['instances'][0],row['masks'][0]),
                                 dict(bin_id=p['bin_id']),[row])
        self.assertEqual(result['status'],'bin_occupied_pass')
        self.assertFalse(result['source_removal_verified'])

    def test_existing_low_confidence_different_class_in_bin_is_sufficient(self):
        f=self.fixture();c=f.factory(0);row=c.acquire()
        p=c.c['profiles']['metal'];x,y,_,_=p['bin_roi']
        mask=np.zeros(row['image'].shape[:2],bool);mask[y+5:y+15,x+5:x+15]=True
        row['masks'].append(mask);row['observation']['instances'].append(dict(label='plastic',confidence=.21))
        result=c.verify_bin_occupied(row,'metal',dict(bin_id=p['bin_id']),row)
        self.assertEqual(result['basis'],'any_bin_detection')
        self.assertFalse(result['frames'][0]['occupied'])

    def test_stale_frame_is_still_rejected(self):
        f=self.fixture();c=f.factory(0);get=f.source.next
        def stale(timeout):
            row=get(timeout);row['observation']['capture_started_ms']-=30000;return row
        f.source.next=stale
        with self.assertRaisesRegex(ValueError,'stale_or_future_frame'):
            c.verification_frames(c.clock()*1000)

    def test_prompted_override_does_not_reintroduce_label_check(self):
        from src.sorting_prompted_motion import PromptedMotionController
        from src.sorting_unified import UnifiedController
        with patch.object(UnifiedController,'verify_selected',return_value={'status':'bin_occupied_pass'}):
            # Call the method on a genuine subclass instance without devices.
            obj=object.__new__(PromptedMotionController);obj.placement_verification=OPTIONS
            self.assertEqual(obj.verify_selected({},({},None),{},[{}])['status'],'bin_occupied_pass')

    def test_batch_defer_cannot_skip_simple_check(self):
        f=self.fixture();f.policy.update(batch_verification_mode='end_of_batch',skip_per_item_visual_verification=True)
        with self.assertRaisesRegex(ValueError,'bin_occupancy_requires_per_item'):f.factory(0)
