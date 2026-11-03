"""Bound workspace filtering keeps all raw detections and rejects new neighbors."""
import copy
import unittest
from types import SimpleNamespace
import numpy as np
from src.sorting_unified import UnifiedController,registered_workspace_roi
from src.sorting_prompted_motion import PromptedMotionController
from src.sorting_item_heights import ItemHeights
from sim.test_sorting_item_heights import plan_for


class RegisteredWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.c=object.__new__(PromptedMotionController)
        config=dict(image_size=[100,100],source_roi=[30,25,60,65],
            profiles={'metal':{'bin_roi':[70,40,95,65]}},layout={'static_rois':[]},
            statistics={'regions':{'unified_source':[20,20,65,70]}},min_scene_iou=.85,max_scene_changed_fraction=.05)
        self.c.c=config;self.c.geometry=SimpleNamespace(original=copy.deepcopy(config))
        self.c.policy=dict(detection_window_px=[0,0,100,100],max_scene_instances=6)
        self.c.motion_options={'full_scene_instance_stability_check_enabled':False}
        self.c.trial_labels=['Drink can'];self.c.changed_pixels=lambda a,b:np.zeros((100,100),bool)
        self.target=dict(label='plastic',official_label='Drink can',zone='source',semantic_gate_passed=True,bbox_xyxy=[35,35,40,45])
        self.mask=np.zeros((100,100),bool);self.mask[35:45,35:40]=1
        self.c.batch_context={'target':(self.target,self.mask)}
        p=plan_for([(self.target,self.mask)],[100,100],[43])
        self.c.item_heights=ItemHeights(p,model='a'*64,configuration='b'*64,image_size=[100,100])

    def row(self,extra=None):
        instances=[dict(self.target)];masks=[self.mask.copy()]
        if extra is not None:
            instances.append(dict(label='metal',official_label='Drink can',zone='source',semantic_gate_passed=True,bbox_xyxy=[80,0,90,10]));masks.append(extra)
        return dict(observation=dict(instances=instances,source_stream_id='stream'),masks=masks,image=np.zeros((100,100,3),np.uint8))

    def test_external_corner_detection_does_not_reinstate_whole_frame_height_matching(self):
        noise=np.zeros_like(self.mask);noise[:10,80:90]=1
        before,after=self.row(),self.row(noise)
        self.c.unchanged(before,after)
        self.assertEqual(len(self.c.source_objects(after)),1)
        self.assertEqual(len(self.c.outside_objects(after)),1)
        self.assertEqual(len(after['masks']),2)
        self.assertTrue(after['target_stability']['all_latest_masks_retained_for_obstacle_checks'])

    def test_new_workspace_neighbor_or_boundary_intrusion_still_requires_height(self):
        for rect in ((50,50,55,55),(29,30,31,40)):
            noise=np.zeros_like(self.mask);x,y,x1,y1=rect;noise[y:y1,x:x1]=1
            with self.assertRaisesRegex(ValueError,'instance_count_changed'):
                self.c.unchanged(self.row(),self.row(noise))

    def test_source_region_survives_target_evidence_crop_without_expanding_to_bins(self):
        self.c.c['source_roi']=[35,35,40,45]
        self.assertEqual(registered_workspace_roi(self.c.c,self.c.geometry.original),[20,20,65,70])
        self.assertIsNone(registered_workspace_roi(self.c.c,{}))
        self.c.geometry.original={}
        noise=np.zeros_like(self.mask);noise[:10,80:90]=1
        self.assertEqual(len(self.c.source_objects(self.row(noise))),2)

    def test_right_hand_object_between_source_and_bin_needs_no_height(self):
        outside=np.zeros_like(self.mask);outside[30:40,65:69]=True
        row=self.row(outside);pairs=self.c.source_objects(row)
        self.assertEqual(len(pairs),1);self.assertEqual(len(self.c.outside_objects(row)),1)
        self.c.item_heights.bind(pairs,'stream',.85)
        self.assertEqual(self.c.item_heights.measurement(pairs[0][0])['object_height_mm'],43)
        self.assertEqual(len(self.c.obstacle_objects(row)),2)

    def test_external_obstacle_projects_on_target_plane_without_fake_height(self):
        calls=[];target=self.target;outside={'label':'metal'}
        self.c.map_object=lambda obj,points:(calls.append(obj) or np.asarray(points))
        self.c.obstacle_points(outside,[[65,35]],target,{id(target)})
        self.assertIs(calls[0],target)
        self.c.obstacle_points(outside,[[65,35]],target,{id(outside)})
        self.assertIs(calls[1],outside)


if __name__=='__main__':unittest.main()
