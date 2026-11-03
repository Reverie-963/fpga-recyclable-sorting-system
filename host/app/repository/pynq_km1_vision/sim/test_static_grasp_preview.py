import copy,unittest
import numpy as np
from src.static_grasp_preview import plan_target, stable_interior, fuse, draw_overlay


def encode(mask):
    flat=mask.ravel();edges=np.flatnonzero(flat[1:]!=flat[:-1])+1
    return ([0] if flat[0] else [])+np.diff(np.r_[0,edges,len(flat)]).tolist()


class PlannerChecks(unittest.TestCase):
    def setUp(self):
        mask=np.zeros((12,12),np.uint8);mask[2:10,3:7]=1
        self.item={'label':'cardboard','confidence':.98,'bbox_xyxy':[3,2,7,10],'mask_rle':encode(mask)}
        self.rows=[{'model_version':'fixed','camera_id':'test','image_size':[12,12],
                    'frame_id':i+1,'capture_started_ms':100+i*100,
                    'instances':[copy.deepcopy(self.item)]} for i in range(3)]
        self.policy={'stable_frames':3,'model_version':'fixed','camera_id':'test','image_size':[12,12],
                     'max_age_ms':1000,'latest_max_age_ms':800,'label':'cardboard','min_confidence':.8,
                     'source_roi_xyxy':[0,0,12,12],'min_area_px':20,'max_area_px':80,
                     'min_interior_px':1,'max_stable_shift_px':3}
        self.cal={'checked_for_supervised_preview':True,'surface_height_mm':65,
                  'camera_id':'test','image_size':[12,12],
                  'surface_image_to_card':np.eye(3).tolist(),
                  'grip_card_to_control':[[1,0,0],[0,1,145],[0,0,1]],
                  'hover_card_to_control_left':[[2/3,0,0],[0,4/7,1040/7],[0,0,1]],
                  'hover_card_to_control_right':[[1,0,0],[0,4/7,1040/7],[0,0,1]],
                  'verified_point_hulls_card_xy_mm':[[[-10,-10],[12,-10],[12,12],[-10,12]]]*2}

    def run_plan(self):
        return plan_target(self.rows,self.policy,self.cal,1000)

    def test_interior_and_separate_height_coordinates(self):
        result=self.run_plan();p=result['target']
        self.assertGreater(p['edge_clearance_px'],0)
        self.assertNotEqual(p['grip_xyz_nominal'][1],p['hover_xyz_nominal'][1])
        self.assertFalse(result['physical_motion']);self.assertFalse(result['unattended_execution_allowed'])

    def test_left_scale_is_distinct(self):
        self.cal['surface_image_to_card'][0][2]=-10
        p=self.run_plan()['target']
        self.assertLess(p['card_xy_mm'][0],0)
        self.assertAlmostEqual(p['hover_xyz_nominal'][0],p['card_xy_mm'][0]*2/3)
        self.assertAlmostEqual(p['grip_xyz_nominal'][0],p['card_xy_mm'][0])

    def test_wrong_model(self):
        self.rows[-1]['model_version']='changed'
        with self.assertRaisesRegex(ValueError,'wrong_model'):self.run_plan()

    def test_any_second_class_blocks(self):
        d=copy.deepcopy(self.item);d['label']='plastic';d['confidence']=.3
        self.rows[1]['instances'].append(d)
        with self.assertRaisesRegex(ValueError,'source_not_single'):self.run_plan()

    def test_stale_capture(self):
        self.policy['max_age_ms']=100
        with self.assertRaisesRegex(ValueError,'stale_or_invalid'):self.run_plan()

    def test_duplicate_frame(self):
        self.rows[1]['frame_id']=1
        with self.assertRaisesRegex(ValueError,'duplicate_or_old'):self.run_plan()

    def test_bbox_mismatch(self):
        self.rows[-1]['instances'][0]['bbox_xyxy'][2]=8
        with self.assertRaisesRegex(ValueError,'mask_bbox'):self.run_plan()

    def test_calibration_pending(self):
        self.cal['checked_for_supervised_preview']=False
        with self.assertRaisesRegex(ValueError,'calibration_checks_pending'):self.run_plan()

    def test_checked_point_region(self):
        self.cal['verified_point_hulls_card_xy_mm']=[[[20,20],[30,20],[30,30],[20,30]]]*2
        with self.assertRaisesRegex(ValueError,'outside_checked'):self.run_plan()

    def test_moving_target(self):
        m=np.zeros((12,12),np.uint8);m[2:10,4:8]=1
        self.rows[-1]['instances'][0].update(mask_rle=encode(m),bbox_xyxy=[4,2,8,10])
        self.policy['max_stable_shift_px']=.1
        with self.assertRaisesRegex(ValueError,'target_moving'):self.run_plan()

    def test_one_pixel_width_change_does_not_jump_along_body(self):
        cfg={'grasp':{'confidence':.8,'min_box_coverage':.9,'clearance_px':3,'min_interior_px':3},
             'vision':{'min_area':100,'max_area':10000},'arm':{'bins':{'cardboard':None}}}
        masks=[]
        for top_wider in (False,True):
            m=np.zeros((120,80),np.uint8);m[10:110,25:55]=1
            if top_wider:m[10:55,24]=1
            else:m[65:110,24]=1
            masks.append(m)
        raw=[];stable=[]
        for m in masks:
            c=fuse([{'label':'cardboard','confidence':.99,'xyxy':[24,10,55,110]}],m*255,cfg,None)[0]
            raw.append(c['center']);point,clearance,peak=stable_interior(m,c,3,.8);stable.append(point)
            self.assertGreaterEqual(clearance,.8*peak-1e-5)
            self.assertEqual(m[round(point[1]),round(point[0])],1)
        self.assertGreater(np.linalg.norm(np.array(raw[0])-raw[1]),20)
        self.assertLessEqual(np.linalg.norm(np.array(stable[0])-stable[1]),2)

    def test_hole_center_still_selects_material(self):
        m=np.zeros((90,90),np.uint8);m[10:80,10:80]=1;m[30:60,30:60]=0
        c={'bbox':[10,10,70,70],'center':[20.,20.]}
        point,clearance,peak=stable_interior(m,c,3,.8)
        self.assertEqual(m[round(point[1]),round(point[0])],1)
        self.assertGreaterEqual(clearance,max(3,.8*peak)-1e-5)

    def test_safe_material_point_selected_inside_checked_region(self):
        m=np.zeros((120,80),np.uint8);m[10:110,25:55]=1
        c={'bbox':[25,10,30,100],'center':[39.,59.]}
        hull=[[[41,62],[42,62],[42,64],[41,64]]]
        point,clearance,peak=stable_interior(m,c,3,.8,np.eye(3),hull)
        self.assertTrue(41<=point[0]<=42 and 62<=point[1]<=64)
        self.assertGreaterEqual(clearance,.8*peak-1e-5)
        self.assertEqual(m[round(point[1]),round(point[0])],1)

    def test_checked_region_only_in_hole_is_rejected(self):
        m=np.zeros((90,90),np.uint8);m[10:80,10:80]=1;m[30:60,30:60]=0
        c={'bbox':[10,10,70,70],'center':[20.,20.]}
        hull=[[[35,35],[55,35],[55,55],[35,55]]]
        with self.assertRaisesRegex(ValueError,'outside_checked_region'):
            stable_interior(m,c,3,.8,np.eye(3),hull)

    def test_changed_placement_changes_coordinates_without_old_position(self):
        original = self.run_plan()['target']
        m = np.zeros((12,12), np.uint8); m[2:10,6:10] = 1
        for row in self.rows:
            row['instances'][0].update(mask_rle=encode(m), bbox_xyxy=[6,2,10,10])
        moved = self.run_plan()['target']
        self.assertAlmostEqual(moved['pick_pixel'][0] - original['pick_pixel'][0], 3)
        self.assertAlmostEqual(moved['grip_xyz_nominal'][0] - original['grip_xyz_nominal'][0], 3)

    def test_no_checked_regions_rejected(self):
        self.cal['verified_point_hulls_card_xy_mm'] = []
        with self.assertRaisesRegex(ValueError, 'missing_height_checked_regions'):
            self.run_plan()

    def test_wrong_calibration_camera_rejected(self):
        self.cal['camera_id'] = 'moved-camera'
        with self.assertRaisesRegex(ValueError, 'calibration_camera_or_size_changed'):
            self.run_plan()

    def test_degenerate_checked_region_rejected(self):
        self.cal['verified_point_hulls_card_xy_mm'][0] = [[0,0],[1,0],[2,0]]
        with self.assertRaisesRegex(ValueError, 'invalid_checked_region'):
            self.run_plan()

    def test_one_height_domain_cannot_stand_in_for_both(self):
        self.cal['verified_point_hulls_card_xy_mm'] = self.cal['verified_point_hulls_card_xy_mm'][:1]
        with self.assertRaisesRegex(ValueError, 'missing_height_checked_regions'):
            self.run_plan()

    def test_overlay_keeps_source_image_and_rejects_other_size(self):
        image = np.zeros((12,12,3), np.uint8)
        overlay = draw_overlay(image, self.cal, self.run_plan())
        self.assertEqual(overlay.shape, image.shape)
        self.assertFalse(image.any())
        with self.assertRaisesRegex(ValueError, 'overlay_image_size_changed'):
            draw_overlay(np.zeros((10,10,3), np.uint8), self.cal, self.run_plan())

    def test_63mm_cannot_silently_relabel_legacy_65mm(self):
        self.cal['surface_height_mm'] = 63
        with self.assertRaisesRegex(ValueError, 'surface_height_mismatch'):
            self.run_plan()
        self.policy['surface_height_mm'] = 63
        with self.assertRaisesRegex(ValueError, 'object_profile_mismatch'):
            self.run_plan()

    def bind_63mm_profile(self):
        self.policy.update(surface_height_mm=63, object_profile_id='carton63')
        self.cal.update(surface_height_mm=63, object_profile_id='carton63',
                        robot_mapping_height_mm=63, grip_model_z_mm=48,
                        hover_model_z_mm=72, working_alpha_deg=-75)

    def test_bound_height_uses_own_checked_control_poses(self):
        self.bind_63mm_profile()
        result = self.run_plan()
        self.assertEqual(result['target']['grip_xyz_nominal'][2], 48)
        self.assertEqual(result['target']['hover_xyz_nominal'][2], 72)
        self.assertEqual(result['surface_height_mm'], 63)
        self.assertEqual(result['object_profile_id'], 'carton63')

    def test_height_binding_rejects_old_mapping_or_unchecked_profile(self):
        self.bind_63mm_profile()
        self.cal['robot_mapping_height_mm'] = 65
        with self.assertRaisesRegex(ValueError, 'robot_mapping_height_not_checked'):
            self.run_plan()
        self.cal['robot_mapping_height_mm'] = 63
        self.cal['checked_for_supervised_preview'] = False
        with self.assertRaisesRegex(ValueError, 'calibration_checks_pending'):
            self.run_plan()

    def test_bound_height_requires_valid_pose_and_profile(self):
        self.bind_63mm_profile()
        self.cal['grip_model_z_mm'] = 80
        with self.assertRaisesRegex(ValueError, 'invalid_height_pose'):
            self.run_plan()
        self.cal['grip_model_z_mm'] = 48
        self.cal['object_profile_id'] = 'other-carton'
        with self.assertRaisesRegex(ValueError, 'object_profile_mismatch'):
            self.run_plan()

if __name__=='__main__':unittest.main()
