"""Camera relocation integration; every device/frame in these tests is synthetic."""
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

from sim.test_sorting_known import config
from src.sorting_demo import DemoSource,DemoClock,DemoMotion
from src.sorting_io import Statistics
from src.sorting_known import compile_known_plan
from src.sorting_loop import Journal,mapped,mask_of
from src.sorting_scene import check_motion_layout
from src.station_calibration import generate_kit
from src.station_geometry import detect_markers,fingerprint,localize_records,world_to_pixels
from src.sorting_relocated_geometry import pixels_to_card,scene_from_frame,RelocatedGeometry,bounded_rectangle
from src.sorting_relocated_plan import compile_relocated_plan
from src.sorting_relocated import RelocatedController,RelocatedMotion
from src.sorting_relocated_run import activate_regions
from src.sorting_relocated_run import motion_shift_limit


class RelocatedRouteTests(unittest.TestCase):
    def test_original_commands_and_guards_match_for_all_three_materials(self):
        c=config()
        for label in c['profiles']:
            source=DemoSource(c,DemoClock());source.items=[i for i in source.items if i['label']==label]
            row=source.next(1);obj=row['observation']['instances'][0];mask=mask_of(obj,c['image_size'])
            geometry=Statistics().geometry(mask)
            mapper=lambda points:mapped(c['profiles'][label]['image_to_card_mm'],points)
            old,old_plan=compile_known_plan(c,label,mask,geometry)
            new,new_plan=compile_relocated_plan(c,label,mask,geometry,mapper)
            self.assertEqual(old,new);self.assertEqual(old_plan,new_plan)
            with self.assertRaisesRegex(ValueError,'explicit_calibrated'):
                compile_relocated_plan(c,label,mask,geometry,None)
            # A callback cannot override original checked point domains.
            with self.assertRaisesRegex(ValueError,'checked_domain'):
                compile_relocated_plan(c,label,mask,geometry,lambda pts:mapper(pts)+[100,0])

    def test_full_relocated_controller_preserves_each_simulated_closed_loop(self):
        with tempfile.TemporaryDirectory() as root:
            for label in ('paper','metal','plastic'):
                c=config();clock=DemoClock();source=DemoSource(c,clock)
                source.items=[i for i in source.items if i['label']==label]
                acquire=source.next
                def source_frame(timeout):
                    row=acquire(timeout);row['observation']['source_frame_id']=str(row['observation']['frame_id']);return row
                source.next=source_frame
                policy=dict(c['layout'],image_size=c['image_size'])
                geometry=SimpleNamespace(runtime=c,reference=source.reference,signature='f'*64,
                    map=lambda label,pts:mapped(c['profiles'][label]['image_to_card_mm'],pts),
                    check=lambda image:check_motion_layout(source.reference,image,policy))
                with Journal(Path(root)/label) as journal:
                    controller=RelocatedController(c,source,Statistics(),DemoMotion(source,clock),journal,
                        geometry,source.reference,root,mode='simulation',clock=clock.monotonic,sleep=clock.sleep)
                    result=controller.cycle()
                self.assertEqual(result['status'],'simulated_visual_pass',result)
                self.assertFalse(result['choice']['station_calibration']['old_image_homography_used'])
                self.assertFalse(result['automatic_camera_update_during_motion'])

    def test_pending_task_blocks_board_mutation_even_when_arm_lease_is_free(self):
        with tempfile.TemporaryDirectory() as root:
            c=config();c['journal_dir']='journal';c['known']['device_lease']='arm.lock'
            ledger=Path(root)/'journal';ledger.mkdir();(ledger/'pending.json').write_text('{}')
            stats=SimpleNamespace(request=lambda *args:self.fail('must not contact board'))
            with self.assertRaisesRegex(ValueError,'physical_trial_unresolved'):
                activate_regions(stats,SimpleNamespace(),c,Path(root))
            self.assertFalse((Path(root)/'arm.lock').exists())

    def test_each_motion_write_requires_frozen_geometry_and_no_external_trial(self):
        # Exercise the callback boundary without constructing any real transport.
        motion=object.__new__(RelocatedMotion)
        with tempfile.TemporaryDirectory() as root:
            motion.external_pending=Path(root)/'pending.json';motion.stop_files=[]
            motion.c=dict(frame_timeout_s=1,max_age_ms=2000)
            row=dict(image=np.zeros((8,10,3),np.uint8),observation=dict(source_stream_id='fixed',
                capture_started_ms=0))
            motion.camera=SimpleNamespace(raw=lambda timeout:row)
            motion.expected_stream=lambda:'fixed';motion.guard_index=0
            motion.geometry=SimpleNamespace(check=lambda image:(_ for _ in ()).throw(ValueError('camera moved')))
            written=[]
            def transport(self,data,duration,guard):guard();written.append(data)
            with patch('src.sorting_relocated.Motion.send',transport):
                with self.assertRaisesRegex(ValueError,'camera moved'):motion.send(b'test',5000)
                motion.external_pending.write_text('{}')
                with self.assertRaisesRegex(ValueError,'trial_unresolved'):motion.send(b'test',5000)
                motion.owned_task_id='this-cycle';motion.external_pending.write_text('{"task_id":"this-cycle"}')
                motion.check_stop()
                motion.external_pending.write_text('{"task_id":"other-cycle"}')
                with self.assertRaisesRegex(ValueError,'trial_unresolved'):motion.check_stop()
            self.assertEqual(written,[])


class RelocatedGeometryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name);generate_kit(root/'kit')
        self.station=json.loads((root/'kit/station.design.json').read_text())
        self.camera=dict(kind='station-camera-v1',camera_id='synthetic',lens_profile_id='fixed',image_size=[640,480],
            camera_matrix=[[650.,0,320],[0,650.,240],[0,0,1.]],dist_coeffs=[0.]*5)
        self.reg=dict(checked=True,world_frame_id=self.station['world_frame_id'],
            world_to_card_xy=[[1.,0,-142.],[0,1.,20.],[0,0,1.]])
        k=np.array(self.camera['camera_matrix']);rv=np.array([3.1,.08,-.03]);r=cv2.Rodrigues(rv)[0];t=-r@np.array([130.,-100.,750.])
        image=np.full((480,640,3),230,np.uint8)
        dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        for m in self.station['markers']:
            p=cv2.projectPoints(np.array(m['corners_world_mm'],float),rv,t,k,np.zeros(5))[0].reshape(4,2)
            code=cv2.aruco.generateImageMarker(dictionary,m['id'],180)
            h=cv2.getPerspectiveTransform(np.float32([[0,0],[179,0],[179,179],[0,179]]),p.astype(np.float32))
            warped=cv2.warpPerspective(code,h,(640,480),borderValue=255)
            image=np.minimum(image,cv2.cvtColor(warped,cv2.COLOR_GRAY2BGR))
        self.image=image;detections=detect_markers(image,self.camera)
        self.assertGreaterEqual(len(detections),3)
        records=[dict(image_sha256=hashlib.sha256(str(i).encode()).hexdigest(),detections=detections) for i in range(3)]
        self.pose=localize_records(records,self.camera,self.station,'fixed_marker_task_space')
        self.c=config();self.c['image_size']=[640,480];self.c['source_roi']=[280,110,405,235]
        for label,rect in [('plastic',[20,300,200,445]),('metal',[450,80,620,240]),('paper',[450,300,620,445])]:
            self.c['profiles'][label]['bin_roi']=rect
        self.row=dict(image=image,bgr_sha256=hashlib.sha256(image.tobytes()).hexdigest())
        self.scene=scene_from_frame(self.c,self.row,self.pose,self.camera,self.station,self.reg,
                                    {k:0 for k in ('bin_plastic','bin_metal','bin_paper')})

    def test_pixel_height_maps_to_original_card_frame_and_changes_with_height(self):
        xyz=[142.,-17.,53.];pixel=world_to_pixels([xyz],self.pose,self.camera,self.station)
        result=pixels_to_card(pixel,53,self.pose,self.camera,self.station,self.reg)
        np.testing.assert_allclose(result,[[0,3]],atol=1e-5)
        wrong=pixels_to_card(pixel,0,self.pose,self.camera,self.station,self.reg)
        self.assertGreater(np.linalg.norm(wrong-result),1)

    def test_regions_and_robot_domains_preserved_and_marker_loss_rejects(self):
        before=copy.deepcopy(self.c)
        g=RelocatedGeometry(self.c,self.camera,self.station,self.reg,self.scene,self.pose,self.image)
        self.assertEqual(self.c,before)
        self.assertEqual(g.robot_domains,{k:p['pick_hulls_card_mm'] for k,p in self.c['profiles'].items()})
        self.assertNotEqual(g.runtime['layout']['static_rois'],self.c['layout']['static_rois'])
        with self.assertRaisesRegex(ValueError,'three_fixed_markers'):
            g.check(np.full_like(self.image,255))
        changed=copy.deepcopy(self.scene);changed['registration_sha256']='a'*64
        with self.assertRaisesRegex(ValueError,'scene_calibration_changed'):
            RelocatedGeometry(self.c,self.camera,self.station,self.reg,changed,self.pose,self.image)

    def test_empty_reference_requires_confirmation_and_never_changes_live_pixels(self):
        g=RelocatedGeometry(self.c,self.camera,self.station,self.reg,self.scene,self.pose,self.image)
        with self.assertRaisesRegex(ValueError,'not_confirmed'):g.warp_empty_reference(self.image)
        scene=dict(self.scene,empty_bins_confirmed=True)
        g=RelocatedGeometry(self.c,self.camera,self.station,self.reg,scene,self.pose,self.image)
        before=self.image.copy();empty=g.warp_empty_reference(self.image)
        np.testing.assert_array_equal(before,self.image)
        self.assertEqual(empty.shape,self.image.shape)

    def test_projected_regions_cannot_be_clipped_or_widened_outside_bin(self):
        with self.assertRaisesRegex(ValueError,'outside_image'):
            bounded_rectangle([[-1,2],[20,2],[20,20],[2,20]],[640,480])
        polygon=np.float32([[10,10],[40,15],[35,40],[15,35]])
        rect=bounded_rectangle(polygon,[640,480],inset=True)
        from src.sorting_relocated_geometry import rectangle_points
        self.assertTrue(all(cv2.pointPolygonTest(polygon,tuple(map(float,p)),False)>=0 for p in rectangle_points(rect)))

    def test_operator_motion_margin_applies_to_corners_and_static_patch_search(self):
        old=copy.deepcopy(self.c)
        g=RelocatedGeometry(self.c,self.camera,self.station,self.reg,self.scene,self.pose,self.image,motion_shift_px=5)
        shifted=cv2.warpAffine(self.image,np.float32([[1,0,4],[0,1,0]]),(640,480),borderValue=(255,255,255))
        self.assertEqual(g.check(shifted)['motion_shift_limit_px'],5)
        self.assertEqual(g.runtime['layout']['max_layout_shift_px'],5)
        self.assertEqual(g.runtime['layout']['search_margin_px'],5)
        self.assertEqual(self.c,old)
        self.assertEqual(g.robot_domains,{k:p['pick_hulls_card_mm'] for k,p in self.c['profiles'].items()})
        shifted=cv2.warpAffine(self.image,np.float32([[1,0,6],[0,1,0]]),(640,480),borderValue=(255,255,255))
        with self.assertRaisesRegex(ValueError,'fixed_board_moved'):g.check(shifted)

    def test_margin_requires_operator_record_and_is_bound_to_geometry_identity(self):
        self.assertEqual(motion_shift_limit({}),1)
        policy=dict(kind='operator-approved-station-motion-margin-v1',operator_statement='把边界放开',
            production_point_domains_changed=False,max_displacement_px=5)
        self.assertEqual(motion_shift_limit(dict(motion_guard_policy=policy)),5)
        for value in (True,0,6,float('nan')):
            with self.assertRaises(ValueError):motion_shift_limit(dict(motion_guard_policy=dict(policy,max_displacement_px=value)))
        with self.assertRaises(ValueError):motion_shift_limit(dict(motion_guard_policy=dict(policy,operator_statement='')))
        original=RelocatedGeometry(self.c,self.camera,self.station,self.reg,self.scene,self.pose,self.image)
        amended=RelocatedGeometry(self.c,self.camera,self.station,self.reg,self.scene,self.pose,self.image,motion_shift_px=5)
        self.assertNotEqual(original.signature,amended.signature)


if __name__=='__main__':unittest.main()
