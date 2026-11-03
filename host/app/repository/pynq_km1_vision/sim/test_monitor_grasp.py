"""Image markers are frame-bound display candidates, never motion permission."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from src.monitor_grasp import display_candidates
from src.yoloe_inference import encode_mask
from src.monitor_control import ControlTelemetry
from sim.test_monitor_control import save


class GraspDisplayTests(unittest.TestCase):
    def test_concave_mask_candidate_stays_inside_and_is_not_calibrated(self):
        mask=np.zeros((15,15),np.uint8);mask[1:14,1:5]=1;mask[10:14,1:14]=1
        raw=dict(image_size=[15,15],model_version='m',frame_id=7,instances=[dict(label='plastic',instance_id=0,mask_rle=encode_mask(mask))])
        out=display_candidates(raw);point=out['display_grasp_points'][0]
        self.assertEqual(mask[point['pixel'][1],point['pixel'][0]],1)
        self.assertFalse(point['control_authorized'])
        self.assertFalse(point['calibrated'])
        self.assertNotIn('display_grasp_points',raw)
        self.assertEqual(out['frame_id'],7)
        self.assertIs(out['instances'],raw['instances'])

    def test_invalid_rle_cannot_create_a_point(self):
        for runs in ([100],[-1,10],[True,8]):
            out=display_candidates(dict(image_size=[3,3],instances=[dict(mask_rle=runs)]))
            self.assertEqual(out['display_grasp_points'],[])

    def test_timing_is_cpu_only_and_bound_to_the_same_observation(self):
        mask=np.zeros((12,16),np.uint8);mask[2:10,3:14]=1
        raw=dict(image_size=[16,12],model_version='m',runtime_id='r',frame_id=7,
            source_frame_id='42',source_stream_id='camera-a',
            instances=[dict(label='metal',instance_id=0,mask_rle=encode_mask(mask))])
        out=display_candidates(raw);timing=out['display_geometry_timing']
        self.assertEqual(timing['backend'],'pc_cpu_opencv_numpy')
        for key in ('model_version','runtime_id','frame_id','source_frame_id','source_stream_id'):
            self.assertEqual(timing[key],raw[key])
        self.assertEqual((timing['decoded_masks'],timing['point_count']),(1,1))
        self.assertGreaterEqual(timing['elapsed_ms'],timing['mask_decode_ms']+timing['point_compute_ms'])
        self.assertGreaterEqual(timing['mask_decode_ms'],0)
        self.assertGreaterEqual(timing['point_compute_ms'],0)
        self.assertFalse(timing['calibrated']);self.assertFalse(timing['control_authorized'])
        self.assertNotIn('display_geometry_timing',raw)

    def test_invalid_input_cannot_inherit_an_upstream_timing_claim(self):
        raw=dict(image_size=[99999,480],display_geometry_timing=dict(elapsed_ms=1,backend='gpu'))
        self.assertNotIn('display_geometry_timing',display_candidates(raw))
        raw['image_size']=[10,8];raw['instances']='invalid'
        self.assertNotIn('display_geometry_timing',display_candidates(raw))

    def test_empty_image_measures_work_without_inventing_a_grasp_point(self):
        out=display_candidates(dict(image_size=[10,8],instances=[]))
        timing=out['display_geometry_timing']
        self.assertEqual(timing['point_count'],0)
        self.assertEqual(timing['decoded_masks'],0)
        self.assertEqual(timing['point_compute_ms'],0)
        self.assertEqual(out['display_grasp_points'],[])

    def test_original_pixels_required_for_executed_point_image(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);trial=root/'trials/can';frame=np.full((8,10,3),120,np.uint8)
            plan=dict(task_id='can',class_from_yoloe='metal',model_version='m',pick_pixel=[3,4],
                evidence_prefix='capture-',stages=[dict(name='stage',steps=[])])
            ph=save(trial/'plan.json',plan)
            save(trial/'journal/stage.json',dict(task_id='can',stage='stage',plan_sha256=ph,
                status='sent_pending_observation',serial_opened=True,steps=[]))
            save(trial/'capture-before-2.json',dict(decoded_bgr_sha256=hashlib.sha256(frame.tobytes()).hexdigest()))
            upload=root/'local-only/yolov8s-deployment/preview/upload';upload.mkdir(parents=True)
            ok,png=cv2.imencode('.png',frame);self.assertTrue(ok);raw=png.tobytes();(upload/'input-image').write_bytes(raw)
            save(upload/'result.json',dict(model_version='m',image_sha256=hashlib.sha256(raw).hexdigest()))
            with patch('src.monitor_control.ROOT',root):
                reader=ControlTelemetry([root/'trials'],refresh_s=0)
                out=reader.grasp_frame(dict(model_sha256='m'),'can')
                self.assertTrue(out['available'])
                self.assertEqual(out['pixel'],[3,4])
                self.assertFalse(out['current_live_frame'])
                (upload/'input-image').write_bytes(b'changed')
                fresh=ControlTelemetry([root/'trials'],refresh_s=0)
                self.assertFalse(fresh.grasp_frame(dict(model_sha256='m'),'can')['available'])

    def test_hash_bound_lossless_frame_and_repeated_child_point_keep_origin(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);trial=root/'trials/plastic';frame=np.full((8,10,3),80,np.uint8)
            trial.mkdir(parents=True);ok,png=cv2.imencode('.png',frame);self.assertTrue(ok)
            raw=png.tobytes();(trial/'before-2.png').write_bytes(raw);digest=hashlib.sha256(raw).hexdigest()
            save(trial/'before-2.json',dict(decoded_bgr_sha256=hashlib.sha256(frame.tobytes()).hexdigest(),
                lossless_png_sha256=digest,selected_instance_index=0,
                observation=dict(model_version='m',source_frame_id=37,instances=[dict(label='plastic')])))
            plan=dict(task_id='plastic',profile='bottle',model_version='m',pick_pixel=[3,4],
                preflight_reference_path='plastic/before-2.png',preflight_reference_sha256=digest,stages=[])
            ph=save(trial/'plan.json',plan)
            record=dict(task_id='plastic',stage='approach',plan_sha256=ph,status='sent_pending_observation',steps=[])
            parent=trial/'journal/stage.json';eh=save(parent,record)
            child=root/'trials/return'
            plan=dict(task_id='return',profile='bottle',model_version='m',pick_pixel=[3,4],
                prior_plan_path='plastic/plan.json',prior_plan_sha256=ph,
                prior_execution_path='plastic/journal/stage.json',prior_execution_sha256=eh,stages=[])
            ph=save(child/'plan.json',plan);record.update(task_id='return',plan_sha256=ph)
            save(child/'journal/stage.json',record)
            with patch('src.monitor_control.ROOT',root):
                reader=ControlTelemetry([root/'trials'],refresh_s=0)
                status=reader.status({},'return')
                self.assertEqual(status['label'],'plastic');self.assertEqual(status['grasp_source_frame'],37)
                self.assertEqual(status['grasp_source_task_id'],'plastic')
                self.assertTrue(reader.grasp_frame({},'return')['available'])
                (trial/'before-2.png').write_bytes(b'corrupt')
                fresh=ControlTelemetry([root/'trials'],refresh_s=0)
                self.assertFalse(fresh.grasp_frame({},'return')['available'])


if __name__=='__main__':unittest.main()
