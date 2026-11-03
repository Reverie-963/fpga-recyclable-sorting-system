import hashlib
import time
from types import SimpleNamespace
import unittest

import numpy as np

from src.sorting_relocated_camera import WindowedStationCamera


class WindowedCameraTests(unittest.TestCase):
    def build(self):
        owner=self
        class Detector:
            def predict(self,image,prompt_mode):
                value=int(image[0,0,0]);time.sleep(.005*(4-value%3))
                return dict(image_sha256='model-'+str(value),instances=[],prompt_mode=prompt_mode)
        class Source:
            def __init__(self):
                self.c=dict(prompt_mode='supervised',max_age_ms=2000);self.detector=Detector();self.count=0;self.single=0
            def raw(self,timeout):
                self.count+=1;value=self.count
                return dict(image=np.full((3,4,3),value,np.uint8),observation=dict(frame_id=value,
                    source_frame_id=str(value),image_sha256='source-'+str(value),
                    capture_started_ms=time.monotonic()*1000))
            def next(self,timeout):self.single+=1;return self.raw(timeout)
            def after(self,stamp):self.barrier=stamp
        self.source=Source();return WindowedStationCamera(self.source)

    def test_parallel_completion_preserves_frame_order_and_both_image_hashes(self):
        source=self.build();rows=[source.next(1) for _ in range(3)]
        self.assertEqual([r['observation']['frame_id'] for r in rows],[1,2,3])
        for i,row in enumerate(rows,1):
            self.assertEqual(row['observation']['image_sha256'],'source-'+str(i))
            self.assertEqual(row['observation']['model_input_image_sha256'],'model-'+str(i))
        self.assertEqual(self.source.count,3)
        source.next(1);self.assertEqual(self.source.single,1)

    def test_retreat_barrier_discards_all_cached_pre_retreat_frames(self):
        source=self.build();source.next(1)
        source.after(123.);row=source.next(1)
        self.assertEqual(row['observation']['frame_id'],4)
        self.assertEqual(self.source.barrier,123.)

    def test_cached_frame_age_is_checked_when_delivered_not_only_when_inferred(self):
        source=self.build();source.next(1)
        source.queue[0]['observation']['capture_started_ms']-=3000
        with self.assertRaisesRegex(ValueError,'expired'):source.next(1)


if __name__=='__main__':unittest.main()
