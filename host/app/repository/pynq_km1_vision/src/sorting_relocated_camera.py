"""Overlap three-frame capture/HTTP inference without relaxing freshness."""
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import copy
import time
import urllib.request

from .sorting_loop import need


class WindowedStationCamera:
    def __init__(self,source):
        self.source=source;self.queue=deque();self.batch_next=True

    def after(self,stamp):
        self.queue.clear();self.batch_next=True;self.source.after(stamp)

    def raw(self,timeout):return self.source.raw(timeout)

    def next(self,timeout):
        if self.queue:
            row=self.queue.popleft()
            need(0<=time.monotonic()*1000-row['observation']['capture_started_ms']<=self.source.c['max_age_ms'],
                 'batched_observation_expired')
            return row
        if not self.batch_next:return self.source.next(timeout)
        self.batch_next=False;deadline=time.monotonic()+timeout
        futures=[]
        def infer(row):
            detector=copy.copy(self.source.detector)
            detector.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            observation=detector.predict(row['image'],prompt_mode=self.source.c['prompt_mode'])
            observation['model_input_image_sha256']=observation.get('image_sha256')
            observation.update(row['observation']);row['observation']=observation
            return row
        with ThreadPoolExecutor(max_workers=3) as pool:
            for _ in range(3):
                need(time.monotonic()<deadline,'three_frame_capture_deadline_exceeded')
                row=self.source.raw(max(.05,deadline-time.monotonic()))
                futures.append(pool.submit(infer,row))
            rows=[future.result() for future in futures]
        need(time.monotonic()<=deadline,'three_frame_inference_deadline_exceeded')
        need(all(0<=time.monotonic()*1000-r['observation']['capture_started_ms']<=self.source.c['max_age_ms']
                 for r in rows),'three_frame_inference_window_expired')
        self.queue.extend(rows)
        return self.queue.popleft()
