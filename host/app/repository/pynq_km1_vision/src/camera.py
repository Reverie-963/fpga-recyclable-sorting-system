"""One-frame mailbox; capture continues during inference/motion. Timestamp is host read completion."""
import threading,time
import cv2

class LatestCamera:
    def __init__(self,cfg):
        self.cap=cv2.VideoCapture(cfg['index']);self.cap.set(3,cfg['width']);self.cap.set(4,cfg['height']);self.cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        if not self.cap.isOpened():self.cap.release();raise RuntimeError('Camera unavailable')
        self.cv=threading.Condition();self.stopped=False;self.error=None;self.latest=None;self.fid=0
        self.thread=threading.Thread(target=self._read,daemon=True);self.thread.start()
    def _read(self):
        warmup=10
        try:
            while not self.stopped:
                ok,frame=self.cap.read();stamp=time.monotonic()
                if not ok:raise RuntimeError('Camera read failed')
                if warmup:warmup-=1;continue
                with self.cv:
                    self.fid+=1;self.latest=(self.fid,stamp,frame);self.cv.notify_all()
        except Exception as e:
            with self.cv:self.error=e;self.cv.notify_all()
        finally:self.cap.release()
    def read(self,after=0,timeout=3):
        deadline=time.monotonic()+timeout
        with self.cv:
            while self.latest is None or self.latest[0]<=after:
                if self.error:raise self.error
                left=deadline-time.monotonic()
                if left<=0:raise TimeoutError('No fresh camera frame')
                self.cv.wait(left)
            if self.error:raise self.error
            fid,stamp,frame=self.latest
            return fid,stamp,frame.copy()
    def close(self):
        self.stopped=True;self.thread.join(timeout=2)
        # Some UVC backends block read until device timeout. Thread is daemon; no concurrent release.
