"""Rate-limited preview on a separate worker; event log buffering bounded in time."""
import json,queue,threading,time
from pathlib import Path
import cv2

class Recorder:
    def __init__(self,folder,preview_interval_s=2,flush_interval_s=1):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.log=(self.folder/'events.jsonl').open('a');self.interval=preview_interval_s;self.flush_interval=flush_interval_s
        self.last_preview=0;self.last_flush=time.monotonic();self.q=queue.Queue(maxsize=1)
        self.worker=threading.Thread(target=self._write,daemon=True);self.worker.start();self.preview_error=None
    def _write(self):
        while True:
            item=self.q.get()
            try:
                if item is None:return
                frame,mask=item
                if not cv2.imwrite(str(self.folder/'latest.jpg'),frame) or not cv2.imwrite(str(self.folder/'mask.png'),mask):raise IOError('Preview write failed')
            except Exception as e:self.preview_error=str(e)
            finally:self.q.task_done()
    def event(self,record,critical=False):
        self.log.write(json.dumps(record,allow_nan=False)+'\n');now=time.monotonic()
        if critical or now-self.last_flush>=self.flush_interval:self.log.flush();self.last_flush=now
    def preview(self,frame,mask):
        now=time.monotonic()
        if self.interval>0 and now-self.last_preview>=self.interval:
            try:self.q.put_nowait((frame.copy(),mask.copy()));self.last_preview=now
            except queue.Full:pass
    def close(self):
        self.q.put(None);self.worker.join();self.log.close()
