"""Board-local camera and ROI evidence service. Never opens an arm interface.

Changing a baseline is explicit, frame-bound and has no motion permission.
The service owns one UVC capture and one persistent PL backend.
"""
import argparse
import base64
from collections import deque
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from urllib.parse import parse_qs,urlsplit
import uuid

import cv2
import numpy as np

from .station_vision import StationVision,CPUVision,pixels


class FrameStore:
    def __init__(self,engine,clock=time.monotonic):
        self.engine=engine;self.clock=clock;self.stream_id=uuid.uuid4().hex
        self.lock=threading.RLock();self.frames=deque(maxlen=4);self.sequence=0
        self.latest_evidence=None;self.error=None;self.faulted=False
        self.startup_comparison=None

    def publish(self,image,captured):
        if (list(pixels(image).shape[1::-1])!=self.engine.size or
            type(captured) not in (int,float) or not math.isfinite(captured)):
            raise ValueError('camera_frame_format_changed')
        raw=image.tobytes()
        frozen=np.frombuffer(raw,dtype=np.uint8).reshape(image.shape)
        with self.lock:
            self.sequence+=1;frame_id=self.sequence
            evidence=None
            if self.engine.baseline_id is not None and not self.faulted:
                try:evidence=self.engine.process(frozen,frame_id,captured)
                except ValueError as exc:
                    self.error=str(exc)
                    if str(exc)!='capture_gap_baseline_invalidated':self.faulted=True
                except Exception as exc:self.faulted=True;self.error=type(exc).__name__+':'+str(exc)
            ok,jpeg=cv2.imencode('.jpg',frozen,[cv2.IMWRITE_JPEG_QUALITY,85])
            if not ok:raise ValueError('jpeg_encode_failed')
            record=dict(id=frame_id,captured=captured,image=frozen,bgr_bytes=raw,
                        jpeg=jpeg.tobytes(),evidence=evidence,png_lock=threading.Lock())
            self.frames.append(record);self.latest_evidence=evidence
            return record

    def latest(self,frame_id=None,complete_window=False,min_frame_id=None):
        with self.lock:
            record=next((r for r in reversed(self.frames)
                if (frame_id is None or r['id']==frame_id) and
                   (min_frame_id is None or r['id']>=min_frame_id) and
                   (not complete_window or r.get('evidence') and r['evidence']['window_complete'] and
                    r['evidence']['baseline_frame_id']==self.engine.baseline_id)),None)
            if record is None or not 0<=self.clock()-record['captured']<2:raise ValueError('fresh_frame_unavailable')
            return record

    def bind(self,body):
        with self.lock:
            if self.faulted:raise ValueError('backend_fault_requires_review')
            if body.get('stream_id')!=self.stream_id or body.get('config_sha256')!=self.engine.config_sha256:
                raise ValueError('baseline_identity_mismatch')
            frame_id=body.get('frame_id')
            if type(frame_id) is not int:raise ValueError('explicit_frame_required')
            frame=self.latest(frame_id)
            if self.clock()-frame['captured']>.5:raise ValueError('baseline_frame_too_old')
            self.engine.bind_baseline(frame['image'],frame_id,frame['captured'])
            self.latest_evidence=None;self.error=None
            return dict(baseline_frame_id=frame_id,baseline_bgr_sha256=self.engine.baseline_sha,
                stream_id=self.stream_id,config_sha256=self.engine.config_sha256,automatic_motion_allowed=False)

    def lossless(self,frame):
        """Cache only immutable pixels; no timestamps, identities or gates cached."""
        with frame['png_lock']:
            if 'png' not in frame:
                ok,png=cv2.imencode('.png',frame['image'])
                if not ok:raise ValueError('png_encode_failed')
                frame['png']=png.tobytes()
                frame['bgr_sha256']=hashlib.sha256(frame['bgr_bytes']).hexdigest()
            return frame['png'],frame['bgr_sha256']

    def bind_snapshot(self,body):
        with self.lock:
            if self.faulted:raise ValueError('backend_fault_requires_review')
            if body.get('stream_id')!=self.stream_id or body.get('config_sha256')!=self.engine.config_sha256:
                raise ValueError('baseline_identity_mismatch')
            if type(body.get('frame_id')) is not int:raise ValueError('explicit_frame_required')
            # Encoding precedes binding. bind() still rechecks the original
            # 500 ms limit after encoding; a failed encode never changes state.
            frame=self.latest(body.get('frame_id'))
            payload,digest=self.lossless(frame)
            result=self.bind(body)
            if result['baseline_bgr_sha256']!=digest:raise ValueError('baseline_snapshot_pixels_changed')
            headers={'X-Frame-Id':frame['id'],'X-Stream-Id':self.stream_id,
                     'X-Frame-Age-Ms':(self.clock()-frame['captured'])*1000,
                     'X-Decoded-Bgr-Sha256':digest,
                     'X-Baseline-Record':base64.b64encode(json.dumps(result,separators=(',',':')).encode()).decode()}
            return payload,headers

    def health(self):
        with self.lock:
            frame=self.frames[-1] if self.frames else None
            age=(self.clock()-frame['captured'])*1000 if frame else None
            return dict(contract='station-vision-v1',stream_id=self.stream_id,
                camera_connected=frame is not None and 0<=age<2000,
                frame_id=frame['id'] if frame else None,frame_age_ms=age,
                baseline_frame_id=self.engine.baseline_id,backend=self.engine.backend.backend,
                assets=getattr(self.engine.backend,'assets',{}),config_sha256=self.engine.config_sha256,
                image_size=self.engine.size,regions=self.engine.regions,threshold=self.engine.threshold,
                window_frames=self.engine.window_frames,camera_error=self.error,faulted=self.faulted,
                startup_comparison=self.startup_comparison,
                capabilities=['region_binding_v1','baseline_snapshot_v1'],
                source='PYNQ /dev/video0',physical_motion=False)

    def reconfigure(self, body):
        """Compare-and-swap rectangles between cycles; invalidate all old evidence."""
        with self.lock:
            if (self.faulted or body.get('stream_id') != self.stream_id or
                body.get('expected_config_sha256') != self.engine.config_sha256 or
                body.get('expected_baseline_frame_id') != self.engine.baseline_id):
                raise ValueError('region_binding_identity_changed')
            binding = body.get('calibration_sha256')
            if not isinstance(binding, str) or len(binding) != 64 or any(c not in '0123456789abcdef' for c in binding):
                raise ValueError('region_calibration_hash_required')
            if type(body.get('frame_id')) is not int: raise ValueError('explicit_region_frame_required')
            frame = self.latest(body['frame_id'])
            if self.clock()-frame['captured'] > .5: raise ValueError('region_binding_frame_too_old')
            previous = self.engine
            candidate = StationVision(previous.backend, body['regions'], previous.size,
                previous.threshold, previous.max_gap_s, previous.window_frames)
            # No PL processing can happen between the policy swap and invalidation.
            # The next explicit baseline bind resets the same backend with new ROIs.
            self.engine = candidate; self.latest_evidence = None
            for row in self.frames: row['evidence'] = None
            self.error = None
            return dict(kind='station-region-binding-v1', stream_id=self.stream_id,
                previous_config_sha256=previous.config_sha256, config_sha256=candidate.config_sha256,
                regions=candidate.regions, calibration_sha256=binding, baseline_invalidated=True,
                automatic_motion_allowed=False)

    def vision(self):
        with self.lock:
            health=self.health();data=self.latest_evidence
            available=bool(data and health['camera_connected'] and not self.faulted and
                self.engine.baseline_id==data['baseline_frame_id'])
            return dict(health,available=available,evidence=data if available else None,
                reason=None if available else self.error or 'explicit_baseline_required',
                cpu_comparison_per_frame=False,automatic_motion_allowed=False,physical_success_verified=False)

    def snapshot_metadata(self, frame):
        """The returned image's cached evidence, not evidence of a newer frame."""
        with self.lock:
            data = frame.get('evidence')
            available = bool(data and not self.faulted and
                             data['baseline_frame_id'] == self.engine.baseline_id)
            return dict(self.health(), frame_id=frame['id'],
                frame_age_ms=(self.clock()-frame['captured'])*1000,
                available=available, evidence=data if available else None,
                automatic_motion_allowed=False, physical_success_verified=False)


def make_server(host,port,token,store):
    if len(token)<32:raise ValueError('token_too_short')
    class Handler(BaseHTTPRequestHandler):
        def setup(self):super().setup();self.request.settimeout(5)
        def log_message(self,*args):pass
        def authorized(self):return hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+token)
        def reply(self,status,value,kind='application/json',headers=None):
            raw=value if isinstance(value,bytes) else json.dumps(value,allow_nan=False).encode()
            try:
                self.send_response(status);self.send_header('Content-Type',kind)
                self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store')
                for k,v in (headers or {}).items():self.send_header(k,str(v))
                self.end_headers();self.wfile.write(raw)
            except (OSError,TimeoutError):pass
        def do_GET(self):
            if not self.authorized():return self.reply(403,dict(error='unauthorized'))
            path=urlsplit(self.path)
            try:
                if path.path=='/health':return self.reply(200,store.health())
                if path.path=='/vision':return self.reply(200,store.vision())
                if path.path in ('/frame.jpg','/snapshot.png'):
                    response_started=time.monotonic()
                    selected=parse_qs(path.query).get('frame_id',[None])[0]
                    complete=parse_qs(path.query).get('window_complete',['0'])[0]=='1'
                    minimum=parse_qs(path.query).get('min_frame_id',[None])[0]
                    frame=store.latest(int(selected) if selected is not None else None,
                        complete_window=complete,min_frame_id=int(minimum) if minimum is not None else None)
                    headers={'X-Frame-Id':frame['id'],'X-Stream-Id':store.stream_id,
                        'X-Frame-Age-Ms':(store.clock()-frame['captured'])*1000}
                    jpeg=path.path=='/frame.jpg'
                    if jpeg:
                        payload=frame['jpeg']
                    else:
                        payload,headers['X-Decoded-Bgr-Sha256']=store.lossless(frame)
                    prepare_ms=(time.monotonic()-response_started)*1000
                    metadata=store.snapshot_metadata(frame)
                    metadata['response_prepare_ms']=prepare_ms
                    headers['X-Station-Metadata']=base64.b64encode(json.dumps(
                        metadata,separators=(',',':')).encode()).decode()
                    return self.reply(200,payload,'image/jpeg' if jpeg else 'image/png',headers)
                return self.reply(404,dict(error='not_found'))
            except (ValueError,TypeError):return self.reply(503,dict(error='fresh_bound_frame_unavailable'))
        def do_POST(self):
            if not self.authorized():return self.reply(403,dict(error='unauthorized'))
            if self.path not in ('/baseline','/regions','/baseline-snapshot.png'):return self.reply(404,dict(error='no_motion_endpoints'))
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=2048 or self.headers.get('Transfer-Encoding'):raise ValueError('invalid_body_size')
                raw=self.rfile.read(size)
                if len(raw)!=size:raise ValueError('incomplete_body')
                data=json.loads(raw)
                if not isinstance(data,dict):raise ValueError('object_body_required')
                if self.path=='/baseline-snapshot.png':
                    payload,headers=store.bind_snapshot(data)
                    return self.reply(200,payload,'image/png',headers)
                return self.reply(200,store.bind(data) if self.path=='/baseline' else store.reconfigure(data))
            except (ValueError,TypeError,KeyError):return self.reply(409,dict(error='baseline_rejected'))
    return ThreadingHTTPServer((host,port),Handler)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--load-overlay',action='store_true')
    parser.add_argument('--host',default='127.0.0.1');parser.add_argument('--port',type=int,default=8783)
    args=parser.parse_args();config=json.loads(args.config.read_text(encoding='utf-8-sig'))
    if config['backend']!='pl_board_local':parser.error('Board service requires explicit PL backend')
    if config.get('synthetic_fixture') is not False:parser.error('Use reviewed station rectangles, not example coordinates')
    StationVision(CPUVision(),config['regions'],config['image_size'],config['threshold'],config['max_gap_s'],config['window_frames'])
    token=os.environ[config['token_env']]
    if len(token)<32:parser.error('Station vision token must contain at least 32 characters')
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'board'))
    from km1_station_vision import StationVisionAccel
    backend=StationVisionAccel((args.config.parent/config['bitfile']).resolve(),config['assets'],
        size=tuple(config['image_size']),load=args.load_overlay)
    engine=StationVision(backend,config['regions'],config['image_size'],config['threshold'],
        config['max_gap_s'],config['window_frames'])
    store=FrameStore(engine);stop=threading.Event()
    try:server=make_server(args.host,args.port,token,store)
    except BaseException:backend.close();raise
    try:
        from .station_vision_selftest import startup_selftest
        comparison=startup_selftest(backend,config)
        Path('startup-comparison-'+str(time.time_ns())+'.json').write_text(json.dumps(comparison,indent=2),encoding='utf-8')
        store.startup_comparison={key:value for key,value in comparison.items() if key!='measurements'}
    except Exception as exc:
        # Stay alive for status/review and retain any AXI-owned buffers. Never
        # start capture, retry the job, reload the overlay or fall back to CPU.
        store.faulted=True;store.error=type(exc).__name__+':'+str(exc)
    def capture():
        cap=cv2.VideoCapture('/dev/video0',cv2.CAP_V4L2)
        try:
            cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'));cap.set(cv2.CAP_PROP_FPS,10)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,config['image_size'][0]);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,config['image_size'][1])
            cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
            if not cap.isOpened():raise RuntimeError('camera_unavailable')
            while not stop.is_set():
                ok,image=cap.read();captured=time.monotonic()
                if not ok:raise RuntimeError('camera_capture_failed')
                store.publish(image,captured)
        except Exception as exc:
            with store.lock:store.faulted=True;store.error=type(exc).__name__+':'+str(exc)
        finally:cap.release()
    worker=None
    if not store.faulted:
        worker=threading.Thread(target=capture,daemon=True);worker.start()
    print(json.dumps(dict(status='station_vision_only',port=args.port,backend=backend.backend,assets=backend.assets)),flush=True)
    try:server.serve_forever()
    finally:
        stop.set()
        if worker is not None:worker.join(timeout=3)
        server.server_close()
        if worker is not None and worker.is_alive():backend.poisoned=True
        backend.close()


if __name__=='__main__':main()
