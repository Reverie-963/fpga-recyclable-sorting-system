"""Bounded camera/YOLOE, statistics and serial adapters for sorting_loop."""
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import time
from urllib.parse import urlsplit
import urllib.request

import cv2
import numpy as np

from .sorting_loop import need


def digest(data):
    return hashlib.sha256(data).hexdigest()


def image_stats(mask, job):
    ys, xs = np.nonzero(mask)
    x, y = xs.astype(np.uint64), ys.astype(np.uint64)
    return dict(m00=len(xs), m10=int(x.sum()), m01=int(y.sum()),
                m20=int((x*x).sum()), m02=int((y*y).sum()), m11=int((x*y).sum()),
                **bbox(mask), job_id=job, status=1)


def bbox(mask):
    ys, xs = np.nonzero(mask)
    return dict(min_row=int(ys.min()) if len(ys) else 0,
                max_row=int(ys.max()) if len(ys) else 0,
                min_col=int(xs.min()) if len(xs) else 0,
                max_col=int(xs.max()) if len(xs) else 0)


class Statistics:
    backend = 'cpu_reference'

    def __init__(self):
        self.job = 0

    def next_job(self):
        self.job += 1
        need(self.job < 2**32, 'statistics_job_ids_exhausted')
        return self.job

    def geometry(self, mask):
        job = self.next_job()
        if (isinstance(mask, np.ndarray) and mask.dtype == np.bool_ and mask.ndim == 2 and
                1 <= mask.shape[0] <= 480 and 1 <= mask.shape[1] <= 640):
            from .mask_geometry_cpu import image_stats_u8
            return image_stats_u8(mask, job)
        return image_stats(mask, job)

    def visual(self, a, b):
        return dict(inter=int((a & b).sum()), erased=int((a & ~b).sum()),
                    beyond=int((~a & b).sum()), union=int((a | b).sum()),
                    **bbox(a ^ b), job_id=self.next_job(), status=1)

    def provenance(self):
        return dict(backend=self.backend, hardware_verified=False)


def pack_mask(mask):
    return base64.b64encode(np.packbits(mask.reshape(-1), bitorder='little').tobytes()).decode('ascii')


def unpack_mask(encoded, size):
    need(isinstance(size, list) and len(size) == 2 and
         all(type(v) is int for v in size) and 1 <= size[0] <= 640 and 1 <= size[1] <= 480,
         'invalid_mask_size')
    raw = base64.b64decode(encoded, validate=True)
    count = size[0]*size[1]
    need(len(raw) == (count+7)//8, 'invalid_packed_mask_length')
    bits = np.unpackbits(np.frombuffer(raw, np.uint8), bitorder='little')
    need(not bits[count:].any(), 'nonzero_mask_padding')
    return bits[:count].reshape(size[1], size[0]).astype(bool)


class PLStatistics(Statistics):
    backend = 'pl_http'

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        url = urlsplit(cfg['url'])
        need(url.scheme == 'http' and url.hostname and not url.username and not url.password
             and not url.query and not url.fragment and url.path in ('', '/'), 'invalid_statistics_url')
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.token = os.environ[cfg['token_env']]
        need(len(self.token) >= 32, 'statistics_token_too_short')
        health = self.request('health')
        need(health['contract'] == 'sorting-statistics-v1' and health['assets'] == cfg['assets'],
             'pl_assets_or_contract_mismatch')
        self.session = health['session_id']

    def request(self, route, data=None):
        headers = {'Authorization': 'Bearer ' + self.token}
        if data is not None:
            headers['Content-Type'] = 'application/json'
        request = urllib.request.Request(self.cfg['url'].rstrip('/') + '/' + route,
                  data=json.dumps(data).encode() if data is not None else None, headers=headers)
        with self.opener.open(request, timeout=self.cfg['timeout_s']) as reply:
            raw = reply.read(100001)
        need(len(raw) <= 100000, 'oversized_statistics_reply')
        return json.loads(raw)

    def checked(self, role, masks, expected):
        nonce = os.urandom(16).hex()
        result = self.request('run', dict(role=role, session_id=self.session, request_id=nonce,
            image_size=[masks[0].shape[1], masks[0].shape[0]], job_id=expected['job_id'],
            masks=[pack_mask(m) for m in masks]))
        need(result['session_id'] == self.session and result['assets'] == self.cfg['assets']
             and result['request_id'] == nonce, 'statistics_response_identity_mismatch')
        from .static_mask_contract import compare_snapshot
        compare_snapshot(result['stats'], expected)
        output = masks[0] if role == 'geometry' else masks[0] ^ masks[1]
        need(result['output_sha256'] == digest(output.astype(np.uint8).tobytes()), 'pl_output_mismatch')
        return result['stats']

    def geometry(self, mask):
        return self.checked('geometry', [mask], super().geometry(mask))

    def visual(self, a, b):
        return self.checked('visual', [a, b], super().visual(a, b))

    def provenance(self):
        return dict(backend=self.backend, assets=self.cfg['assets'], session_id=self.session,
                    each_result_cpu_compared=True)


class CameraYoloe:
    """Pull fresh camera JPEGs and infer exactly those bytes; never use overlays.

    Clock values are controller-local receipt estimates including bridge age,
    not exposure hardware timestamps. A stream restart stops this session.
    """
    def __init__(self, c, detector):
        self.c, self.detector = c, detector
        url = urlsplit(c['camera_url'])
        need(url.scheme == 'http' and url.hostname in ('127.0.0.1', 'localhost') and
             url.path == '/frame.jpg' and not url.username and not url.password and
             not url.query and not url.fragment, 'use_local_camera_bridge')
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.previous = None
        self.previous_stamp = -1
        self.barrier = -1
        self.sequence = 0
        self.stream = None

    def after(self, stamp):
        self.barrier = stamp

    def next(self, timeout):
        deadline = time.monotonic()+timeout
        row = self._raw(deadline)
        before_semantic=getattr(self,'before_semantic',None)
        if callable(before_semantic):before_semantic(row)
        predict_frame=getattr(self.detector,'predict_frame',None)
        observation = (predict_frame(row,prompt_mode=self.c['prompt_mode']) if predict_frame else
                       self.detector.predict(row['image'], prompt_mode=self.c['prompt_mode']))
        need(time.monotonic() <= deadline, 'inference_deadline_exceeded')
        # Preserve the detector's verified PNG input hash separately from the
        # camera JPEG hash. These encodings contain the same decoded pixels.
        observation['model_input_image_sha256'] = observation.get('image_sha256')
        observation.update(row['observation'])
        row['observation'] = observation
        return row

    def raw(self, timeout):
        """Fresh original camera pixels and identity, without model inference."""
        return self._raw(time.monotonic()+timeout)

    def _raw(self, deadline):
        while time.monotonic() < deadline:
            with self.opener.open(self.c['camera_url'], timeout=min(1.5, max(.05, deadline-time.monotonic()))) as reply:
                raw = reply.read(8500001)
                stream, source_id = reply.headers.get('X-Stream-Id'), reply.headers.get('X-Frame-Id')
                age = float(reply.headers['X-Frame-Age-Ms'])
            now = time.monotonic()*1000
            need(stream and source_id and source_id.isdigit() and math.isfinite(age) and
                 0 <= age <= self.c['max_age_ms'] and len(raw) <= 8500000, 'invalid_camera_packet')
            need(self.stream is None or stream == self.stream, 'camera_stream_restarted')
            if self.previous is not None:
                need(int(source_id) >= self.previous, 'upstream_frame_reordered')
            stamp = now-age
            if (self.previous == int(source_id) or stamp <= self.barrier or
                    stamp-self.previous_stamp < self.c['min_frame_gap_ms']):
                time.sleep(.03)
                continue
            image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            need(image is not None and list(image.shape[:2][::-1]) == self.c['image_size'], 'camera_size_changed')
            self.previous, self.previous_stamp, self.stream = int(source_id), stamp, stream
            self.sequence += 1
            observation = dict(camera_id=self.c['camera_id'], frame_id=self.sequence,
                capture_started_ms=stamp, source_stream_id=stream, source_frame_id=source_id,
                image_sha256=digest(raw))
            return dict(image=image, observation=observation, jpeg_sha256=digest(raw),
                        bgr_sha256=digest(image.tobytes()))
        raise TimeoutError('No new post-barrier camera frame')


class HTTPYoloe:
    """Reuse the deployed GPU model; submit the exact decoded pixels as PNG.

    The existing preview /detect endpoint returns the input byte hash. Binding
    that hash plus model/runtime/prompt identity prevents using an overlay or
    a result from another image as a control observation. No inference retry.
    """
    def __init__(self, c, model_config):
        from .yoloe_inference import read_config, config_hash
        self.c = c
        url = urlsplit(c['detector_url'])
        need(url.scheme == 'http' and url.hostname in ('127.0.0.1', 'localhost') and
             url.path in ('', '/') and not url.username and not url.password and
             not url.query and not url.fragment, 'use_local_yoloe_service')
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        health = self.request('health')
        need(health['model_loaded'] is True, 'deployed_yoloe_not_loaded')
        self.family = c.get('model_family', 'yoloe')
        # Preserve the default YOLOE-only guard. A trained model is an explicit
        # configuration migration, never accepted through the old fingerprint.
        if self.family == 'yoloe':
            need(health.get('model_family', 'yoloe') == 'yoloe' and
                 'supervised' not in health.get('model_versions', {}), 'detector_service_is_not_yoloe')
        self.model_ids = health['model_versions']
        self.execution = health['execution']
        need(self.family in ('yoloe', 'yolo11s', 'yolov8s'), 'unsupported_control_model_family')
        need(c['prompt_mode'] in self.model_ids, 'deployed_model_or_prompt_mode_changed')
        if self.family in ('yolo11s', 'yolov8s'):
            need(health.get('model_family') == self.family and c['prompt_mode'] == 'supervised',
                 'deployed_model_family_mismatch')
            self.runtime_id = health['runtime_id']
        else:
            need(c['prompt_mode'] in ('text','visual'), 'deployed_model_family_mismatch')
            self.runtime_id = config_hash(read_config(model_config))
        need(self.execution['device'] == c['device'] and
             self.execution['precision'] == ('fp16' if c['half'] else 'fp32'), 'deployed_yoloe_execution_changed')

    def request(self, route, data=None):
        body = json.dumps(data).encode() if data is not None else None
        request = urllib.request.Request(self.c['detector_url'].rstrip('/')+'/'+route,
            data=body, headers={'Content-Type':'application/json'} if body is not None else {})
        with self.opener.open(request, timeout=self.c['frame_timeout_s']) as reply:
            raw = reply.read(15000001)
        need(len(raw) <= 15000000, 'oversized_yoloe_reply')
        return json.loads(raw)

    def execution_id(self, mode):
        from .yoloe_inference import config_hash
        if self.family in ('yolo11s', 'yolov8s'):
            need(mode == 'supervised', 'trained_model_has_no_prompt_modes')
            return self.runtime_id
        return config_hash(dict(config=self.runtime_id, prompt_mode=mode,
                                model=self.model_ids[mode], execution=self.execution))

    def predict(self, image, prompt_mode):
        ok, encoded = cv2.imencode('.png', image)
        need(ok, 'cannot_encode_control_frame')
        raw = encoded.tobytes()
        result = self.request('detect', dict(image_base64=base64.b64encode(raw).decode(), prompt_mode=prompt_mode))
        need(result['image_sha256'] == digest(raw), 'yoloe_input_image_hash_mismatch')
        need(result['model_version'] == self.c['model_version'] and
             result['runtime_id'] == self.c['runtime_id'] and result['prompt_mode'] == prompt_mode and
             result['image_size'] == [image.shape[1], image.shape[0]], 'yoloe_result_identity_mismatch')
        return result


class Motion:
    def __init__(self, arm, stop_file):
        self.arm, self.stop_file = arm, Path(stop_file)
        self.connection = None
        self.echo = None

    def check_stop(self):
        need(not self.stop_file.exists(), 'operator_stop_requested')

    def open(self, plan):
        self.check_stop()
        from serial.tools import list_ports
        from .arm_transport import ArmConnection
        identity = self.arm['usb_identity']
        matches = [p for p in list_ports.comports() if p.device == self.arm['port'] and
                   p.vid == identity['vid'] and p.pid == identity['pid'] and
                   (identity.get('serial_number') is None or p.serial_number == identity['serial_number'])]
        need(len(matches) == 1, 'arm_usb_identity_mismatch')
        self.connection = ArmConnection(self.arm)
        for _, data, duration in plan:
            self.connection.validate(data, duration)
        self.connection.open()

    def send(self, data, duration, guard=None):
        self.check_stop()
        if self.connection.mode == 'direct':
            self.connection.port.reset_input_buffer()
            self.echo = data
            poll_timeout=self.arm.get('echo_poll_timeout_s',.05)
            need(type(poll_timeout) in (int,float) and .01<=poll_timeout<=.05,'invalid_echo_poll_timeout')
            self.connection.port.timeout = poll_timeout
        def final_guard():
            self.check_stop()
            if guard:
                guard()
        # V3 invokes this after ARM and immediately before MOVE; direct mode
        # invokes it immediately before the UART write.
        self.connection.send(data, duration, guard=final_guard)

    def wait(self, seconds):
        started = time.monotonic()
        deadline = started+seconds
        received = bytearray()
        payload = self.echo
        completed = False
        try:
            while time.monotonic() < deadline:
                self.check_stop()
                if self.echo is not None:
                    received.extend(self.connection.port.read(4096))
                    need(len(received) <= 65536, 'excessive_serial_response')
                else:
                    time.sleep(min(.05, max(0, deadline-time.monotonic())))
            completed = True
        finally:
            self.last_receipt = dict(sent_hex=payload.hex() if payload is not None else None,
                received_hex=received.hex(), echo_matched=payload in received if payload is not None else None,
                completed_movement_wait=completed, waited_s=time.monotonic()-started, servo_ack_claimed=False)
        if self.echo is not None:
            need(self.echo in received, 'missing_serial_echo_execution_unknown')
        self.echo = None

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None
