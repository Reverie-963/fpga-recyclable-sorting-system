"""New board RGB evidence plus explicit PC mask geometry; no legacy PL RPCs."""
import copy
import base64
import json
import time
from urllib.parse import urlsplit
import urllib.request
import urllib.error

import cv2
import numpy as np

from .sorting_io import CameraYoloe, Statistics, digest
from .sorting_loop import need
from .station_vision import CPUVision, StationVision
from .sorting_scene import adjust_exposure_reference


def station_policy(c):
    cfg = c['statistics']
    url = urlsplit(cfg['url'])
    need(cfg['backend'] == 'pl_board_local' and url.scheme == 'http' and
         url.hostname in ('127.0.0.1', 'localhost') and url.path in ('', '/') and
         not any((url.username, url.password, url.query, url.fragment)), 'local_station_bridge_required')
    expected = dict(source=c['source_roi'], static_a=c['layout']['static_rois'][0],
                    static_b=c['layout']['static_rois'][1])
    expected.update({p['bin_id']: p['bin_roi'] for p in c['profiles'].values()})
    need(cfg['regions'] == expected and cfg['image_size'] == c['image_size'], 'station_roi_mismatch')
    need(0 < cfg['threshold'] <= c['rgb_difference_threshold'] and cfg['window_frames'] >= 3,
         'station_visual_policy_relaxed')
    engine = StationVision(CPUVision(), cfg['regions'], cfg['image_size'], cfg['threshold'],
                           cfg['max_gap_s'], cfg['window_frames'])
    return engine.config_sha256


class StationStatistics(Statistics):
    """Mask reductions stay on PC; the new PL RGB gate is mandatory separately."""
    backend = 'pl_board_local'

    def __init__(self, c):
        super().__init__()
        self.c, self.cfg = c, c['statistics']
        need(type(self.cfg.get('baseline_snapshot_enabled',False)) is bool,'invalid_baseline_snapshot_option')
        self.signature = station_policy(c)
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.stream = None
        self.baseline = None
        self.checked(self.request('/health'))

    def request(self, route, body=None):
        request = urllib.request.Request(self.cfg['url'].rstrip('/')+route,
            data=json.dumps(body).encode() if body is not None else None,
            headers={'Content-Type': 'application/json'} if body is not None else {})
        with self.opener.open(request, timeout=self.cfg.get('timeout_s', 3)) as reply:
            raw = reply.read(128001)
        need(len(raw) <= 128000, 'oversized_station_response')
        return json.loads(raw)

    def checked(self, data):
        need(data['contract'] == 'station-vision-v1' and data['backend'] == self.backend and
             data['assets'] == self.cfg['assets'] and data['config_sha256'] == self.signature and
             data['image_size'] == self.c['image_size'] and data['regions'] == self.cfg['regions'] and
             data['threshold'] == self.cfg['threshold'] and data['window_frames'] == self.cfg['window_frames'],
             'station_identity_changed')
        need(data['camera_connected'] and not data['faulted'] and
             0 <= data['frame_age_ms'] < self.c['max_age_ms'], 'station_camera_or_backend_unavailable')
        need(isinstance(data['stream_id'], str) and data['stream_id'] and
             (self.stream is None or data['stream_id'] == self.stream), 'station_stream_changed')
        self.stream = data['stream_id']
        if self.baseline is not None:
            need(data['baseline_frame_id'] == self.baseline['baseline_frame_id'], 'station_baseline_replaced')
        return data

    def provenance(self):
        return dict(backend=self.backend, assets=self.cfg['assets'], stream_id=self.stream,
            config_sha256=self.signature, geometry_backend='pc_numpy',
            semantic_mask_statistics_backend='pc_numpy', rgb_region_statistics_backend=self.backend,
            legacy_dual_core_used=False, silent_cpu_fallback=False)

    def bind_exact_capture(self,row):
        obs=row['observation'];frame=int(obs['source_frame_id'])
        need(row.get('source_encoding')=='png' and obs['source_stream_id']==self.stream and
             0<=time.monotonic()*1000-obs['capture_started_ms']<=500,'station_exact_baseline_frame_expired')
        bgr=digest(row['image'].tobytes())
        need(bgr==row['bgr_sha256'],'station_baseline_pixel_mismatch')
        result=self.request('/baseline',dict(stream_id=self.stream,config_sha256=self.signature,frame_id=frame))
        need(result['stream_id']==self.stream and result['config_sha256']==self.signature and
             result['baseline_frame_id']==frame and result['baseline_bgr_sha256']==bgr,'station_baseline_pixel_mismatch')
        self.baseline=copy.deepcopy(result);self.baseline_row=row;self.exact_baseline_row=row
        return result

    def bind(self, row):
        if getattr(self,'exact_baseline_row',None) is row:
            need(self.baseline['baseline_frame_id']==int(row['observation']['source_frame_id']) and
                 self.baseline['stream_id']==row['observation']['source_stream_id'] and
                 self.baseline['config_sha256']==self.signature,'station_baseline_pixel_mismatch')
            self.checked(self.request('/health'))
            need(digest(row['image'].tobytes())==self.baseline['baseline_bgr_sha256'],'station_baseline_pixel_mismatch')
            need(0<=time.monotonic()*1000-row['observation']['capture_started_ms']<=self.c['max_age_ms'],'station_baseline_frame_expired')
            return copy.deepcopy(self.baseline)
        self.exact_baseline_row=None
        obs = row['observation']
        need(obs['source_stream_id'] == self.stream and
             time.monotonic()*1000-obs['capture_started_ms'] <= self.c['max_age_ms'], 'station_baseline_frame_expired')
        # Select a fresh explicit board frame before transferring its PNG. The
        # board still enforces the original 500 ms limit at the binding point.
        current = self.checked(self.request('/health'))
        body=dict(stream_id=self.stream,config_sha256=self.signature,frame_id=current['frame_id'])
        combined=(self.cfg.get('baseline_snapshot_enabled',False) and
                  'baseline_snapshot_v1' in current.get('capabilities',[]))
        if combined:
            request=urllib.request.Request(self.cfg['url'].rstrip('/')+'/baseline-snapshot.png',
                data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            started=time.monotonic()*1000
            # An unknown POST result is an error. Never retry the mutation or
            # switch to the old POST/GET path after sending this request.
            with self.opener.open(request,timeout=self.cfg.get('timeout_s',3)) as reply:
                raw,headers=reply.read(8500001),reply.headers
            encoded=headers.get('X-Baseline-Record','')
            need(isinstance(encoded,str) and 0<len(encoded)<=4096,'station_baseline_record_missing')
            result=json.loads(base64.b64decode(encoded,validate=True))
        else:
            result=self.request('/baseline',body)
            url=self.cfg['url'].rstrip('/')+'/snapshot.png?frame_id='+str(current['frame_id'])
            started=time.monotonic()*1000
            with self.opener.open(url, timeout=self.cfg.get('timeout_s', 3)) as reply:
                raw, headers = reply.read(8500001), reply.headers
        need(raw.startswith(b'\x89PNG') and len(raw) <= 8500000, 'station_baseline_snapshot_invalid')
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        need(image is not None and list(image.shape[1::-1]) == self.c['image_size'], 'station_baseline_size_changed')
        bgr = digest(image.tobytes())
        need(result['stream_id'] == self.stream and result['config_sha256'] == self.signature and
             result['baseline_frame_id'] == current['frame_id'] and
             headers['X-Stream-Id'] == self.stream and int(headers['X-Frame-Id']) == current['frame_id'] and
             headers['X-Decoded-Bgr-Sha256'] == bgr == result['baseline_bgr_sha256'], 'station_baseline_pixel_mismatch')
        reference, photometry = adjust_exposure_reference(row['image'], image,
            dict(self.c['layout'], image_size=self.c['image_size']))
        changed = np.abs(image.astype(np.int16)-reference.astype(np.int16)).sum(axis=2) > self.c['rgb_difference_threshold']
        for name, (x0,y0,x1,y1) in self.cfg['regions'].items():
            need(float(changed[y0:y1,x0:x1].mean()) <= self.c['max_scene_changed_fraction'],
                 'station_precommit_scene_changed:'+name)
        self.baseline_row = dict(image=image, source_encoding='png', source_bytes_sha256=digest(raw),
            jpeg_sha256=None, bgr_sha256=bgr, observation=dict(camera_id=self.c['camera_id'],
            source_stream_id=self.stream, source_frame_id=str(current['frame_id']),
            capture_started_ms=started-float(headers['X-Frame-Age-Ms']),
            image_sha256=digest(raw), photometry=photometry, evidence_type='exact_board_motion_baseline'))
        self.baseline = copy.deepcopy(result)
        return result

    def evidence(self, row):
        evidence = row.get('station_vision')
        need(self.baseline is not None and isinstance(evidence, dict), 'station_rgb_evidence_missing')
        need(evidence['backend'] == self.backend and evidence['config_sha256'] == self.signature and
             evidence['frame_id'] == int(row['observation']['source_frame_id']) and
             row['observation']['source_stream_id'] == self.stream and
             evidence['frame_bgr_sha256'] == row['bgr_sha256'] and
             evidence['baseline_frame_id'] == self.baseline['baseline_frame_id'] and
             evidence['baseline_bgr_sha256'] == self.baseline['baseline_bgr_sha256'],
             'station_cycle_evidence_changed')
        need(time.monotonic()*1000-row['observation']['capture_started_ms'] <= self.c['max_age_ms'],
             'station_cycle_evidence_expired')
        need(set(evidence['regions']) == set(self.cfg['regions']), 'station_region_set_changed')
        for name, rect in self.cfg['regions'].items():
            value = evidence['regions'][name]
            area = (rect[2]-rect[0])*(rect[3]-rect[1])
            need(all(type(value[k]) is int for k in ('changed_px', 'persistent_px', 'area_px', 'delta_sum')) and
                 value['area_px'] == area and 0 <= value['persistent_px'] <= value['changed_px'] <= area and
                 0 <= value['delta_sum'] <= 765*area, 'station_region_statistics_invalid')
        return evidence

    def verify_cycle(self, after, bin_id, source, save_extra):
        first = int(after[0]['observation']['source_frame_id'])
        rows, extra = list(after), 0
        deadline = time.monotonic()+self.c['frame_timeout_s']
        while True:
            evidence = [self.evidence(row) for row in rows]
            windows = [e for e in evidence if e['window_complete'] and
                       e['consecutive_frames'] == self.cfg['window_frames'] and
                       e['frame_id']-self.cfg['window_frames']+1 >= first]
            if windows:
                break
            need(time.monotonic() < deadline, 'station_post_retreat_window_incomplete')
            row = source.raw(max(.05, deadline-time.monotonic()))
            save_extra(extra, row)
            rows.append(row)
            extra += 1
        for e in evidence:
            regions = e['regions']
            need(regions['source']['changed_px'] >= self.c['min_bin_gain_px'] and
                 regions[bin_id]['changed_px'] >= self.c['min_bin_gain_px'], 'station_source_or_bin_change_missing')
            for name in regions:
                if name not in ('source', bin_id):
                    limit = self.c['max_other_bin_changed_fraction'] if name.startswith('bin_') else self.c['max_scene_changed_fraction']
                    need(regions[name]['changed_px']/regions[name]['area_px'] <= limit,
                         'station_unexpected_region_change:'+name)
        need(any(e['regions'][bin_id]['persistent_px'] >= self.c['min_bin_gain_px'] and
                 e['regions']['source']['persistent_px'] >= self.c['min_bin_gain_px'] for e in windows),
             'station_change_not_persistent')
        return dict(status='board_rgb_evidence_pass', provenance=self.provenance(),
                    baseline=self.baseline, frames=evidence, semantic_verified=False,
                    independently_validated=False, physical_success_verified=False)


class StationCamera(CameraYoloe):
    """Infer lossless pixels of the exact frame that produced board evidence."""
    def __init__(self, c, detector, statistics, lossless=True):
        super().__init__(c, detector)
        self.statistics = statistics
        need(lossless or detector is None, 'semantic_camera_requires_lossless_pixels')
        self.lossless = lossless
        self.post_retreat_count = None
        self.first_post_retreat_frame = None
        need(c['camera_url'].rsplit('/', 1)[0].rstrip('/') == c['statistics']['url'].rstrip('/'),
             'station_camera_bridge_mismatch')

    def before_semantic(self,row):
        if not getattr(self.statistics,'bind_next_capture',False):return
        self.statistics.bind_next_capture=False
        # A stale capture takes the original fresh-baseline path later. An
        # unknown POST response is never retried or hidden by fallback.
        age=time.monotonic()*1000-row['observation']['capture_started_ms']
        if not 0<=age<=450:return
        self.statistics.bind_exact_capture(row)

    def after(self, stamp):
        super().after(stamp)
        self.post_retreat_count = 0
        self.first_post_retreat_frame = None

    def _raw(self, deadline):
        while time.monotonic() < deadline:
            started = time.monotonic()*1000
            complete = self.post_retreat_count is not None and self.post_retreat_count >= 2
            url = self.statistics.cfg['url'].rstrip('/')+('/snapshot.png' if self.lossless else '/frame.jpg')
            if complete:
                minimum = max(self.first_post_retreat_frame+self.statistics.cfg['window_frames']-1,
                              self.previous+1)
                url += '?window_complete=1&min_frame_id='+str(minimum)
            try:
                with self.opener.open(url, timeout=max(.05, min(1.5, deadline-time.monotonic()))) as reply:
                    raw = reply.read(8500001)
                    headers = reply.headers
            except urllib.error.HTTPError as exc:
                if complete and exc.code == 503:
                    exc.close()
                    time.sleep(.02)
                    continue
                raise
            metadata = headers.get('X-Station-Metadata', '')
            need(metadata and len(metadata) <= 16000, 'station_snapshot_metadata_missing')
            data = self.statistics.checked(json.loads(base64.b64decode(metadata, validate=True)))
            frame = data['frame_id']
            prepare = data.get('response_prepare_ms', 0)
            need(type(prepare) in (int, float) and np.isfinite(prepare) and
                 0 <= prepare <= time.monotonic()*1000-started+1, 'station_response_timing_invalid')
            # Server work precedes its age sample. Remove only this measured
            # work from the RTT bound; no host/board clock synchronization.
            stamp = started+prepare-data['frame_age_ms']
            need(type(frame) is int and (self.previous is None or frame >= self.previous), 'station_frame_reordered')
            if frame == self.previous or stamp <= self.barrier or stamp-self.previous_stamp < self.c['min_frame_gap_ms']:
                time.sleep(.02)
                continue
            need(len(raw) <= 8500000 and raw.startswith(b'\x89PNG' if self.lossless else b'\xff\xd8') and
                 headers['X-Stream-Id'] == data['stream_id'] and int(headers['X-Frame-Id']) == frame,
                 'station_snapshot_identity_mismatch')
            image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            need(image is not None and list(image.shape[1::-1]) == self.c['image_size'], 'station_snapshot_size')
            bgr = digest(image.tobytes())
            need(not self.lossless or bgr == headers['X-Decoded-Bgr-Sha256'], 'station_snapshot_pixels_changed')
            evidence = data.get('evidence')
            need(not complete or isinstance(evidence, dict) and evidence['window_complete'] and
                 evidence['consecutive_frames'] == self.statistics.cfg['window_frames'] and
                 frame >= minimum, 'station_complete_window_unavailable')
            if evidence is not None:
                need(data.get('available') and evidence['frame_id'] == frame and
                     (not self.lossless or evidence['frame_bgr_sha256'] == bgr) and
                     evidence['config_sha256'] == self.statistics.signature,
                     'station_snapshot_evidence_mismatch')
            need(time.monotonic() <= deadline, 'station_snapshot_deadline_exceeded')
            self.previous, self.previous_stamp, self.stream = frame, stamp, data['stream_id']
            self.sequence += 1
            if self.post_retreat_count is not None:
                self.first_post_retreat_frame = self.first_post_retreat_frame or frame
                self.post_retreat_count += 1
            encoding = 'png' if self.lossless else 'jpeg'
            bound_evidence = evidence if self.lossless else None
            return dict(image=image, source_png_bytes=raw if self.lossless else None, source_encoding=encoding, source_bytes_sha256=digest(raw),
                jpeg_sha256=None if self.lossless else digest(raw), bgr_sha256=bgr, station_vision=bound_evidence,
                observation=dict(camera_id=self.c['camera_id'], frame_id=self.sequence,
                    capture_started_ms=stamp, source_stream_id=self.stream, source_frame_id=str(frame),
                    image_sha256=digest(raw), source_encoding=encoding, station_vision=bound_evidence,
                    marker_guard_only=not self.lossless))
        raise TimeoutError('No fresh frame from board vision')
