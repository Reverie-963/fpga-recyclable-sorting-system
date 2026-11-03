"""Guided camera-only calibration. Reuses the existing loopback JPEG bridge."""
import argparse
from collections import deque
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
import time
from urllib.parse import urlsplit
import urllib.request

import cv2
import numpy as np

from .station_calibration import calibrate_intrinsics, read_json, write_json
from .station_geometry import (camera_parameters, detect_markers, fingerprint, fit_world_registration,
    localize_records, need, registration_ray_point)
from .station_relocalization import FixedMarkerTracker


def inspect_board(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ok, points = cv2.findChessboardCorners(gray, (5, 4), cv2.CALIB_CB_ADAPTIVE_THRESH |
        cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK)
    if not ok:
        return None, '请把整张棋盘放进画面，露出全部20个内部交点'
    points = cv2.cornerSubPix(gray, points, (5, 5), (-1, -1),
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, .01)).reshape(20, 2)
    x, y, w, h = cv2.boundingRect(points.astype(np.float32))
    if w < 60 or h < 40:
        return None, '棋盘太小，请靠近相机一些'
    if x < 5 or y < 5 or x+w >= image.shape[1]-5 or y+h >= image.shape[0]-5:
        return None, '棋盘靠边，请向画面内移动一点'
    sharpness = float(cv2.Laplacian(gray[y:y+h, x:x+w], cv2.CV_64F).var())
    if sharpness < 60:
        return None, '画面模糊，请停稳或改善光线'
    return points, ''


def corner_distance(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return min(float(np.linalg.norm(a-b, axis=1).mean()), float(np.linalg.norm(a-b[::-1], axis=1).mean()))


def stable_boards(rows):
    return len(rows) >= 3 and max(corner_distance(rows[-1]['corners'], r['corners']) for r in rows) <= 1.2


def novel_board(points, previous):
    return not previous or min(corner_distance(points, p) for p in previous) >= 10


def registration_from_frames(rows, camera, station, reverse=False):
    pose = localize_records([dict(image_sha256=r['sha'], detections=r['markers']) for r in rows], camera, station,
                            stability_mode='fixed_marker_task_space')
    card = np.array([[(col-2)*20., (2-row)*20.] for row in range(1, 5) for col in range(5)])
    worlds = []
    for row in rows:
        points = row['corners'][::-1] if reverse else row['corners']
        worlds.append([registration_ray_point(p, 0, pose, camera, station)['world_xyz_mm'][:2] for p in points])
    worlds = np.asarray(worlds)
    need(np.max(np.linalg.norm(worlds-worlds[-1], axis=2)) <= 1, 'registration_card_not_stable')
    # Hold out eight actual positions, not a second copy of training points.
    check = np.array([i in (1, 3, 6, 8, 11, 13, 16, 18) for i in range(20)])
    pairs = dict(world_frame_id=station['world_frame_id'], world_xy_mm=worlds[-1, ~check].tolist(),
        card_xy_mm=card[~check].tolist(), check_world_xy_mm=worlds[-1, check].tolist(), check_card_xy_mm=card[check].tolist())
    registration = fit_world_registration(pairs)
    registration.update(method='simultaneous_markers_and_restored_legacy_chessboard',
        operator_restored_legacy_card=True, card_orientation_reversed=reverse, image_sha256=[r['sha'] for r in rows],
        pose_stability_quality=pose['stability_quality'])
    return pose, registration, pairs


class GuidedSession:
    def __init__(self, root, history, station, camera_url, camera_id='pynq-c270', lens_profile='c270-fixed-640x480', resume=False):
        url = urlsplit(camera_url)
        need(url.scheme == 'http' and url.hostname in ('127.0.0.1', 'localhost') and url.path == '/frame.jpg'
             and not url.username and not url.password and not url.query and not url.fragment, 'loopback_camera_bridge_required')
        self.root = Path(root)
        if resume:
            need(self.root.is_dir(), 'resume_session_missing')
        else:
            self.root.mkdir(parents=True, exist_ok=False)
        self.history_root = Path(history).resolve(); self.history = read_json(self.history_root/'bundle.json')
        self.station, self.camera_url = station, camera_url
        self.lock = threading.RLock(); self.rows = deque(maxlen=3); self.saved_corners = []
        self.fit, self.checks, self.seen_hashes = [], [], set()
        self.frame = None; self.preview = None; self.stream = None; self.frame_id = -1
        self.auto = False; self.stage = 'prepare'; self.note = '准备好棋盘后，点击开始自动拍照'
        self.error = None; self.reverse = False; self.camera = None; self.registration = None
        self.relocator = None; self.relocation_saved_generation = 0; self.verification = None
        self.goal_fit = 10; self.goal_check = 3; self.busy = False; self.closed = threading.Event()
        self.manifest = dict(camera_id=camera_id, lens_profile_id=lens_profile, image_size=[640, 480],
            pattern_inner_corners=[5, 4], square_mm=20, fit_images=self.fit, check_images=self.checks,
            minimum_fit_images=10, require_leave_one_out_stability=True, calibration_model='brown4')
        identity = dict(kind='guided-camera-calibration-v1', camera_url=camera_url,
            history_bundle_sha256=fingerprint(self.history), station_sha256=fingerprint(station),
            automatic_motion_allowed=False, capture_starts_paused=True)
        if resume:
            old = read_json(self.root/'session.json')
            need(all(old.get(k) == identity[k] for k in ('kind', 'camera_url', 'history_bundle_sha256', 'station_sha256')),
                 'resume_session_identity_changed')
            for meta_path in sorted((self.root/'images').glob('*.json')):
                meta = read_json(meta_path); image_path = meta_path.with_suffix('.jpg')
                raw = image_path.read_bytes()
                need(hashlib.sha256(raw).hexdigest() == meta['image_sha256'] and
                     meta['lens_profile_id'] == lens_profile, 'resume_photo_or_lens_changed')
                need(meta['image_sha256'] not in self.seen_hashes, 'resume_duplicate_photo')
                need(meta['split'] in ('fit', 'check'), 'resume_split_invalid')
                name = 'images/'+image_path.name
                (self.fit if meta['split'] == 'fit' else self.checks).append(name)
                self.seen_hashes.add(meta['image_sha256']); self.saved_corners.append(np.asarray(meta['corners_px'], np.float32))
            for line in (self.root/'events.jsonl').read_text(encoding='utf-8').splitlines():
                event = json.loads(line)
                if event['event'] != 'historical_seed_imported': continue
                name = event['file']; path = (self.root/name).resolve()
                need(path.is_relative_to(self.root.resolve()), 'resume_history_path_invalid')
                raw = path.read_bytes(); sha = hashlib.sha256(raw).hexdigest()
                need(sha == event['source_sha256'] and event['same_lens_settings_confirmed_by_operator'] is True,
                     'resume_historical_seed_changed')
                if sha in self.seen_hashes: continue
                points, _ = inspect_board(cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR))
                need(points is not None, 'resume_historical_board_not_detected')
                self.fit.append(name); self.seen_hashes.add(sha); self.saved_corners.append(points)
            manifests = sorted(self.root.glob('intrinsics-manifest-*.json'), key=lambda p:p.stat().st_mtime)
            if manifests:
                order = read_json(manifests[-1])
                for key, files in (('fit_images', self.fit), ('check_images', self.checks)):
                    previous = [name for name in order[key] if name in files]
                    files[:] = previous+[name for name in files if name not in previous]
            self.goal_fit = max(10, len(self.fit))
            self.stage = 'review_existing'; self.note = '已恢复全部照片。先重新计算现有照片，不继续要求补拍'
            if (self.root/'camera.json').exists():
                self.camera = read_json(self.root/'camera.json'); self.stage = 'place_markers'
                camera_parameters(self.camera)
                need(self.camera['camera_id'] == camera_id and self.camera['lens_profile_id'] == lens_profile,
                     'resume_camera_identity_changed')
                self.note = '镜头计算已完成。接下来登记固定标记与旧棋盘坐标'
            if (self.root/'world-registration.json').exists():
                self.registration = read_json(self.root/'world-registration.json'); self.stage = 'registered'
                self.note = '已恢复坐标登记，可以进行换机位检查'
                for path in sorted(self.root.glob('verify-*-registration.json'), reverse=True):
                    candidate = read_json(path)
                    if candidate.get('previous_registration_sha256') == fingerprint(self.registration):
                        self.verification = candidate; self.stage = 'verified'
                        self.note = '换机位检查已通过；现在只看固定标记自动恢复机位，无需旧棋盘'
                        break
                self._start_relocator()
            self._event('resumed', fit_images=len(self.fit), check_images=len(self.checks), calibration_model='brown4')
        else:
            (self.root/'images').mkdir()
            write_json(self.root/'session.json', identity)
            self._event('created')

    def _event(self, name, **extra):
        with (self.root/'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(event=name, time=time.time(), **extra), ensure_ascii=False)+'\n')

    def _start_relocator(self):
        profiles = {}
        for name, data in self.history['classes'].items():
            path = (self.history_root/data['profile_file']).resolve()
            need(path.is_relative_to(self.history_root), 'profile_path_outside_history')
            profiles[name] = read_json(path)
        self.relocator = FixedMarkerTracker(self.camera, self.station, self.registration, profiles)
        self.relocation_saved_generation = 0

    def localization(self):
        with self.lock:
            return self.relocator.snapshot() if self.relocator else dict(ready=False,
                reason='registration_required', automatic_motion_allowed=False, production_integrated=False)

    def _track_markers(self, row):
        if self.relocator is None: return
        self.relocator.update(row)
        result = self.relocator.snapshot()
        if result['ready'] and result['camera_generation'] != self.relocation_saved_generation:
            suffix = 'relocate-'+str(time.time_ns())
            for index, original in enumerate(self.relocator.rows):
                (self.root/(suffix+f'-{index}.jpg')).write_bytes(original['raw'])
            write_json(self.root/(suffix+'.json'), result)
            self.relocation_saved_generation = result['camera_generation']
            self._event('marker_only_localized', file=suffix+'.json', frame_id=row['frame_id'])

    def state(self):
        with self.lock:
            return dict(stage=self.stage, note=self.note, error=self.error, auto=self.auto, busy=self.busy,
                fit=len(self.fit), fit_goal=self.goal_fit, checks=len(self.checks), check_goal=self.goal_check,
                history_used=sum('/history-' in name for name in self.fit),
                seed_candidates=len(self.history['intrinsics_seeds']), camera_ready=self.frame is not None,
                lens_ready=self.camera is not None, registered=self.registration is not None,
                classes={k: dict(height_mm=v['parameters']['height_mm'], commands=v['historical_commands'])
                         for k, v in self.history['classes'].items()},
                automatic_motion_allowed=False, reverse=self.reverse,
                camera_error=self.error, frame_id=self.frame_id,
                localization=self.localization(),
                verification=None if self.verification is None else dict(
                    passed=True, move_check_max_mm=self.verification['move_check_max_mm']),
                calibration_quality=None if self.camera is None else {k:self.camera.get(k) for k in
                    ('fit_rms_px', 'check_rms_px', 'check_max_px', 'leave_one_out_stability', 'calibration_model')})

    def _capture(self, row, split):
        need(row['sha'] not in self.seen_hashes, 'duplicate_capture')
        name = f'images/{split}-{len(self.fit)+len(self.checks):03}.jpg'
        (self.root/name).write_bytes(row['raw'])
        write_json((self.root/name).with_suffix('.json'), dict(image_sha256=row['sha'],
            decoded_bgr_sha256=hashlib.sha256(row['image'].tobytes()).hexdigest(),
            source_frame_id=row['frame_id'], source_stream_id=row['stream'], source_age_ms=row['age'],
            split=split, corners_px=row['corners'].tolist(), lens_profile_id=self.manifest['lens_profile_id']))
        (self.fit if split == 'fit' else self.checks).append(name)
        self.seen_hashes.add(row['sha']); self.saved_corners.append(row['corners'])
        self._event('captured', file=name, split=split)

    def begin(self, use_history=False):
        with self.lock:
            if self.stage == 'need_more':
                raise ValueError('当前是参数质量问题，请先查看诊断，不按张数反复追加照片')
            need(not self.busy and self.stage in ('prepare', 'fit', 'check', 'need_more'), 'capture_not_available')
            if self.stage == 'prepare' and use_history:
                for seed in self.history['intrinsics_seeds']:
                    if seed['camera_id'] != self.manifest['camera_id'] or seed['image_size'] != self.manifest['image_size']:
                        continue
                    raw = (self.history_root/seed['file']).read_bytes()
                    need(hashlib.sha256(raw).hexdigest() == seed['file_sha256'], 'historical_seed_changed')
                    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                    points, reason = inspect_board(image)
                    if points is None or not novel_board(points, self.saved_corners) or len(self.fit) >= 6:
                        continue
                    name = f'images/history-{len(self.fit):03}.jpg'; (self.root/name).write_bytes(raw)
                    self.fit.append(name); self.seen_hashes.add(seed['file_sha256']); self.saved_corners.append(points)
                    self._event('historical_seed_imported', file=name, source_sha256=seed['file_sha256'],
                                same_lens_settings_confirmed_by_operator=True)
            self.stage = 'fit' if len(self.fit) < self.goal_fit else 'check'
            self.auto = True; self.error = None; self.rows.clear()
            self.note = '缓慢移动棋盘到不同位置并前后左右倾斜；停稳约一秒自动拍照'
            self._event('capture_started', use_history=bool(use_history))

    def _calculate(self):
        try:
            manifest_path = self.root/('intrinsics-manifest-'+str(len(self.fit))+'-'+fingerprint(self.manifest)[:10]+'.json')
            if manifest_path.exists(): need(read_json(manifest_path) == self.manifest, 'frozen_manifest_differs')
            else: write_json(manifest_path, self.manifest)
            camera = calibrate_intrinsics(self.manifest, self.root)
            with self.lock:
                write_json(self.root/'camera.json', camera); self.camera = camera
                self.stage = 'place_markers'; self.note = '镜头计算通过。固定标记板，把旧棋盘恢复原位置；确认画面中的C2和方向'
                self.error = None; self._event('intrinsics_calculated', check_rms_px=camera['check_rms_px'])
        except Exception as exc:
            with self.lock:
                self.stage = 'need_more'; self.error = str(exc)
                self.note = '现有照片未通过质量检查，已停止采集；先检查具体原因，不自动要求再补三张'
                self._event('intrinsics_rejected', reason=str(exc))
        finally:
            with self.lock: self.busy = False

    def process(self, row):
        with self.lock:
            if self.stream is not None and self.stream != row['stream']:
                self.auto = False; self.error = '相机流已重启；先暂停并核对相机设置，再新建采集会话'
                self.rows.clear()
                if self.relocator: self.relocator.invalidate('camera_stream_changed_recheck_settings')
                return
            if row['frame_id'] <= self.frame_id:
                return
            self.stream, self.frame_id = row['stream'], row['frame_id']
            self.frame = row
            self._track_markers(row)
            points, reason = inspect_board(row['image'])
            overlay = row['image'].copy()
            if self.relocator and self.relocator.pose:
                for label_index, (name, domain) in enumerate(self.relocator.snapshot().get('legacy_domains', {}).items()):
                    outline = np.rint(domain['projected_pixels'][:-1]).astype(np.int32)
                    cv2.polylines(overlay, [outline], True, (0, 180, 220), 1, cv2.LINE_AA)
                    center = tuple(np.rint(domain['projected_pixels'][-1]).astype(int))
                    label = (center[0]+12, center[1]+label_index*15)
                    cv2.line(overlay, center, label, (0, 150, 180), 1, cv2.LINE_AA)
                    cv2.putText(overlay, name+str(domain['height_mm']), label,
                                cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 150, 180), 1, cv2.LINE_AA)
            if points is not None:
                cv2.drawChessboardCorners(overlay, (5, 4), points.reshape(-1, 1, 2), True)
                labels = points[::-1] if self.reverse else points
                for index, name, shift in ((2, 'C1', (-12, -15)), (7, 'C2 ORIGIN', (-85, 35)), (8, 'D2', (15, 2))):
                    p = np.rint(labels[index]).astype(int)
                    target = (max(5, int(p[0]+shift[0])), max(15, int(p[1]+shift[1])))
                    cv2.circle(overlay, tuple(p), 5, (20, 20, 240), 1)
                    cv2.line(overlay, tuple(p), target, (20, 20, 240), 1)
                    cv2.putText(overlay, name, target, cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 240), 1, cv2.LINE_AA)
            self.preview = cv2.imencode('.jpg', overlay)[1].tobytes()
            if not self.auto: return
            if points is None:
                self.rows.clear(); self.note = reason; return
            row = dict(row, corners=points)
            if self.stage in ('register', 'verify'):
                row['markers'] = detect_markers(row['image'], self.camera)
                if len([m for m in row['markers'] if m['id'] in (10, 11, 12, 13)]) < 3:
                    self.rows.clear(); self.note = '请让至少三枚固定标记和旧棋盘同时完整可见'; return
            self.rows.append(row)
            if not stable_boards(self.rows): self.note = '棋盘正在移动，请停稳约一秒'; return
            if self.stage in ('fit', 'check'):
                if not novel_board(points, self.saved_corners):
                    self.note = self._next_hint(); return
                if row['sha'] in self.seen_hashes: return
                self._capture(row, self.stage); self.rows.clear()
                if len(self.fit) >= self.goal_fit: self.stage = 'check'
                self.note = self._next_hint()
                if len(self.fit) >= self.goal_fit and len(self.checks) >= self.goal_check:
                    self.auto = False; self.busy = True; self.stage = 'calculating'
                    self.note = '照片已收齐，正在自动计算镜头参数'
                    threading.Thread(target=self._calculate, daemon=True).start()
            elif self.stage in ('register', 'verify'):
                action = self.stage; self.auto = False
                try:
                    pose, registration, pairs = registration_from_frames(list(self.rows), self.camera, self.station, self.reverse)
                    if action == 'verify':
                        old_h = np.asarray(self.registration['world_to_card_xy'])
                        new_h = np.asarray(registration['world_to_card_xy'])
                        # Compare both coordinate bridges over actual detected
                        # card locations; never silently update the old bridge.
                        xy = np.asarray(pairs['world_xy_mm']+pairs['check_world_xy_mm'])
                        old = (old_h @ np.c_[xy, np.ones(len(xy))].T).T[:, :2]
                        new = (new_h @ np.c_[xy, np.ones(len(xy))].T).T[:, :2]
                        drift = float(np.linalg.norm(old-new, axis=1).max())
                        need(drift <= 1.5, 'moved_camera_check_differs_over_1_5mm')
                        registration.update(previous_registration_sha256=fingerprint(self.registration),
                                            move_check_max_mm=drift)
                    suffix = action+'-'+str(time.time_ns())
                    write_json(self.root/(suffix+'-pose.json'), pose)
                    write_json(self.root/(suffix+'-pairs.json'), pairs)
                    write_json(self.root/(suffix+'-registration.json'), registration)
                    for index, original in enumerate(self.rows):
                        (self.root/(suffix+f'-{index}.jpg')).write_bytes(original['raw'])
                    if action == 'register':
                        write_json(self.root/'world-registration.json', registration); self.registration = registration
                        self.verification = None; self._start_relocator()
                    else:
                        self.verification = registration
                    self.stage = 'registered' if action == 'register' else 'verified'
                    self.note = '登记通过。只移动相机、保持棋盘和标记不动，再点击换机位检查' if action == 'register' else '换机位坐标检查通过；三类旧映射保留。本页不执行机械动作'
                    self.error = None; self._event(action+'_checked', check_max_mm=registration['check_max_mm'])
                except Exception as exc:
                    self.error = str(exc); self.note = '本次检查未通过，原登记保留；核对棋盘原点、方向和清晰度'
                    self.stage = 'place_markers' if action == 'register' else 'registered'
                    self._event(action+'_rejected', reason=str(exc))

    def _next_hint(self):
        if len(self.fit) >= self.goal_fit:
            return '进入保留检查照片：再换三个不同位置/倾角，停稳自动拍；不计入拟合'
        if self.saved_corners:
            centers = np.array([np.asarray(p).mean(0) for p in self.saved_corners])
            if np.ptp(centers[:, 0]) < 128: return '请把棋盘移到画面左侧或右侧，再停稳'
            if np.ptp(centers[:, 1]) < 96: return '请把棋盘移到画面上方或下方，再停稳'
        return '请改变棋盘倾角或远近；不要只在同一位置平放重复拍摄'

    def action(self, name, data):
        if name == 'begin': self.begin(data.get('same_lens_confirmed') is True); return
        with self.lock:
            need(not self.busy, '计算进行中，请稍等')
            if name == 'pause': self.auto = False; self.rows.clear(); self.note = '已暂停，已保存照片保留'
            elif name == 'calculate':
                need(self.camera is None and len(self.fit) >= 10 and len(self.checks) >= 3,
                     '现有照片尚不足，或已完成镜头计算')
                self.auto = False; self.busy = True; self.stage = 'calculating'
                self.note = '正在使用全部已有照片重新计算，不采集新照片'
                threading.Thread(target=self._calculate, daemon=True).start()
            elif name == 'reverse': self.reverse = not self.reverse; self.rows.clear()
            elif name in ('register', 'verify'):
                need(self.camera is not None, '请先完成镜头采图与计算')
                need(data.get('board_scale_confirmed') is True and data.get('old_card_restored') is True,
                     '请先确认打印尺寸和旧棋盘原位置及朝向')
                need(name != 'verify' or self.registration is not None, '请先完成坐标登记')
                self.stage = name; self.auto = True; self.rows.clear(); self.error = None
                self.note = '保持旧棋盘与标记板同时可见，停稳后自动保存三帧'
            else: raise ValueError('unknown_action')

    def run(self):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while not self.closed.is_set():
            try:
                with opener.open(self.camera_url, timeout=1.5) as response:
                    raw = response.read(8500001); age = float(response.headers['X-Frame-Age-Ms'])
                    sid = response.headers['X-Stream-Id']; fid = int(response.headers['X-Frame-Id'])
                need(len(raw) <= 8500000 and sid and 0 <= age <= 1500, 'camera_frame_stale')
                image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                need(image is not None and list(image.shape[:2][::-1]) == [640, 480], 'camera_frame_size_changed')
                with self.lock:
                    if self.error and self.error.startswith('相机画面暂不可用：'): self.error = None
                self.process(dict(raw=raw, sha=hashlib.sha256(raw).hexdigest(), image=image,
                    frame_id=fid, stream=sid, age=age))
            except Exception as exc:
                with self.lock:
                    self.error = '相机画面暂不可用：'+str(exc); self.rows.clear(); self.frame = None
                    if self.relocator: self.relocator.invalidate('camera_frame_unavailable')
            self.closed.wait(.35)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--history', type=Path, required=True); p.add_argument('--station', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--camera-url', default='http://127.0.0.1:8780/frame.jpg')
    p.add_argument('--port', type=int, default=8782)
    p.add_argument('--resume', action='store_true', help='Restore a paused session without discarding or overwriting photographs')
    args = p.parse_args()
    need(1024 <= args.port <= 65535, 'invalid_local_port')
    session = GuidedSession(args.output, args.history, read_json(args.station), args.camera_url, resume=args.resume)
    token = secrets.token_urlsafe(24)
    page = (Path(__file__).parent/'monitor_assets/station_guided.html').read_text(encoding='utf-8').replace('__TOKEN__', token)
    origins = {'http://127.0.0.1:'+str(args.port), 'http://localhost:'+str(args.port)}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def reply(self, status, kind, body):
            self.send_response(status); self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body))); self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff'); self.end_headers()
            try: self.wfile.write(body)
            except (ConnectionError, TimeoutError): pass
        def allowed(self):
            return 'http://'+self.headers.get('Host', '') in origins
        def do_GET(self):
            if not self.allowed(): self.send_error(403); return
            if self.path == '/': self.reply(200, 'text/html; charset=utf-8', page.encode())
            elif self.path == '/state': self.reply(200, 'application/json', json.dumps(session.state(), ensure_ascii=False).encode())
            elif self.path == '/localization':
                self.reply(200, 'application/json', json.dumps(session.localization(), ensure_ascii=False).encode())
            elif self.path.startswith('/preview.jpg'):
                with session.lock: raw = session.preview
                if raw is None: self.send_error(503); return
                self.reply(200, 'image/jpeg', raw)
            elif self.path in ('/fixed-markers.svg', '/chessboard.svg'):
                name = 'fixed_markers_a4.svg' if self.path == '/fixed-markers.svg' else 'intrinsics_chessboard_a4.svg'
                self.reply(200, 'image/svg+xml', (Path(__file__).parents[1]/'data/station_calibration'/name).read_bytes())
            else: self.send_error(404)
        def do_POST(self):
            if (not self.allowed() or self.headers.get('Origin') not in origins or
                    self.headers.get('X-Calibration-Token') != token): self.send_error(403); return
            try:
                size = int(self.headers.get('Content-Length', '0')); need(0 < size <= 4096, 'invalid_body')
                data = json.loads(self.rfile.read(size)); session.action(self.path.removeprefix('/'), data)
                self.reply(200, 'application/json', json.dumps(session.state(), ensure_ascii=False).encode())
            except (ValueError, KeyError) as exc:
                self.reply(400, 'application/json', json.dumps(dict(error=str(exc)), ensure_ascii=False).encode())
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    threading.Thread(target=session.run, daemon=True).start()
    print('Guided camera-only calibration ready: http://127.0.0.1:'+str(args.port), flush=True)
    try: server.serve_forever()
    finally: session.closed.set(); server.server_close()


if __name__ == '__main__': main()
