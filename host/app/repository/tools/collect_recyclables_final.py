"""Guided camera-only collection using an existing loopback camera monitor.

Raw candidates remain unannotated. A capture is an operator declaration of the
shown arrangement, never an identity/hand detector or an acceptance result.
"""
import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import cv2
import numpy as np

NAMES = {'can': '金属', 'plastic': '塑料', 'cardboard': '纸类'}
BINS = {'can': '右上金属箱', 'plastic': '左侧塑料箱', 'cardboard': '右下纸类箱'}
# Diagnostic image regions for this fixed workcell. They do not define reachability.
REGIONS = {'source': (175, 95, 420, 225), 'can': (405, 85, 640, 265),
           'plastic': (10, 265, 242, 425), 'cardboard': (395, 258, 628, 425)}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def make_registry(source, old_manifest):
    """Use actual latest dataset ownership, not the superseded initial registry."""
    owners = {}
    for row in old_manifest['images']:
        for oid in row['object_ids']:
            if oid in owners and owners[oid] != row['split']:
                raise ValueError('Historical object crosses splits: ' + oid)
            owners[oid] = row['split']
    result = []
    prepared = source.get('schema_version') == 'recyclable-prepared-registry-v1'
    seen = set()
    for item in source['objects']:
        oid = item['object_id']
        if oid in seen:
            raise ValueError('Duplicate object ID: ' + oid)
        seen.add(oid)
        if prepared:
            if item['material'] not in NAMES or item['reserved_split'] not in ('train', 'val', 'test'):
                raise ValueError('Invalid material or split')
            if oid in owners and owners[oid] != item['reserved_split']:
                raise ValueError('Historical ownership cannot change: ' + oid)
            if item['reserved_split'] == 'test' and (oid in owners or item.get('previously_used_for_development') is not False):
                raise ValueError('Held-out test requires a previously unused physical object')
            if any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in oid):
                raise ValueError('Object IDs must use safe ASCII names')
            result.append(dict(item))
            continue
        if oid not in owners:
            continue
        result.append(dict(item, reserved_split=owners[oid],
                           material=item['legacy_model_class'], previously_used_for_development=True))
    return result


def make_plan(objects, split, register_new=False):
    active = [o for o in objects if o['reserved_split'] == split]
    rows = []
    def add(title, ids, scene, pose=None, material=None):
        row = dict(id=f'{len(rows)+1:04}', title=title, object_ids=ids,
                   scene_type=scene, pose_id=pose, region='source')
        if material:
            row['bin_material'] = material
            row['region'] = material
        rows.append(row)
    if register_new and split == 'train':
        for obj in objects:
            if obj.get('requires_reference'):
                use = {'train':'训练', 'val':'验证', 'test':'封存测试'}[obj['reserved_split']]
                add(f'登记新增实物【{obj["label"]}】（{use}）：源区绿色框中只放这一件，手退出后拍参考照。以后此编号始终对应同一件；参考照不计训练张数。',
                    [obj['object_id']], 'reference', 'reference')
    # One representative full-workcell empty frame per actual session is the
    # minimum. Extra genuine lighting conditions are optional, never a quota
    # of artificial distractors or duplicate stationary captures.
    add('空场景：源区和三个箱子全部清空，只保留固定箱体、机械臂和标记。不要另放U盘、碎纸或其他物品；手退出，拍1张全工位空景即可。', [], 'empty')
    for obj in active:
        oid, label, material = obj['object_id'], obj['label'], obj['material']
        for scene in ('source', 'bin'):
            positions = ['中间', '左侧/近侧边缘', '右侧/远侧边缘']
            poses = ['正面朝上、长轴顺放', '正面朝上、转约90度', '背面朝上、横放', '侧面朝上或瓶口/罐底换向']
            for pi, position in enumerate(positions):
                for di, pose in enumerate(poses):
                    location = '源区' if scene == 'source' else BINS[material]
                    add(f'只放【{label}】到{location}{position}：{pose}。其余区域清空，手退出。',
                        [oid], scene, f'{scene}_p{pi*4+di+1:02}', material if scene == 'bin' else None)
    per_scene = 36 if split == 'train' else 12
    grouped = {m: [o for o in active if o['material'] == m] for m in NAMES}
    if all(grouped.values()):
        for m in NAMES:
            wrong = [b for b in NAMES if b != m]
            for i in range(per_scene // 3):
                obj = grouped[m][i % len(grouped[m])]
                dest = wrong[i % 2]
                add(f'错箱：只把【{obj["label"]}】放进{BINS[dest]}，换到第{i//2+1}种可见面/位置；其余清空。仍按真实材质标注。',
                    [obj['object_id']], 'wrong_bin', material=dest)
        for a, b in [('can', 'plastic'), ('can', 'cardboard'), ('plastic', 'cardboard')]:
            for i in range(per_scene // 3):
                oa, ob = grouped[a][i % len(grouped[a])], grouped[b][i % len(grouped[b])]
                add(f'源区同时放【{oa["label"]}】和【{ob["label"]}】，保持间距、无叠放；第{i+1}种不同位置/方向。三个箱子留空。',
                    [oa['object_id'], ob['object_id']], 'multi')
    return rows


class Collection:
    def __init__(self, args):
        self.args = args
        self.lock = threading.RLock()
        self.frames = deque(maxlen=4)
        self.error = '等待实时画面'
        self.note = '准备当前提示，手退出后按空格拍照'
        self.armed = False
        self.armed_at = 0
        self.countdown = args.seconds
        self.auto = False
        self.root = args.output.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        registry_path = self.root / 'object_registry.json'
        objects = make_registry(read_json(args.registry), read_json(args.previous_manifest))
        self.prepared = read_json(args.registry).get('schema_version') == 'recyclable-prepared-registry-v1'
        if registry_path.exists():
            registered = read_json(registry_path)['objects']
            strip_refs = lambda rows: [{k:v for k,v in o.items() if k not in ('reference_image','reference_sha256')} for o in rows]
            if strip_refs(registered) != strip_refs(objects):
                raise ValueError('Existing object registry differs; use its original sources')
            objects = registered
        self.objects = objects
        if self.prepared and args.split != 'train' and any(o.get('requires_reference') and not o.get('reference_image') for o in objects):
            raise ValueError('Complete new-object reference registration in the training setup session first')
        save_json(registry_path, dict(objects=objects, physical_motion=False,
                  note='Frozen physical object ownership; reference photographs are not training or evaluation images.'))
        self.session_dir = self.root / 'sessions' / args.session
        self.session_dir.mkdir(parents=True, exist_ok=True)
        path = self.session_dir / 'session.json'
        frozen = dict(session_id=args.session, split=args.split, workcell_id=args.workcell,
                      objects=[o['object_id'] for o in objects if o['reserved_split'] == args.split],
                      plan=make_plan(objects, args.split, self.prepared), physical_motion=False,
                      camera_url=args.camera_url, created_at=datetime.now(timezone.utc).isoformat())
        if path.exists():
            old = read_json(path)
            if any(old[k] != frozen[k] for k in ('split', 'workcell_id', 'objects', 'camera_url', 'plan')):
                raise ValueError('Session metadata changed; cannot silently change session/split')
            frozen = old
        else:
            save_json(path, frozen)
        self.plan = frozen['plan']
        progress = self.session_dir / 'progress.json'
        self.events = read_json(progress)['events'] if progress.exists() else []
        self.rows = read_json(self.root / 'split_manifest.json')['images'] if (self.root / 'split_manifest.json').exists() else []
        # One shared monitor must not be simultaneously used for another split.
        self.persist()

    def index(self):
        completed = {e['step_id'] for e in self.events if e['status'] in ('captured', 'registered', 'skipped')}
        return next((i for i, step in enumerate(self.plan) if step['id'] not in completed), len(self.plan))

    def persist(self):
        save_json(self.root / 'split_manifest.json', dict(schema_version='recyclable-final-dataset-v1',
                  class_names=list(NAMES), object_classes={o['object_id']: o['material'] for o in self.objects},
                  images=self.rows, independent_test_claimed=False,
                  collection_note='Collection in progress. Pending contour review and full coverage; reference images do not enter training/evaluation.'))
        save_json(self.root / 'object_registry.json', dict(objects=self.objects, physical_motion=False,
                  note='Frozen physical object ownership; reference photographs are not training or evaluation images.'))
        save_json(self.session_dir / 'progress.json', dict(events=self.events, physical_motion=False))

    def state(self):
        with self.lock:
            index = self.index()
            age = time.monotonic() - self.frames[-1]['received'] if self.frames else None
            current = self.plan[index] if index < len(self.plan) else None
            completed = {e['step_id'] for e in self.events if e['status'] in ('captured','registered','skipped')}
            active_completed = sum(step['id'] in completed for step in self.plan)
            next_pending = next((step for step in self.plan[index+1:] if step['id'] not in completed), None)
            return dict(index=index, total=len(self.plan), current=current,
                        completed_steps=active_completed,
                        next=next_pending['title'] if next_pending else '本场次结束，其余已完成步骤无需重拍',
                        split=self.args.split, session=self.args.session,
                        saved=sum(e['status']=='captured' for e in self.events),
                        registered=sum(bool(o.get('requires_reference') and o.get('reference_image')) for o in self.objects),
                        registration_total=sum(bool(o.get('requires_reference')) for o in self.objects),
                        object_count=len(self.objects), regions={k:list(v) for k,v in REGIONS.items()},
                        skipped=sum(e['status']=='skipped' for e in self.events),
                        scene_counts=dict(Counter(r['scene_type'] for r in self.rows)),
                        age=round(age, 1) if age is not None else None, error=self.error,
                        note=self.note, armed=self.armed, auto=self.auto,
                        countdown=round(self.countdown, 1), output=str(self.root),
                        gallery=([e['image'] for e in self.events if e['status']=='registered']+
                                 [r['image'] for r in self.rows if r['session_id']==self.args.session])[-8:],
                        objects=[dict(id=o['object_id'], label=o['label'], split=o['reserved_split'],
                                      has_reference=bool(o.get('reference_image'))) for o in self.objects])

    def command(self, action):
        with self.lock:
            if action in ('pause', 'end_session'):
                self.armed = self.auto = False
                self.note = '已暂停；按空格仅拍当前一项'
            elif action in ('capture', 'auto'):
                if self.index() == len(self.plan):
                    raise ValueError('场次已完成')
                if action == 'auto' and self.plan[self.index()]['scene_type'] == 'reference':
                    raise ValueError('登记阶段请逐件按空格拍照，完成登记后再连续采集')
                if not self.frames or time.monotonic()-self.frames[-1]['received'] > 2 or self.error:
                    raise ValueError('实时相机未就绪')
                if self.armed:
                    raise ValueError('正在等候停稳，请勿重复按键')
                self.auto = action == 'auto'
                self.armed = True
                self.armed_at = time.monotonic()
                self.countdown = self.args.seconds if self.auto else 1.5
                self.note = '请按当前提示摆放；手退出，等待画面停稳'
            elif action == 'skip':
                if self.armed:
                    raise ValueError('先暂停再跳过')
                if self.index() == len(self.plan):
                    raise ValueError('场次已完成')
                if self.plan[self.index()]['scene_type'] == 'reference':
                    raise ValueError('新增实物必须逐件登记，不能跳过')
                self.events.append(dict(step_id=self.plan[self.index()]['id'], status='skipped'))
                self.persist()
            elif action == 'undo':
                if self.armed:
                    raise ValueError('先暂停再撤销')
                if not self.events:
                    raise ValueError('没有可撤销项目')
                event = self.events[-1]
                if event['status'] in ('captured', 'registered'):
                    # Preserve raw bytes and audit trail; remove candidate from active manifest.
                    src = self.root / event['image']
                    dest = self.session_dir / 'rejected' / (str(time.time_ns())+'-'+src.name)
                    dest.parent.mkdir(exist_ok=True)
                    src.replace(dest)
                    self.rows = [r for r in self.rows if r['image'] != event['image']]
                    if event['status'] == 'registered':
                        obj = next(o for o in self.objects if o['object_id']==event['object_id'])
                        obj['reference_image'] = obj['reference_sha256'] = None
                    with (self.session_dir / 'rejected.jsonl').open('a', encoding='utf-8') as f:
                        f.write(json.dumps(dict(event, quarantined=str(dest)), ensure_ascii=False)+'\n')
                self.events.pop()
                self.persist()
                self.note = '上一项已撤销，可重新摆放拍照'
            else:
                raise ValueError('未知操作')

    def capture(self, step):
        if len(self.frames) < 3:
            return '等待三张新帧'
        frames = list(self.frames)[-3:]
        if any(f['received'] < self.armed_at for f in frames) or frames[-1]['received']-frames[0]['received'] < .45:
            return '等待拍照请求之后的新帧'
        if frames[-1]['received_sequence_ns'] <= frames[0]['received_sequence_ns']:
            return '相机时间未前进'
        x1, y1, x2, y2 = (0, 0, 640, 480) if step['scene_type']=='empty' else REGIONS[step['region']]
        crops = [f['gray'][y1:y2, x1:x2] for f in frames]
        # Whole frame catches hands entering/exiting outside the target ROI too.
        motion = [float(np.mean(cv2.absdiff(a['gray'], b['gray']) > 18)) for a,b in zip(frames,frames[1:])]
        roi_motion = [float(np.mean(cv2.absdiff(a,b) > 18)) for a,b in zip(crops,crops[1:])]
        if max(motion) > .012 or max(roi_motion) > .018:
            return '画面还在动，等手退出并停稳'
        scores = [float(cv2.Laplacian(c, cv2.CV_64F).var()) for c in crops]
        best = frames[int(np.argmax(scores))]
        if max(scores) < 12:
            return '当前区域清晰度不足，请检查遮挡/对焦'
        digest = hashlib.sha256(best['jpeg']).hexdigest()
        if step['scene_type'] == 'reference':
            oid = step['object_ids'][0]
            obj = next(o for o in self.objects if o['object_id']==oid)
            if obj.get('reference_image'):
                raise ValueError('This reference is already registered; resume its original progress')
            for other in self.objects:
                if other.get('requires_reference') and other.get('reference_image'):
                    old = cv2.imread(other['reference_image'], cv2.IMREAD_GRAYSCALE)
                    delta = cv2.absdiff(best['gray'][y1:y2,x1:x2], old[y1:y2,x1:x2])
                    if float(np.mean(delta > 18)) < .012 and float(delta.mean()) < 3:
                        return '与已登记实物参考照近似重复，请换下一件并展示不同包装正面'
            relative = 'references/'+oid+'.jpg'
            path = self.root / relative
            path.parent.mkdir(exist_ok=True)
            with path.open('xb') as f:
                f.write(best['jpeg'])
            obj.update(reference_image=str(path), reference_sha256=digest)
            self.events.append(dict(step_id=step['id'], status='registered', image=relative,
                                    object_id=oid, quality_laplacian_variance=max(scores)))
            self.persist()
            return None
        if any(r['image_sha256']==digest for r in self.rows):
            return '与已有照片完全重复，请改变姿态'
        # Compare to ALL accepted candidates in this scene/physical-object group,
        # including resumed sessions, so a stationary series cannot inflate coverage.
        group = [r for r in self.rows if r['object_ids']==step['object_ids'] and
                 r['scene_type']==step['scene_type'] and r.get('bin_material')==step.get('bin_material') and
                 (step['scene_type']!='empty' or r['capture_session_id']==self.args.session)]
        crop = best['gray'][y1:y2, x1:x2]
        for row in group:
            old = cv2.imread(str(self.root / row['image']), cv2.IMREAD_GRAYSCALE)
            delta = cv2.absdiff(crop, old[y1:y2, x1:x2])
            if float(np.mean(delta > 18)) < .012 and float(delta.mean()) < 3:
                return '与同组已有姿态近似重复，请换位置/可见面；空景无变化可暂停或跳过'
        filename = f'{self.args.session}-{step["id"]}-{time.time_ns()}.jpg'
        relative = f'images/{self.args.split}/{filename}'
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as f:
            f.write(best['jpeg'])
        row = dict(image=relative, split=self.args.split, object_ids=step['object_ids'],
                   session_id=self.args.session, capture_session_id=self.args.session,
                   workcell_id=self.args.workcell, scene_type=step['scene_type'],
                   annotation_status='pending', pose_id=step['pose_id'],
                   image_sha256=digest, received_sequence_ns=best['received_sequence_ns'],
                   source_board_timestamp_available=False,
                   received_at=best['received_at'], source_monitor=self.args.camera_url,
                   source_monitor_frame_age_s=best['source_age'],
                   operator_arrangement_confirmed=True, physical_motion=False,
                   quality=dict(laplacian_variance=max(scores), whole_frame_motion=motion,
                                checked_region=[x1,y1,x2,y2],
                                region_motion=roi_motion, three_frame_hashes=[hashlib.sha256(f['jpeg']).hexdigest() for f in frames]),
                   quality_limit='Stability is not proof of no hand, correct identity, pose or segmentation.')
        if 'bin_material' in step:
            row['bin_material'] = step['bin_material']
        self.rows.append(row)
        self.events.append(dict(step_id=step['id'], status='captured', image=relative))
        self.persist()
        return None

    def worker(self):
        reconnect_at = 0
        while True:
            try:
                with urlopen(self.args.camera_url+'/state', timeout=4) as response:
                    state = json.load(response)
                if state.get('mode') != 'waiting':
                    if state.get('mode') == 'error' and time.monotonic()-reconnect_at > 15:
                        reconnect_at = time.monotonic()
                        request = Request(self.args.camera_url+'/action', data=b'{"command":"reconnect"}',
                                          headers={'Content-Type':'application/json'}, method='POST')
                        with urlopen(request, timeout=4) as response:
                            response.read()
                    raise ValueError('相机源未就绪：'+str(state.get('mode')))
                if state.get('camera_age_seconds') is None or state['camera_age_seconds'] > 1.5:
                    raise ValueError('相机画面已过期')
                with urlopen(self.args.camera_url+'/frame?t='+str(time.time_ns()), timeout=4) as response:
                    jpeg = response.read(2_000_000)
                gray = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
                if gray is None or gray.shape != (480,640):
                    raise ValueError('相机应为640×480')
                # The monitor only exposes a JPEG, not the board timestamp. Require
                # distinct source frames; received_sequence_ns is a local receipt sequence,
                # and metadata explicitly records that limitation.
                now = time.monotonic()
                with self.lock:
                    self.error = None
                    if not self.frames or self.frames[-1]['jpeg'] != jpeg:
                        self.frames.append(dict(jpeg=jpeg, gray=gray, received=now,
                             received_at=datetime.now(timezone.utc).isoformat(),
                             source_age=state['camera_age_seconds'], received_sequence_ns=time.monotonic_ns()))
                    if self.frames and now-self.frames[-1]['received'] > 2:
                        raise ValueError('画面内容未更新，请恢复相机')
                    if self.armed and self.index() < len(self.plan):
                        self.countdown = max(0, (self.args.seconds if self.auto else 1.5)-(now-self.armed_at))
                        if self.countdown == 0:
                            reason = self.capture(self.plan[self.index()])
                            if reason:
                                self.note = reason
                            else:
                                self.note = '已保存；请按下一项摆放，手退出后按空格。参考照只作编号对照，不计训练张数。'
                                self.armed = self.auto and self.index() < len(self.plan)
                                self.armed_at = now
            except Exception as exc:
                with self.lock:
                    self.error = str(exc)
                    self.armed = self.auto = False
                    self.frames.clear()
                time.sleep(.7)
            time.sleep(.22)


HTML = r'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>三类材料 · 样本采集台</title><style>
*{box-sizing:border-box}body{margin:0;background:#101722;color:#e7eef7;font:17px system-ui}header{padding:18px 28px;border-bottom:1px solid #304055}h1{font-size:24px;margin:0 0 6px}small{color:#aebed0}main{display:grid;grid-template-columns:minmax(500px,680px) minmax(310px,1fr);gap:20px;padding:22px;max-width:1400px;margin:auto}.panel{background:#1c2838;border:1px solid #32435b;border-radius:14px;padding:20px;margin-bottom:16px}img{max-width:100%;border-radius:8px}#live{width:640px;aspect-ratio:4/3}h2{font-size:25px;line-height:1.6;margin:12px 0;color:#fff}button{border:0;border-radius:8px;padding:13px 18px;margin:5px 4px 5px 0;font:inherit;cursor:pointer;background:#344a65;color:white}button.primary{background:#63e0b6;color:#102820;font-weight:700}button:disabled{opacity:.4}#status{color:#63e0b6}#notice{color:#ffd28a;min-height:52px}#count{font-size:30px;font-weight:bold}progress{width:100%;height:16px;accent-color:#63e0b6}#gallery{display:flex;gap:8px;flex-wrap:wrap}#gallery img{width:145px}.pill{font-size:14px;background:#35465b;border-radius:6px;padding:4px 8px}.muted{color:#acbdd0;font-size:15px}#reference{width:160px}label{display:block;padding:10px 0}a{color:#74e4bd}@media(max-width:900px){main{display:block}} .warn{color:#ffd28a}
</style><header><h1>三类材料 · 样本采集台 <span class=pill id=split></span></h1><small>固定相机 · 16件实物（7金属 / 5塑料 / 4纸类） · 按姿态选1张 · 全程只拍照　｜　手退出后按空格，高效连续采集</small></header>
<main><section><div class=panel><b id=status>连接实时相机…</b><p style="position:relative"><img id=live alt="实时工位画面"><svg viewBox="0 0 640 480" style="position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none"><rect id=roi fill="none" stroke="#72ffd1" stroke-width="2" stroke-dasharray="7 5"/><text id=roilabel fill="#72ffd1" font-size="15"></text></svg></p><small>绿色框仅提示当前拍摄区域，不代表机械可达范围。源区＝机械臂上方空白工作区；左箱塑料 / 右上金属 / 右下纸类。</small></div><div class=panel><b>最近保存的原图（待轮廓标注）</b><div id=gallery></div></div></section>
<aside><div class=panel><div id=count>准备中</div><progress id=progress value=0 max=1></progress><h2 id=instruction></h2><p id=notice></p><label><input id=confirm type=checkbox> 我会按提示摆放，拍摄时让手完全退出；画面没有其他待分拣物</label><button class=primary id=capture onclick="act('capture')">拍这一张（空格）</button><button id=auto onclick="autoStart()">连续采集（每项8秒起）</button><button onclick="act('pause')">暂停（Esc）</button><button onclick="act('skip')">跳过当前项</button><button onclick="act('undo')">撤销上一项</button><button onclick="act('end_session')">结束本场次</button><p class=muted id=next></p><small>单拍等待1.5秒＋停稳；连续模式按每项8秒起，重复/移动会自动等待。姿态拍完自动进入下一项，可随时暂停。每件源区12张＋箱内12张。</small></div>
<div class=panel><b>当前实物参考</b><p id=obj></p><img id=reference><p class=muted>图中实物身份和摆放由你核对。参考图只帮助认样，不作新样本。旧实物沿用历史归属；新增实物先登记编号。测试件单独保留，参考照不参加训练。</p></div><div class=panel><b>已归档场景</b><p id=counts></p><p class=warn>照片先进入待标注区。16件固定方案：训练8件、验证4件、测试4件，最低507张有效图（空景每场次先拍1张）。照片全部待标注；不启动训练或替换模型。</p><small id=path></small></div></aside></main>
<style>@media(min-width:561px) and (max-width:900px){body{font-size:14px}header{padding:12px}header h1{font-size:21px}main{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);padding:10px;gap:10px}.panel{padding:12px}h2{font-size:19px;line-height:1.5}#count{font-size:22px}button{font-size:14px;padding:10px}#reference{width:120px}}</style>
<script>
let state=null, busy=false, oldStep='', oldGallery='';
async function act(command){if(busy)return;if((command==='capture'||command==='auto')&&!document.querySelector('#confirm').checked){document.querySelector('#notice').textContent='先勾选摆放与手退出确认，再拍照';return}busy=true;try{let r=await fetch('/action',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({command})});if(!r.ok)document.querySelector('#notice').textContent=await r.text()}finally{busy=false}}
function autoStart(){act('auto')}
document.addEventListener('keydown',e=>{if(e.code==='Space'&&!['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName)){e.preventDefault();act('capture')}if(e.code==='Escape')act('pause')});
function beep(){try{let a=new AudioContext(),o=a.createOscillator(),g=a.createGain();g.gain.value=.08;o.connect(g);g.connect(a.destination);o.start();o.stop(a.currentTime+.14);o.onended=()=>a.close()}catch{}}
async function tick(){try{let s=await(await fetch('/state')).json();state=s;let fresh=s.age!==null&&s.age<2&&!s.error;document.querySelector('#status').textContent=fresh?'● 实时画面 · '+s.age+'秒前':'● 相机暂停/过期：'+(s.error||'无新帧');document.querySelector('#status').style.color=fresh?'#63e0b6':'#ff9797';document.querySelector('#split').textContent=({train:'训练场次',val:'验证场次',test:'封存测试采集'})[s.split];document.querySelector('#count').textContent=`已拍 ${s.saved} 张 · 已登记 ${s.registered||0}/10件 · ${Math.min((s.completed_steps??s.index)+1,s.total)}/${s.total}`;document.querySelector('#progress').max=s.total;document.querySelector('#progress').value=s.completed_steps??s.index;document.querySelector('#instruction').textContent=s.current?s.current.title:'本场次采集完成。照片待标注复核；请结束本场次，再安排验证采集。';document.querySelector('#notice').textContent=(s.armed?(s.countdown>0?s.countdown+'秒后检查停稳 · ':'正在检查 · '):'')+s.note;document.querySelector('#next').textContent='下一项：'+s.next;document.querySelector('#capture').disabled=!fresh||s.armed||!s.current;document.querySelector('#auto').disabled=!fresh||s.armed||!s.current||s.current.scene_type==='reference';document.querySelector('#counts').textContent=Object.entries(s.scene_counts).map(([k,v])=>({source:'源区',bin:'箱内',wrong_bin:'错箱',multi:'多物体',empty:'空场景'}[k]+': '+v)).join('　')||'尚未拍照';document.querySelector('#path').textContent=s.output;let step=s.current?.id||'done';if(oldStep&&oldStep!==step)beep();oldStep=step;let ids=s.current?.object_ids||[];document.querySelector('#obj').textContent=ids.map(id=>s.objects.find(o=>o.id===id)?.label||id).join(' + ')||'空场景：无待分拣物';let ref=document.querySelector('#reference');let hasRef=ids.length&&s.objects.find(o=>o.id===ids[0])?.has_reference;ref.hidden=!hasRef;if(hasRef&&ref.dataset.id!==ids[0]){ref.src='/reference/'+encodeURIComponent(ids[0]);ref.dataset.id=ids[0]}let raw=s.current?.scene_type==='empty'?[0,0,640,480]:(s.regions?.[s.current?.region||'source']||[175,95,420,225]);let rect=[raw[0],raw[1],raw[2]-raw[0],raw[3]-raw[1]];let roi=document.querySelector('#roi');['x','y','width','height'].forEach((k,i)=>roi.setAttribute(k,rect[i]));let rt=document.querySelector('#roilabel');rt.setAttribute('x',rect[0]+4);rt.setAttribute('y',rect[1]-5);rt.textContent='当前拍摄区域';let gal=JSON.stringify(s.gallery);if(gal!==oldGallery){document.querySelector('#gallery').replaceChildren(...s.gallery.map(p=>{let a=document.createElement('a');a.href='/image/'+p;a.target='_blank';let i=document.createElement('img');i.src=a.href;a.append(i);return a}));oldGallery=gal}}catch{document.querySelector('#status').textContent='服务连接中断，请暂停摆放'}setTimeout(tick,500)}tick();async function frame(){try{let r=await fetch('/frame?t='+Date.now());if(r.ok){let b=await r.blob(),i=document.querySelector('#live'),old=i.src;i.src=URL.createObjectURL(b);if(old.startsWith('blob:'))URL.revokeObjectURL(old)}}catch{}setTimeout(frame,500)}frame();
</script></html>'''


def serve_frontend(upstream, port):
    """Refresh the view without interrupting a running capture writer/camera."""
    origin = urlparse(upstream)
    if origin.scheme != 'http' or origin.hostname != '127.0.0.1' or not origin.port or origin.path:
        raise ValueError('frontend-for must be a loopback HTTP origin')
    class View(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass
        def reply(self, data, kind, status=200):
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)
        def do_GET(self):
            try:
                if self.path == '/':
                    self.reply(HTML.encode(), 'text/html; charset=utf-8')
                elif self.path == '/state' or self.path.startswith(('/frame?', '/image/', '/reference/')):
                    with urlopen(upstream+self.path, timeout=5) as response:
                        self.reply(response.read(2_000_000), response.headers['Content-Type'])
                else:
                    self.reply(b'Not found', 'text/plain', 404)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                self.reply(str(exc).encode(), 'text/plain; charset=utf-8', 503)
        def do_POST(self):
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if (self.path != '/action' or not 0 < size <= 1024 or
                    self.headers.get('Origin') not in (None, f'http://127.0.0.1:{port}')):
                    raise ValueError('Invalid request')
                request = Request(upstream+'/action', data=self.rfile.read(size),
                                  headers={'Content-Type':'application/json'}, method='POST')
                with urlopen(request, timeout=5) as response:
                    self.reply(response.read(1024), 'application/json')
            except Exception as exc:
                self.reply(str(exc).encode(), 'text/plain; charset=utf-8', 400)
    server = ThreadingHTTPServer(('127.0.0.1', port), View)
    print(f'http://127.0.0.1:{port}', flush=True)
    server.serve_forever()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path)
    p.add_argument('--registry', type=Path)
    p.add_argument('--previous-manifest', type=Path)
    p.add_argument('--session')
    p.add_argument('--frontend-for', help='Serve an updated view for an already running collection writer')
    p.add_argument('--split', choices=['train','val','test'], default='train')
    p.add_argument('--workcell', default='fixed-workcell-20261028')
    p.add_argument('--camera-url', default='http://127.0.0.1:8772')
    p.add_argument('--port', type=int, default=8773)
    p.add_argument('--seconds', type=float, default=8)
    args = p.parse_args()
    if args.frontend_for:
        serve_frontend(args.frontend_for, args.port)
        return
    if any(getattr(args, name) is None for name in ('output', 'registry', 'previous_manifest', 'session')):
        p.error('output, registry, previous-manifest and session are required for collection')
    url = urlparse(args.camera_url)
    if url.scheme != 'http' or url.hostname != '127.0.0.1' or url.path or not url.port:
        p.error('camera-url must be a loopback HTTP origin')
    if not args.session or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.session):
        p.error('session must use ASCII letters, digits, hyphens or underscores')
    if not np.isfinite(args.seconds) or args.seconds < 8:
        p.error('seconds must be finite and at least 8')
    args.output.mkdir(parents=True, exist_ok=True)
    # Exclusive root lock prevents two live collection sessions/splits mixing.
    lock_path = args.output / 'collection.lock'
    lock_file = lock_path.open('a+b')
    lock_file.seek(0)
    if lock_file.read(1) == b'':
        lock_file.write(b'0'); lock_file.flush()
    lock_file.seek(0)
    import msvcrt
    try:
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        p.error('Another collection session is already using this dataset')
    c = Collection(args)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass
        def send(self, data, kind='application/json', status=200):
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(data)
        def do_GET(self):
            try:
                if self.path == '/':
                    self.send(HTML.encode(), 'text/html; charset=utf-8')
                elif self.path == '/state':
                    self.send(json.dumps(c.state(), ensure_ascii=False).encode())
                elif self.path.startswith('/frame?'):
                    with c.lock:
                        frame = c.frames[-1]['jpeg'] if c.frames and time.monotonic()-c.frames[-1]['received'] < 2 else b''
                    self.send(frame, 'image/jpeg', 200 if frame else 503)
                elif self.path.startswith('/reference/'):
                    oid = self.path.removeprefix('/reference/')
                    item = next(o for o in c.objects if o['object_id']==oid)
                    path = Path(item['reference_image'])
                    data = path.read_bytes()
                    if hashlib.sha256(data).hexdigest() != item['reference_sha256']:
                        raise ValueError('Reference hash mismatch')
                    self.send(data, 'image/jpeg')
                elif self.path.startswith('/image/'):
                    relative = self.path.removeprefix('/image/')
                    with c.lock:
                        if not (any(r['image']==relative for r in c.rows) or
                                any(e.get('image')==relative and e['status']=='registered' for e in c.events)):
                            raise ValueError('Unknown image')
                    self.send((c.root / relative).read_bytes(), 'image/jpeg')
                else:
                    self.send(b'Not found', 'text/plain', 404)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                self.send(str(exc).encode(), 'text/plain; charset=utf-8', 400)
        def do_POST(self):
            try:
                if self.headers.get('Origin') not in (None, f'http://127.0.0.1:{args.port}'):
                    raise ValueError('Origin rejected')
                size = int(self.headers.get('Content-Length', '0'))
                if self.path != '/action' or not 0 < size <= 1024:
                    raise ValueError('Invalid request')
                command = json.loads(self.rfile.read(size))['command']
                c.command(command)
                self.send(b'{}')
                if command == 'end_session':
                    threading.Thread(target=server.shutdown, daemon=True).start()
            except (ValueError, KeyError, TypeError) as exc:
                self.send(str(exc).encode(), 'text/plain; charset=utf-8', 400)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    threading.Thread(target=c.worker, daemon=True).start()
    print(f'http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        lock_file.close()


if __name__ == '__main__':
    main()
