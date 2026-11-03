"""Immutable operator-declared empty-scene photographs; no training or PL writes."""
import hashlib
import json
from pathlib import Path
import re
import time
import uuid


class EmptyBackgroundRecords:
    def __init__(self, root):
        self.root = Path(root)

    def save(self, packet, *, confirmed_empty, settings_sha256):
        if confirmed_empty is not True:
            raise ValueError('请先清空源区与三个箱子，手退出画面，再勾选空场确认')
        if packet is None:
            raise ValueError('没有新鲜相机画面，无法记录空背景')
        age = (time.monotonic()-packet['received'])*1000+packet['upstream_age_ms']
        if not 0 <= age < 2000:
            raise ValueError('相机画面已过期，请等待新画面')
        record_id = uuid.uuid4().hex
        self.root.mkdir(parents=True, exist_ok=True)
        photo = self.root / (record_id+'.jpg')
        # Publish metadata last; interrupted writes never appear as records.
        photo.write_bytes(packet['jpeg'])
        record = dict(kind='operator-declared-empty-background-v1', id=record_id,
            recorded_at=time.time(), source_stream_id=packet['identity'][0],
            source_frame_id=packet['identity'][1], capture_age_ms=age,
            image_sha256=hashlib.sha256(packet['jpeg']).hexdigest(),
            image_size=[packet['bgr'].shape[1],packet['bgr'].shape[0]],
            settings_sha256=settings_sha256, operator_confirmed_empty=True,
            automatically_verified_empty=False, model_trained=False,
            board_baseline_changed=False, image_url='/background/'+record_id+'.jpg')
        (self.root/(record_id+'.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
        return record

    def list(self):
        if not self.root.exists():
            return []
        records=[]
        for path in self.root.glob('*.json'):
            if not re.fullmatch(r'[0-9a-f]{32}',path.stem):
                continue
            record=json.loads(path.read_text(encoding='utf-8'))
            if record['id'] != path.stem:
                raise ValueError('背景记录标识不匹配')
            records.append(record)
        return sorted(records,key=lambda r:r['recorded_at'],reverse=True)[:30]

    def image(self, record_id):
        if not re.fullmatch(r'[0-9a-f]{32}',record_id):
            raise ValueError('无效的背景记录编号')
        record=json.loads((self.root/(record_id+'.json')).read_text(encoding='utf-8'))
        data=(self.root/(record_id+'.jpg')).read_bytes()
        if record['id'] != record_id or hashlib.sha256(data).hexdigest() != record['image_sha256']:
            raise ValueError('背景照片校验失败')
        return data
