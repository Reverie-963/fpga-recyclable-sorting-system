"""Local operator measurement library. No model, motion or device access."""
import copy
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import re
import threading
import uuid


def checked_number(value,low,high):
    if type(value) not in (int,float) or not math.isfinite(value) or not low<=value<=high:
        raise ValueError('尺寸超出允许范围')
    return value


def checked_profile(value):
    fields={'name','pose','material','height_mm','width_mm','official_label','reference_model','features','thumbnail','measurement_source'}
    required=fields-{'width_mm','thumbnail','measurement_source'}
    if not isinstance(value,dict) or set(value)-fields or not required.issubset(value):
        raise ValueError('物品档案格式无效')
    result=copy.deepcopy(value)
    for key,limit in (('name',60),('pose',24),('official_label',120)):
        if not isinstance(result[key],str) or not 1<=len(result[key].strip())<=limit:
            raise ValueError('请填写有效的物品名称和姿态')
        result[key]=result[key].strip()
    if result['material'] not in ('paper','plastic','metal'):raise ValueError('物品材质无效')
    checked_number(result['height_mm'],1,120)
    if result.get('width_mm') is not None:checked_number(result['width_mm'],1,150)
    result.setdefault('width_mm',None)
    if 'measurement_source' in result and (not isinstance(result['measurement_source'],str) or not 1<=len(result['measurement_source'])<=300):
        raise ValueError('尺寸记录来源无效')
    # Historical measurements remain selectable without inventing an image or
    # model identity. Both references must be absent; these never auto-match.
    unbound=result['reference_model'] is None and result['features'] is None
    if not unbound and (not isinstance(result['reference_model'],str) or not re.fullmatch('[a-f0-9]{64}',result['reference_model'])):
        raise ValueError('档案缺少识别来源')
    features=result['features']
    if not unbound and (not isinstance(features,dict) or set(features)!={'rgb','hist','aspect','area'}):
        raise ValueError('物品参考特征无效')
    for key,count in (() if unbound else (('rgb',3),('hist',12))):
        if not isinstance(features[key],list) or len(features[key])!=count:raise ValueError('物品参考特征无效')
        for item in features[key]:checked_number(item,0,1)
    if not unbound:
        checked_number(features['aspect'],1,1000);checked_number(features['area'],0.000001,1)
    thumb=result.get('thumbnail','')
    if not isinstance(thumb,str) or len(thumb)>100000 or (thumb and not re.fullmatch(r'data:image/(?:jpeg|png);base64,[A-Za-z0-9+/]+=*',thumb)):
        raise ValueError('参考照片无效或过大')
    result['thumbnail']=thumb
    return result


class ItemProfileStore:
    def __init__(self,path):self.path=Path(path);self.lock=threading.RLock()

    def read(self):
        with self.lock:
            try:data=json.loads(self.path.read_text(encoding='utf-8'))
            except FileNotFoundError:return dict(schema='item-profiles-v1',revision=0,profiles=[])
            if (not isinstance(data,dict) or data.get('schema')!='item-profiles-v1' or type(data.get('revision')) is not int
                or not isinstance(data.get('profiles'),list) or len(data['profiles'])>128):raise ValueError('物品档案库损坏，请保留文件并检查')
            ids=set()
            for row in data['profiles']:
                if not isinstance(row,dict) or not re.fullmatch('[a-f0-9]{32}',str(row.get('id',''))) or row['id'] in ids:
                    raise ValueError('物品档案库标识无效')
                ids.add(row['id'])
                if type(row.get('revision')) is not int or row['revision']<1:raise ValueError('物品档案版本无效')
                checked_profile({k:v for k,v in row.items() if k not in ('id','revision','updated_at')})
            return copy.deepcopy(data)

    def command(self,payload):
        if not isinstance(payload,dict) or payload.get('action') not in ('save','delete'):raise ValueError('档案操作无效')
        allowed={'action','expected_revision','id','profile'} if payload['action']=='save' else {'action','expected_revision','id'}
        if set(payload)-allowed:raise ValueError('档案字段无效')
        with self.lock:
            data=self.read()
            if type(payload.get('expected_revision')) is not int or payload['expected_revision']!=data['revision']:
                raise ValueError('档案已在其他页面更新，请刷新后再试')
            item_id=payload.get('id')
            existing=next((r for r in data['profiles'] if r['id']==item_id),None)
            if item_id is not None and existing is None:raise ValueError('物品档案不存在')
            if payload['action']=='delete':
                if existing is None:raise ValueError('请选择要移除的物品档案')
                data['profiles']=[r for r in data['profiles'] if r['id']!=item_id]
            else:
                row=checked_profile(payload.get('profile'))
                if existing is None and len(data['profiles'])>=128:raise ValueError('档案最多保存128项')
                row.update(id=item_id or uuid.uuid4().hex,revision=(existing['revision']+1 if existing else 1),updated_at=datetime.now(timezone.utc).isoformat())
                data['profiles']=[r for r in data['profiles'] if r['id']!=row['id']]+[row]
            data['revision']+=1
            self.path.parent.mkdir(parents=True,exist_ok=True)
            temporary=self.path.with_name(self.path.name+'.'+uuid.uuid4().hex+'.tmp')
            try:
                temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
                os.replace(temporary,self.path)
            finally:
                if temporary.exists():temporary.unlink()
            return copy.deepcopy(data)
