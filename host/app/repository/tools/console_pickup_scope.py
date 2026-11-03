"""Advisory pickup membership for the live console; never an execution gate.

The registered source region is projected from a fresh station observation.
Its region-frame ID is deliberately separate from the inference frame ID.
Execution still derives regions and matches object masks on its own fresh frame.
"""
import copy
import json
import threading
import time
from urllib.parse import urlsplit

from voice_dimensions import capture_station_frame,projected_layout,mask_of,pickup_candidate


class PickupScope:
    def __init__(self,layout,root,status,capture=capture_station_frame,clock=time.monotonic):
        self.layout,self.root,self.status,self.capture,self.clock=layout,root,status,capture,clock
        self.lock=threading.Lock();self.cached=None;self.cached_at=0;self.key=None

    def annotate(self,value):
        value=copy.deepcopy(value);instances=value.get('instances',[])
        try:
            state=self.status();binding=state.get('execution_model') or {}
            if (state.get('board_root')!=self.root or value.get('model_version')!=binding.get('model_sha256') or
                value.get('configuration_sha256')!=binding.get('configuration_sha256') or
                value.get('image_size')!=self.layout['image_size'] or not value.get('source_stream_id')):
                raise ValueError('pickup_scope_identity_changed')
            key=(value['source_stream_id'],binding.get('profile_sha256'),binding.get('revision'))
            with self.lock:
                if self.cached is None or key!=self.key or self.clock()-self.cached_at>.8:
                    started=self.clock()
                    _,image,meta=self.capture()
                    if meta['stream']!=value['source_stream_id']:raise ValueError('pickup_scope_camera_changed')
                    layout=projected_layout(self.layout,image,state.get('min_fixed_markers',3))
                    meta=dict(meta,age=meta['age']+(self.clock()-started)*1000)
                    self.cached=(layout,meta);self.cached_at=self.clock();self.key=key
                layout,meta=self.cached
            if meta['age']+(self.clock()-self.cached_at)*1000>2500:raise ValueError('pickup_scope_expired')
            value['pickup_scope']=dict(kind='registered-pickup-region-v1',status='ready',source_roi=layout['source_roi'],
                image_size=value['image_size'],source_stream_id=meta['stream'],region_frame_id=meta['frame'],
                advisory_only=True,control_authorized=False)
            eligible=set();masks=[]
            for index,obj in enumerate(instances):
                mask=mask_of(obj['mask_rle'],value['image_size']);masks.append(mask)
                candidate=pickup_candidate(mask,layout);zone='source' if candidate else 'outside'
                area=int(mask.sum())
                for label,rect in zip(layout.get('bin_labels',[]),layout['bins']):
                    x,y,x1,y1=rect
                    if area and int(mask[y:y1,x:x1].sum())>=area*.95:zone='bin_'+label;break
                obj.update(station_zone=zone,station_grasp_candidate=candidate)
                if candidate:eligible.add(obj.get('instance_id',index))
            points=[]
            for point in value.get('display_grasp_points',[]):
                identifier=point.get('instance_id')
                if identifier is not None:
                    if identifier in eligible:points.append(point)
                elif isinstance(point.get('pixel'),list) and len(point['pixel'])==2:
                    x,y=point['pixel'];w,h=value['image_size']
                    if type(x) in (int,float) and type(y) in (int,float) and 0<=x<w and 0<=y<h and any(
                        obj.get('instance_id',i) in eligible and masks[i][int(y),int(x)] for i,obj in enumerate(instances)):
                        points.append(point)
            value['display_grasp_points']=points
        except (OSError,ValueError,KeyError,TypeError):
            # Preserve all raw classifications while refusing to invent a
            # whole-frame pickup region during an identity/localisation outage.
            value['pickup_scope']=dict(kind='registered-pickup-region-v1',status='unavailable',advisory_only=True,control_authorized=False)
            for obj in instances:obj.update(station_grasp_candidate=False,station_zone='unresolved')
            value['display_grasp_points']=[]
        return value


def attach(server,scope):
    original=server.RequestHandlerClass
    class Handler(original):
        def reply(self,code,raw,mime='application/json'):
            if self.command=='GET' and urlsplit(self.path).path=='/live-result' and code==200:
                value=json.loads(raw) if isinstance(raw,(bytes,str)) else raw
                raw=json.dumps(scope.annotate(value),allow_nan=False).encode()
            return super().reply(code,raw,mime)
    server.RequestHandlerClass=Handler
