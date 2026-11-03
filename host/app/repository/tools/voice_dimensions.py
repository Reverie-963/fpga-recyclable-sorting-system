"""Read-only dimensions for one explicit ASR request, using the console library.

No action, heartbeat, registration or motion is submitted by this PC service.
The native controller still binds the returned masks to its fresh observation.
"""
import base64
import copy
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import posixpath
import sys
import threading
import time
import urllib.request
import uuid

from item_profile_store import ItemProfileStore


def mask_of(runs, size):
    import numpy as np
    if (not isinstance(size,list) or len(size)!=2 or any(type(n) is not int or not 1<=n<=v for n,v in zip(size,(640,480))) or
        not isinstance(runs,list) or not 1<=len(runs)<=8192 or any(type(n) is not int or n<0 for n in runs) or sum(runs)!=size[0]*size[1]):
        raise ValueError('voice_invalid_mask')
    return np.repeat(np.arange(len(runs))%2,runs).astype(bool).reshape(size[1],size[0])


def features(rgba, runs, size):
    """Same sampling, upper median and thresholds as ui/item_profiles.js."""
    import numpy as np
    mask=mask_of(runs,size);ys,xs=np.nonzero(mask);count=len(xs)
    if count<4 or list(rgba.shape)!=[size[1],size[0],4]:raise ValueError('voice_reference_image_mismatch')
    sample=np.arange(0,count,max(1,count//2048));x,y=xs[sample],ys[sample]
    if count>30:
        keep=(x>0)&(y>0)&(x<size[0]-1)&(y<size[1]-1);x,y=x[keep],y[keep]
        keep=mask[y,x-1]&mask[y,x+1]&mask[y-1,x]&mask[y+1,x];x,y=x[keep],y[keep]
    if not len(x):raise ValueError('voice_reference_mask_too_thin')
    rgb=rgba[y,x,:3].astype(float)/255;hist=[0]*12
    for r,g,b in rgb:
        hi,lo=max(r,g,b),min(r,g,b);d=hi-lo
        hue=0 if d==0 else (g-b)/d if hi==r else (b-r)/d+2 if hi==g else (r-g)/d+4
        hist[min(11,int(((hue+6)%6)*2))]+=1
    xx=float((xs.astype(float)**2).sum()/count-(xs.sum()/count)**2)
    yy=float((ys.astype(float)**2).sum()/count-(ys.sum()/count)**2)
    xy=float((xs.astype(float)*ys).sum()/count-xs.sum()*ys.sum()/count**2)
    delta=math.sqrt((xx-yy)**2+4*xy*xy);large,small=(xx+yy+delta)/2,(xx+yy-delta)/2
    return dict(rgb=np.sort(rgb,axis=0)[len(rgb)//2].tolist(),hist=[n/len(rgb) for n in hist],
        aspect=min(1000,math.sqrt(max(large,.1)/max(small,.1))),area=count/(size[0]*size[1]))


def match(profiles,query,material,model):
    rows=[]
    for p in profiles:
        f=p.get('features')
        if not f or p['material']!=material or p['reference_model']!=model:continue
        area=query['area']/f['area'];aspect=max(query['aspect']/f['aspect'],f['aspect']/query['aspect'])
        color=math.sqrt(sum((a-b)**2 for a,b in zip(f['rgb'],query['rgb']))/3)
        hist=sum(abs(a-b) for a,b in zip(f['hist'],query['hist']))/2
        if .85<=area<=1.18 and aspect<=1.2 and color<=.10 and hist<=.2:
            rows.append((color*.6+hist*.3+abs(math.log(area))*.1,p))
    rows.sort(key=lambda row:row[0])
    return rows[0][1] if rows and (len(rows)==1 or rows[1][0]-rows[0][0]>=.04) else None


def bound_layout(client,root):
    """Read the hash-bound layout, without opening a camera or UART."""
    ftp=client.open_sftp()
    try:
        def read(path):
            with ftp.open(path,'rb') as stream:raw=stream.read(2_000_001)
            if len(raw)>2_000_000:raise ValueError('voice_layout_too_large')
            return raw
        folder=root+'/local-only/station-grasp-integration';pointer=json.loads(read(folder+'/active.json'))
        binding=posixpath.normpath(folder+'/'+pointer['path']);raw=read(binding)
        if hashlib.sha256(raw).hexdigest()!=pointer['sha256']:raise ValueError('voice_binding_changed')
        bound=json.loads(raw);ref=bound['files']['config'];raw=read(posixpath.normpath(posixpath.dirname(binding)+'/'+ref['path']))
        if hashlib.sha256(raw).hexdigest()!=ref['sha256']:raise ValueError('voice_layout_changed')
        c=json.loads(raw)
        references={}
        for name in ('camera','station','scene','registration','unified_policy'):
            record=bound['files'][name];body=read(posixpath.normpath(posixpath.dirname(binding)+'/'+record['path']))
            if hashlib.sha256(body).hexdigest()!=record['sha256']:raise ValueError('voice_geometry_reference_changed')
            references[name]=json.loads(body)
        review=bound.get('visual_review_policy');factor=1
        if review is not None:
            factor=review.get('tolerance_factor')
            if (review.get('kind')!='operator-vision-review-margin-v1' or not review.get('operator_statement') or
                review.get('preserve_timings') is not True or type(factor) not in (int,float) or not math.isfinite(factor) or not 1<=factor<=3):
                raise ValueError('voice_invalid_bound_review_policy')
        return dict(image_size=c['image_size'],source_roi=c['source_roi'],
            bins=[p['bin_roi'] for p in c['profiles'].values()],bin_labels=list(c['profiles']),static_rois=c['layout']['static_rois'],sha256=ref['sha256'],
            highest_mm=max(p['object_height_mm'] for p in c['profiles'].values()),
            review_factor=factor,**references)
    finally:ftp.close()


def source_frame(vision):
    """Fetch the exact JPEG inference pixels, never a newer camera frame."""
    import cv2
    import numpy as np
    identifier=str(vision['source_frame_id'])
    if not identifier.isdigit():raise ValueError('voice_invalid_source_frame')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open('http://127.0.0.1:8780/frame.jpg?frame_id='+identifier,timeout=4) as response:
        raw=response.read(2_000_001);headers=response.headers
    if (len(raw)>2_000_000 or headers.get('X-Frame-Id')!=identifier or
        headers.get('X-Stream-Id')!=vision['source_stream_id'] or float(headers.get('X-Frame-Age-Ms','inf'))>2500):
        raise ValueError('voice_source_frame_changed')
    image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
    if image is None or list(image.shape[1::-1])!=vision['image_size']:raise ValueError('voice_source_image_changed')
    digest=hashlib.sha256(image.tobytes()).hexdigest()
    if digest!=vision.get('source_bgr_sha256'):raise ValueError('voice_source_pixels_changed')
    return image


def capture_station_frame():
    """Read-only frame/identity source shared by UI regions and voice matching."""
    import cv2
    import numpy as np
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open('http://127.0.0.1:8780/snapshot.png',timeout=4) as response:
        raw=response.read(2_000_001);headers=response.headers
    if len(raw)>2_000_000:raise ValueError('voice_source_image_too_large')
    image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
    if image is None:raise ValueError('voice_source_image_unavailable')
    digest=hashlib.sha256(image.tobytes()).hexdigest()
    if digest!=headers.get('X-Decoded-Bgr-Sha256'):raise ValueError('voice_source_pixels_changed')
    stream,frame=headers.get('X-Stream-Id'),headers.get('X-Frame-Id')
    age=float(headers.get('X-Frame-Age-Ms','inf'))
    if not stream or not frame or not 0<=age<=2500:raise ValueError('voice_source_frame_expired')
    return raw,image,dict(digest=digest,stream=stream,frame=frame,age=age)


def fresh_observation(predict):
    """One image-only inference, bound to its freshly captured PNG pixels."""
    started=time.monotonic();raw,image,meta=capture_station_frame()
    digest,stream,frame,age=(meta[k] for k in ('digest','stream','frame','age'))
    result=predict(dict(image_base64=base64.b64encode(raw).decode()))
    if result.get('source_bgr_sha256')!=digest:raise ValueError('voice_inference_pixels_changed')
    result.update(source_stream_id=stream,source_frame_id=frame,frame_id=frame,
        result_age_ms=age+(time.monotonic()-started)*1000)
    return result,image


def projected_layout(layout,image,minimum_markers):
    """Read-only image-region projection; all motion geometry stays on PS."""
    import cv2
    import numpy as np
    project=Path(__file__).resolve().parents[1]/'pynq_km1_vision'
    if str(project) not in sys.path:sys.path.insert(0,str(project))
    from src.station_geometry import detect_markers,estimate_pose,world_to_pixels,fingerprint
    camera,station=layout['camera'],layout['station']
    pose=estimate_pose(detect_markers(image,camera),camera,station,minimum_markers,layout['review_factor'])
    pose=dict(pose,kind='station-pose-preview-v1',camera_sha256=fingerprint(camera),station_sha256=fingerprint(station))
    def rectangle(points,inset=False):
        p=np.asarray(points,float);size=np.asarray(layout['image_size'])
        low=np.maximum(0,np.ceil(p.min(0)) if inset else np.floor(p.min(0))).astype(int)
        high=np.minimum(size,(np.floor(p.max(0)) if inset else np.ceil(p.max(0)))+1).astype(int)
        if inset:
            for _ in range(int(max(size))):
                corners=((low[0],low[1]),(high[0],low[1]),(high[0],high[1]),(low[0],high[1]))
                if all(cv2.pointPolygonTest(p.astype(np.float32),tuple(map(float,q)),False)>=0 for q in corners):break
                low+=1;high-=1
        if not (high-low>=8).all():raise ValueError('voice_projected_region_unusable')
        return [int(low[0]),int(low[1]),int(high[0]),int(high[1])]
    regions={}
    for name,record in layout['scene']['scene_regions'].items():
        xyz=np.asarray(record['world_xyz_mm'],float)
        if name=='source':xyz=np.vstack((xyz,np.c_[xyz[:,:2],np.full(len(xyz),layout['highest_mm'])]))
        regions[name]=rectangle(world_to_pixels(xyz,pose,camera,station),name!='source')
    if layout.get('unified_policy') and layout.get('registration'):
        from src.station_geometry import map_xy
        poly=layout['unified_policy']['source_card_polygon_mm'];inverse=np.linalg.inv(np.asarray(layout['registration']['world_to_card_xy'],float))
        world=np.asarray([map_xy(inverse,p) for p in poly]);highest=max(v['object_height_mm'] for v in layout['unified_policy']['models'].values())
        xyz=np.vstack([np.c_[world,np.full(len(world),z)] for z in (0,highest)])
        regions['source']=rectangle(world_to_pixels(xyz,pose,camera,station))
    return dict(layout,source_roi=regions['source'],bins=[r for name,r in regions.items() if name!='source'],
        bin_labels=[name.removeprefix('bin_') for name in regions if name!='source'])


def pickup_candidate(mask,layout):
    # Conservatively exclude registered bins and fixed patches. This is only a
    # measurement filter; the board's adjusted geometry remains authoritative.
    def pixels(rect):
        x,y,x1,y1=rect
        return int(mask[y:y1,x:x1].sum())
    area=int(mask.sum())
    if not area or any(pixels(r)>=area*.95 for r in layout['bins']):return False
    if any(pixels(r) for r in layout['static_rois']):return False
    return pixels(layout['source_roi'])>0


class Dimensions:
    def __init__(self,request,store,layout,board_root,frame_loader=None,observation_loader=None):
        self.request,self.store,self.layout,self.board_root=request,store,layout,board_root
        self.frame_loader=frame_loader
        self.observation_loader=observation_loader

    def prepare(self,data):
        if (not isinstance(data,dict) or set(data)!={'request_id','material','profile_sha256','revision'} or
            data['material'] not in ('AUTO','PAPER','PLASTIC','METAL')):raise ValueError('voice_invalid_preparation')
        if data['request_id'] not in (str(uuid.UUID(data['request_id'])),uuid.UUID(data['request_id']).hex):raise ValueError('voice_invalid_request_id')
        started=time.monotonic();before=self.request('/operation');binding=before.get('execution_model') or {}
        def check(state):
            if state.get('board_root')!=self.board_root or state.get('compute_location')!='pynq_ps':raise ValueError('voice_controller_changed')
            if state.get('busy') or state.get('blockers'):raise ValueError('voice_station_not_ready')
            b=state.get('execution_model') or {}
            if b.get('profile_sha256')!=data['profile_sha256'] or b.get('revision')!=data['revision'] or b!=binding:
                raise ValueError('voice_model_changed')
        check(before);library=self.store.read();source=None
        if self.observation_loader is None:vision=self.request('/live-result')
        else:vision,source=self.observation_loader()
        observed_at=time.monotonic()
        age=vision.get('result_age_ms')
        if type(age) not in (int,float) or not math.isfinite(age) or not 0<=age<=2500:raise ValueError('voice_frame_expired')
        if (vision.get('model_version')!=binding.get('model_sha256') or vision.get('configuration_sha256')!=binding.get('configuration_sha256') or
            vision.get('image_size')!=self.layout['image_size'] or not vision.get('source_stream_id') or not vision.get('frame_id')):
            raise ValueError('voice_frame_identity_changed')
        layout=self.layout
        if source is not None or self.frame_loader is not None:
            layout=projected_layout(layout,source if source is not None else self.frame_loader(vision),before.get('min_fixed_markers',3))
        import cv2
        import numpy as np
        image=cv2.imdecode(np.frombuffer(base64.b64decode(vision['overlay_base64'],validate=True),np.uint8),cv2.IMREAD_COLOR)
        if image is None:raise ValueError('voice_reference_image_unavailable')
        rgba=cv2.cvtColor(image,cv2.COLOR_BGR2RGBA);items=[];target_found=False
        for obj in vision.get('instances',[]):
            label=obj.get('label');size=vision['image_size']
            if label not in ('paper','metal','plastic'):continue
            mask=mask_of(obj['mask_rle'],size)
            if not pickup_candidate(mask,layout) or obj.get('station_grasp_candidate') is False or str(obj.get('station_zone','')).startswith('bin_'):continue
            if data['material']=='AUTO' or label==data['material'].lower():target_found=True
            # Heights of untargeted source objects are also required for the
            # board's neighbour/scene geometry. Material selection only chooses
            # which registered objects move.
            p=match(library['profiles'],features(rgba,obj['mask_rle'],size),label,vision['model_version'])
            if p is None:raise ValueError('voice_dimensions_missing_or_ambiguous:'+str(obj.get('official_label','unknown')))
            item=dict(item_id=len(items)+1,label=label,official_label=obj['official_label'],mask_rle=copy.deepcopy(obj['mask_rle']),
                object_height_mm=p['height_mm'],measurement_profile={k:p[k] for k in ('id','revision','name','pose')})
            if p.get('width_mm') is not None:
                if before.get('per_instance_widths_supported') is not True:raise ValueError('voice_measured_width_not_supported')
                item['grasp_width_mm']=p['width_mm']
            items.append(item)
        if not target_found or not 1<=len(items)<=12:raise ValueError('voice_no_registered_source_objects')
        check(self.request('/operation'))
        if self.store.read()['revision']!=library['revision']:raise ValueError('voice_library_changed')
        if age+(time.monotonic()-observed_at)*1000>2500:raise ValueError('voice_frame_expired_during_matching')
        plan=dict(kind='instance-heights-v1',model_sha256=vision['model_version'],configuration_sha256=vision['configuration_sha256'],
            source_stream_id=vision['source_stream_id'],image_size=vision['image_size'],items=items)
        return dict(request_id=data['request_id'],profile_sha256=binding['profile_sha256'],revision=binding['revision'],height_plan=plan,
            library_revision=library['revision'],frame_id=vision['frame_id'],layout_sha256=self.layout['sha256'],physical_request_sent=False)


def make_server(port,token,service):
    gate=threading.Lock();seen=set()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,status,value):
            raw=json.dumps(value,allow_nan=False).encode();self.send_response(status)
            self.send_header('Content-Type','application/json');self.send_header('Cache-Control','no-store')
            self.send_header('Content-Length',str(len(raw)));self.end_headers()
            try:self.wfile.write(raw)
            except OSError:pass
        def authorized(self):return hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+token)
        def do_GET(self):
            if not self.authorized():return self.reply(403,dict(error='voice_dimensions_authentication_required'))
            return self.reply(200,dict(kind='voice-library-dimensions-v1',automatic_motion=False)) if self.path=='/health' else self.reply(404,{})
        def do_POST(self):
            if not self.authorized():return self.reply(403,dict(error='voice_dimensions_authentication_required'))
            if self.path!='/prepare':return self.reply(404,{})
            if not gate.acquire(False):return self.reply(409,dict(error='voice_dimensions_busy'))
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=1024 or self.headers.get('Transfer-Encoding'):raise ValueError('voice_dimensions_invalid_body')
                data=json.loads(self.rfile.read(length));key=data.get('request_id')
                if not isinstance(key,str) or key in seen or len(seen)>=1000:raise ValueError('voice_preparation_no_replay')
                seen.add(key)
                return self.reply(200,service.prepare(data))
            except (OSError,ValueError,KeyError,TypeError) as exc:return self.reply(409,dict(error=str(exc)[:180],physical_request_sent=False))
            finally:gate.release()
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)
