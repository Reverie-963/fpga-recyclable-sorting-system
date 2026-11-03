"""Board-local console, original supervised controller, and durable ownership.

The PC supplies authenticated visual results only. Starting this server never
opens an actuator; an explicit operation with the existing scene checks is needed.
Runtime source amendments are frozen; original qualification evidence is retained.
"""
import argparse
import copy
import hashlib
import hmac
import json
import os
from pathlib import Path
import signal
import threading
import time
import urllib.request
from urllib.parse import urlsplit
from contextlib import contextmanager,nullcontext

from .board_control_ownership import BoardOwnership
from .motion_sequence import scene_check_mode
from .execution_model_context import current_profile,model_context


class StationAuthorization(urllib.request.BaseHandler):
    def __init__(self,token,url='http://127.0.0.1:8783'):
        parsed=urlsplit(url)
        if (parsed.scheme!='http' or parsed.hostname not in ('127.0.0.1','localhost') or
            not parsed.port or parsed.path not in ('','/') or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError('local_station_transport_required')
        self.origin=(parsed.scheme,parsed.hostname,parsed.port);self.token=token
    def http_request(self,request):
        parsed=urlsplit(request.full_url)
        if (parsed.scheme,parsed.hostname,parsed.port)==self.origin:
            request.add_unredirected_header('Authorization','Bearer '+self.token)
        return request


class ReadinessView:
    """Display snapshots never replace the fresh operation-admission check."""
    def __init__(self):
        self.local=threading.local();self.lock=threading.Lock()
        self.value=['board_readiness_refresh_pending']
    @contextmanager
    def display(self):
        previous=getattr(self.local,'display',False);self.local.display=True
        try:yield
        finally:self.local.display=previous
    def check(self,fresh):
        if getattr(self.local,'display',False):
            with self.lock:return list(self.value)
        result=fresh()
        with self.lock:self.value=list(result)
        return result


def renew_operation_heartbeat(operations,data,authority,clock=time.monotonic):
    # No model/file/readiness work in this acknowledgement. Only the client that
    # started the operation can renew the unchanged twelve-second leases.
    with operations.lock:
        if data.get('token')!=operations.token:raise ValueError('page_expired_reload')
        if data.get('owner')!=operations.owner or data.get('operation_id')!=operations.operation_id:
            raise ValueError('not_operation_owner')
        if authority.lease is not None:authority.ownership.check(authority.lease,heartbeat=True)
        operations.last_heartbeat=clock()
        return dict(heartbeat_accepted=True,operation_id=operations.operation_id)


def preview_display(report):
    if report.get('compute_location')!='pynq_ps':raise ValueError('board_preview_identity_required')
    ready=report.get('status')=='preview_only' and isinstance(report.get('choice'),dict) and bool(report.get('plan'))
    return dict(status='preview_ready' if ready else report.get('status','blocked'),reason=report.get('reason'),
        candidates=[dict(choice=report['choice'],steps=report['plan'])] if ready else [],
        rejected=report.get('rejected_candidates',[]),decisions=report.get('original_labels_by_frame',[]),
        physical_motion=False,compute_location='pynq_ps')


def initial_decision_thresholds(options,model_scope,read_settings):
    """Initialize against the same selected profile used by later operations."""
    with model_scope():
        return dict(options['decision_thresholds'],min_confidence=read_settings()[0]['confidence'])


class Authority:
    def __init__(self, ownership, source_check=lambda:None, max_frame_age_ms=None, clock=time.monotonic,
                 motion_scene_check_mode='per_command'):
        self.ownership=ownership
        self.source_check=source_check;self.max_frame_age_ms=max_frame_age_ms;self.clock=clock
        self.last_capture_ms=None
        self.lease=None
        self.port_attempted=False
        self.write_attempted=False
        if motion_scene_check_mode not in ('per_command','before_sequence'):
            raise ValueError('invalid_motion_scene_check_mode')
        self.motion_scene_check_mode=motion_scene_check_mode
        self.sequence_started=False

    def check(self):
        if self.lease is None:raise ValueError('board_operation_ownership_required')
        self.ownership.check(self.lease)

    def check_capture(self):
        if self.max_frame_age_ms is not None and (self.last_capture_ms is None or
                not 0<=self.clock()*1000-self.last_capture_ms<=self.max_frame_age_ms):
            raise ValueError('board_prewrite_capture_expired')

    def begin(self, owner, request_id):
        if self.lease is not None:raise ValueError('board_operation_already_running')
        self.lease=self.ownership.claim(owner,request_id)
        self.port_attempted=self.write_attempted=False
        self.sequence_started=False

    def finish(self, report=None, failed=False):
        if self.lease is None:return
        lease,self.lease=self.lease,None
        physical=self.port_attempted or self.write_attempted
        status=(report or {}).get('status')
        jaw_completed=(status=='jaw_calibration_command_completed' and
            (report or {}).get('mode')=='supervised_jaw_calibration' and
            len((report or {}).get('steps',[]))==1 and
            (report or {})['steps'][0].get('transport_receipt',{}).get('echo_matched') is True and
            (report or {})['steps'][0].get('transport_receipt',{}).get('completed_movement_wait') is True)
        outcome=('completed' if not failed and (status=='completed_visual_evidence' or jaw_completed) else
                 'needs_review' if physical else 'rejected_no_motion')
        try:self.ownership.finish(lease,outcome)
        except ValueError:
            # Heartbeat/stop/restart remains blocked; a worker cannot clear it.
            if self.ownership.status()['status']!='blocked':raise


def translated_arm(arm,mapping):
    device=mapping['arm_device']
    if (arm['port']!=mapping['qualified_port'] or arm['usb_identity']!=mapping['usb_identity'] or
        not device.startswith('/dev/ttyUSB') or not device.removeprefix('/dev/ttyUSB').isdigit()):
        raise ValueError('board_arm_transport_mapping_changed')
    result=copy.deepcopy(arm);result['port']=device
    return result


def guard_connection_type(base, authority):
    """Keep original serial setup, byte validation, echo and complete waits."""
    class GuardedConnection(base):
        def open(self):
            authority.source_check()
            authority.check()
            authority.port_attempted=True
            return super().open()

        def send(self,data,duration,guard=None):
            first=not authority.sequence_started
            if authority.motion_scene_check_mode=='per_command' or first:
                authority.source_check()
            authority.check()
            def before_write():
                if guard:guard()
                authority.check()
                if authority.motion_scene_check_mode=='per_command' or first:
                    authority.check_capture()
                authority.sequence_started=True
                authority.write_attempted=True
            return super().send(data,duration,before_write)

        def request(self,line):
            authority.check()
            authority.write_attempted=True
            return super().request(line)
    return GuardedConnection


class VisualInputs:
    def __init__(self,url,token,model,policy_hash,camera_hash,authority):
        self.url=url.rstrip('/');self.token=token;self.model=model
        self.policy_hash,self.camera_hash=policy_hash,camera_hash
        self.authority=authority;self.expected=None;self._frame_png=None;self.markers_only_supported=False
        self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(self,path,payload=None):
        if path!='/health':self.authority.check()
        headers={'Authorization':'Bearer '+self.token,'Content-Type':'application/json'}
        active=current_profile()
        if active is not None:
            headers['X-Execution-Profile']=active['id']
            if active.get('route_id'):headers['X-Execution-Route']=active['route_id']
        request=urllib.request.Request(self.url+path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers=headers)
        with self.opener.open(request,timeout=6) as response:raw=response.read(8_500_001)
        if len(raw)>8_500_000:raise ValueError('board_visual_reply_too_large')
        if path!='/health':self.authority.check()
        return json.loads(raw)

    def freeze(self,settings):
        self.authority.check()
        health=self.request('/health')
        self.authority.check()
        active=current_profile();model=active['policy']['model'] if active is not None else self.model
        if (health.get('ready') is not True or health.get('settings')!=settings or
            health.get('model_hashes',{}).get(model['name'])!=model['sha256'] or
            active is not None and health.get('execution_profile',{}).get('profile_sha256')!=active['profile_sha256']):
            raise ValueError('board_current_visual_identity_changed')
        self._frame_png=None
        self.markers_only_supported=health.get('markers_only_supported') is True
        self.expected={k:copy.deepcopy(health.get(k)) for k in
            ('settings','generation','model_hashes','execution','station_filter_config_sha256','execution_profile')}

    def predict(self,image,encoded_png=None):
        import cv2
        import base64
        if self.expected is None:raise ValueError('board_visual_snapshot_required')
        if encoded_png is None:
            ok,png=cv2.imencode('.png',image)
            if not ok:raise ValueError('cannot_encode_board_visual_input')
            encoded_png=png.tobytes()
        if not isinstance(encoded_png,bytes) or not encoded_png.startswith(b'\x89PNG\r\n\x1a\n') or len(encoded_png)>8500000:
            raise ValueError('invalid_board_visual_png')
        result=self.request('/detect',dict(image_base64=base64.b64encode(encoded_png).decode(),control_only=True))
        digest=hashlib.sha256(image.tobytes()).hexdigest()
        active=current_profile();model=active['policy']['model'] if active is not None else self.model
        if (result.get('source_bgr_sha256')!=digest or result.get('image_size')!=list(image.shape[1::-1]) or
            result.get('model_sha256')!=model['sha256'] or
            result.get('settings_sha256')!=self.expected['settings']['settings_sha256'] or
            result.get('policy_sha256')!=self.expected['settings']['policy_sha256'] or
            result.get('execution_profile')!=self.expected.get('execution_profile') or
            result.get('generation')!=self.expected['generation']):
            raise ValueError('board_visual_result_identity_changed')
        markers=result.get('fixed_marker_observation',{})
        if (markers.get('source_bgr_sha256')!=digest or markers.get('camera_sha256')!=self.camera_hash or
            markers.get('location')!='pc_vision' or not isinstance(markers.get('detections'),list)):
            raise ValueError('board_visual_markers_changed')
        filter_hash=self.expected.get('station_filter_config_sha256')
        if filter_hash and result.get('station_filter',{}).get('config_sha256')!=filter_hash:
            raise ValueError('board_visual_filter_changed')
        result['model_input_png_sha256']=hashlib.sha256(encoded_png).hexdigest()
        return result

    def remember_frame(self,row):
        png=row.get('source_png_bytes')
        if png is not None:
            if hashlib.sha256(png).hexdigest()!=row['observation']['image_sha256'] or hashlib.sha256(row['image'].tobytes()).hexdigest()!=row['bgr_sha256']:
                raise ValueError('board_visual_frame_pixels_changed')
            self._frame_png=(row['bgr_sha256'],png,row['observation']['image_sha256'])

    def markers(self,image):
        if not self.markers_only_supported:return self.predict(image)['fixed_marker_observation']['detections']
        import base64,cv2
        digest=hashlib.sha256(image.tobytes()).hexdigest()
        png=self._frame_png[1] if self._frame_png is not None and self._frame_png[0]==digest else None
        if png is None:
            ok,encoded=cv2.imencode('.png',image)
            if not ok:raise ValueError('cannot_encode_board_marker_input')
            png=encoded.tobytes()
        result=self.request('/markers',dict(image_base64=base64.b64encode(png).decode(),control_only=True))
        active=current_profile();model=active['policy']['model'] if active is not None else self.model
        markers=result.get('fixed_marker_observation',{})
        if (self.expected is None or result.get('source_bgr_sha256')!=digest or result.get('image_size')!=list(image.shape[1::-1]) or
            result.get('model_sha256')!=model['sha256'] or result.get('generation')!=self.expected['generation'] or
            result.get('settings_sha256')!=self.expected['settings']['settings_sha256'] or
            result.get('policy_sha256')!=self.expected['settings']['policy_sha256'] or
            result.get('execution_profile')!=self.expected.get('execution_profile') or
            result.get('scope')!='fixed_markers_only' or result.get('semantic_inference_performed') is not False or
            markers.get('source_bgr_sha256')!=digest or markers.get('camera_sha256')!=self.camera_hash or
            markers.get('location')!='pc_vision' or not isinstance(markers.get('detections'),list)):
            raise ValueError('board_visual_marker_identity_changed')
        return markers['detections']

    def predict_frame(self,row):
        image=row['image'];png=row.get('source_png_bytes')
        cached=self._frame_png
        if not (png is not None and cached is not None and len(cached)==3 and cached[1] is png and
                cached[0]==row['bgr_sha256'] and cached[2]==row['observation']['image_sha256']):
            self.remember_frame(row)
        # predict() still checks the decoded model result against the current
        # image hash. Only an identical immutable PNG object reuses its proof.
        return self.predict(image,encoded_png=png)

    def verify(self):
        self.authority.check()
        health=self.request('/health')
        self.authority.check()
        if any(health.get(k)!=v for k,v in self.expected.items()):
            raise ValueError('board_visual_identity_changed_during_operation')


def guard_camera_capture(camera_type,authority,visual=None):
    """Bind both semantic next() and raw() to their actual validated capture."""
    original=camera_type._raw
    def captured(self,deadline):
        authority.check()
        row=original(self,deadline)
        authority.check()
        authority.last_capture_ms=row['observation']['capture_started_ms']
        if visual is not None:visual.remember_frame(row)
        return row
    camera_type._raw=captured


def install_board_adapters(authority,visual,station_token,camera,station_url,arm_mapping):
    """Compose transport inputs around original controller implementations.

    This dedicated process admits one operation. No installed package, original
    source file, mechanical threshold or historical qualification is rewritten.
    """
    import base64
    import cv2
    from . import arm_transport,sorting_station_io,sorting_prompted_motion,station_geometry,sorting_io
    from . import sorting_unified_run
    from .prompted_observation import single_frame_pose
    from .sorting_loop import need,mask_of
    urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({}),StationAuthorization(station_token,station_url)))

    original_request=sorting_station_io.StationStatistics.request
    def station_request(self,route,body=None):
        if body is not None:authority.check()
        need(self.cfg['url'].rstrip('/')==station_url,'board_station_transport_changed')
        request=urllib.request.Request(station_url+route,
            data=json.dumps(body).encode() if body is not None else None,
            headers={'Authorization':'Bearer '+station_token,'Content-Type':'application/json'})
        with self.opener.open(request,timeout=self.cfg.get('timeout_s',3)) as reply:raw=reply.read(128001)
        need(len(raw)<=128000,'oversized_station_response')
        return json.loads(raw)
    sorting_station_io.StationStatistics.request=station_request
    original_statistics_init=sorting_station_io.StationStatistics.__init__
    def statistics_init(self,*args,**kwargs):
        original_statistics_init(self,*args,**kwargs)
        # bind/verification retrieve PNG bytes through this instance opener,
        # bypassing request(). Keep the same origin-scoped authentication there.
        self.opener.add_handler(StationAuthorization(station_token,station_url))
    sorting_station_io.StationStatistics.__init__=statistics_init
    original_camera_init=sorting_station_io.StationCamera.__init__
    def camera_init(self,*args,**kwargs):
        original_camera_init(self,*args,**kwargs)
        self.opener.add_handler(StationAuthorization(station_token,station_url))
    sorting_station_io.StationCamera.__init__=camera_init
    guard_camera_capture(sorting_station_io.StationCamera,authority,visual)
    original_provenance=sorting_station_io.StationStatistics.provenance
    def provenance(self):
        result=original_provenance(self)
        result.update(geometry_backend='pynq_ps_numpy',semantic_mask_statistics_backend='pynq_ps_numpy',
                      compute_location='pynq_ps',perception_location='pc')
        return result
    sorting_station_io.StationStatistics.provenance=provenance

    def remote_markers(image,parameters):
        need(parameters==camera,'board_marker_camera_parameters_changed')
        authority.check()
        result=visual.markers(image)
        authority.check()
        return result
    station_geometry.detect_markers=remote_markers
    from . import sorting_neutral_return
    sorting_neutral_return.detect_markers=remote_markers

    def semantic_observation(image,result):
        result=json.loads(json.dumps({k:v for k,v in result.items() if k!='overlay_base64'},allow_nan=False))
        for obj in result['instances']:
            need(obj['material'] in ('paper','plastic','metal'),'unknown_prompted_motion_material')
            mask_of(obj,result['image_size']);obj['label']=obj['material']
        input_sha=result.get('model_input_png_sha256')
        if input_sha is None:
            ok,png=cv2.imencode('.png',image);need(ok,'cannot_encode_prompted_control_image')
            input_sha=hashlib.sha256(png).hexdigest()
        return dict(result,model_version=result['model_sha256'],runtime_id=result['settings_sha256'],
                    prompt_mode='text',image_sha256=input_sha)

    def raw_window(c,cam,station,registration,scene=None,**kwargs):
        authority.check()
        from .board_prompted_runtime import capture_config,locate
        stats_cfg=copy.deepcopy(c);stats_cfg['statistics']['url']=station_url
        request=urllib.request.Request(station_url+'/health',headers={'Authorization':'Bearer '+station_token})
        with urllib.request.urlopen(request,timeout=3) as reply:health=json.load(reply)
        capture=capture_config(stats_cfg,health)
        stats=sorting_station_io.StationStatistics(capture)
        source=sorting_station_io.StationCamera(capture,None,stats)
        rows=[]
        started=time.monotonic()
        count=c['stable_frames']
        for _ in range(count):
            row=source.raw(c['frame_timeout_s'])
            result=visual.predict_frame(row)
            row['observation']['fixed_marker_observation']=result['fixed_marker_observation']
            prepared=semantic_observation(row['image'],result)
            prepared['model_input_image_sha256']=prepared['image_sha256']
            prepared.update(row['observation'])
            row['_prepared_observation']=prepared
            rows.append(row)
        need(time.monotonic()-started<=c['frame_timeout_s'],'board_initial_window_expired')
        need(all(0<=time.monotonic()*1000-r['observation']['capture_started_ms']<=c['max_age_ms']
                 for r in rows),'board_initial_frame_expired')
        binding={'motion_guard_policy':{'min_fixed_markers':kwargs.get('min_fixed_markers',3),
            'initial_max_displacement_px':kwargs.get('max_marker_shift_px',.5)},
            'review_policy':{'tolerance_factor':kwargs.get('review_tolerance_factor',1)}}
        if count==1:
            r=rows[0]
            record=dict(image_sha256=r['source_bytes_sha256'],
                detections=r['observation']['fixed_marker_observation']['detections'],
                source_frame_id=int(r['observation']['source_frame_id']),source_stream_id=r['observation']['source_stream_id'])
            pose=single_frame_pose(record,cam,station,kwargs.get('min_fixed_markers',3),kwargs.get('review_tolerance_factor',1))
        else:
            raise ValueError('board_host_current_policy_requires_single_frame')
        return rows,pose,stats
    sorting_prompted_motion.raw_prompted_window=raw_window

    class Detector:
        def __init__(self,settings,url=None):self.settings=settings
        def verify_identity(self):visual.verify()
        @staticmethod
        def frame_marker_detector(image,parameters):
            need(parameters==camera,'board_marker_camera_parameters_changed')
            return visual.markers(image)
        def predict_frame(self,row,prompt_mode='text'):
            need(prompt_mode=='text','prompted_motion_requires_text')
            return semantic_observation(row['image'],visual.predict_frame(row))
        def predict(self,image,prompt_mode='text'):
            need(prompt_mode=='text','prompted_motion_requires_text')
            return semantic_observation(image,visual.predict(image))

    sorting_prompted_motion.PromptedHTTP=Detector
    original_lease=sorting_unified_run.BatchLease
    class OwnedLease(original_lease):
        def check(self):
            authority.check();return super().check()
    sorting_prompted_motion.BatchLease=OwnedLease
    sorting_neutral_return.BatchLease=OwnedLease
    original_motion_init=sorting_io.Motion.__init__
    def motion_init(self,arm,stop_file):
        original_motion_init(self,translated_arm(arm,arm_mapping),stop_file)
    sorting_io.Motion.__init__=motion_init
    arm_transport.ArmConnection=guard_connection_type(arm_transport.ArmConnection,authority)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','binding','policy','settings','calibration','station-token-file',
                 'vision-token-file','control-token-file','ownership-db'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--vision-url',required=True)
    parser.add_argument('--port',type=int,default=18810)
    parser.add_argument('--execution-enabled',action='store_true')
    parser.add_argument('--state-root',type=Path)
    parser.add_argument('--station-config',type=Path,required=True)
    args=parser.parse_args()
    state_root=args.state_root or args.root/'state'
    preview_link=args.root/'local-only/yoloe-recyclables/motion-preview-ledger'
    preview_state=state_root/'preview-ledger'
    preview_state.mkdir(parents=True,exist_ok=True)
    preview_link.parent.mkdir(parents=True,exist_ok=True)
    if not preview_link.exists():preview_link.symlink_to(preview_state,target_is_directory=True)
    if preview_link.resolve()!=preview_state.resolve():raise ValueError('board_preview_state_not_shared')
    if not args.ownership_db.is_file():raise ValueError('existing_authoritative_owner_required')
    os.environ.update(YOLOE_RECYCLABLE_POLICY=str(args.policy.resolve()),
        YOLOE_RECYCLABLE_SETTINGS=str(args.settings.resolve()),
        YOLOE_RECYCLABLE_CALIBRATION=str(args.calibration.resolve()),YOLOE_RECYCLABLE_URL=args.vision_url)
    import cv2
    cv2.setNumThreads(1)
    from .prompted_recyclables import read_policy
    from .monitor_prompted_operation import PromptedBackend,PromptedOperations
    from .sorting_prompted_console import create_console
    from .sorting_prompted_motion import calibrated_settings,run
    from .sorting_known import bound_file
    policy=read_policy(args.policy)
    binding=json.loads(args.binding.read_text())
    camera_path=bound_file(args.binding.parent,binding['files']['camera'])
    camera=json.loads(camera_path.read_text())
    config_path=bound_file(args.binding.parent,binding['files']['config'])
    config=json.loads(config_path.read_text());arm_mapping=config['board_transport']
    translated_arm(config['arm'],arm_mapping)
    station_config_bytes=args.station_config.read_bytes()
    station_config=json.loads(station_config_bytes)
    if station_config['assets']!=config['statistics']['assets']:
        raise ValueError('board_station_assets_changed')
    camera_hash=hashlib.sha256(camera_path.read_bytes()).hexdigest()
    scope_path=args.root/'scope-frozen.json'
    scope_bytes=scope_path.read_bytes();scope=json.loads(scope_bytes)
    def check_sources():
        if args.station_config.read_bytes()!=station_config_bytes:raise ValueError('board_station_configuration_changed')
        if scope_path.read_bytes()!=scope_bytes:raise ValueError('board_source_manifest_changed')
        for name,digest in scope.items():
            path=(args.root/name).resolve()
            if not path.is_relative_to(args.root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                raise ValueError('board_control_source_changed:'+name)
    check_sources()
    from .prompted_motion_policy import read_motion_options
    options=read_motion_options()
    model_catalog=args.root/'execution-models.json';models=None
    if model_catalog.is_file():
        from .execution_models import ExecutionModels
        models=ExecutionModels(json.loads(model_catalog.read_text()),state_root/'execution-model-binding.json')
    def model_scope():return models.context() if models is not None else nullcontext()
    def using_model(method):
        def wrapped(*a,**kw):
            with model_scope():return method(*a,**kw)
        return wrapped
    owner=BoardOwnership(args.ownership_db)
    authority=Authority(owner,check_sources,options['frame_max_age_ms'],
                        motion_scene_check_mode=scene_check_mode(options))
    visual=VisualInputs(args.vision_url,args.vision_token_file.read_text().strip(),policy['model'],
                        None,camera_hash,authority)
    install_board_adapters(authority,visual,args.station_token_file.read_text().strip(),camera,'http://127.0.0.1:8783',arm_mapping)
    readiness_view=ReadinessView()
    class Backend(PromptedBackend):
        def reset_requires_confirmation(self):
            return super().reset_requires_confirmation() or owner.status()['status']=='blocked'
        def check(self):
            return readiness_view.check(self.fresh_check)
        @using_model
        def fresh_check(self):
            check_sources()
            blockers=super().check()
            from .board_serial_inventory import identify_arm,usb_serial_inventory
            inventory=identify_arm(usb_serial_inventory(),arm_mapping['usb_identity'])
            if (inventory['status']!='arm_usb_identified_not_opened' or
                inventory['selected']['device']!=arm_mapping['arm_device']):
                blockers.append('board_arm_usb_mapping_not_current')
            try:
                settings,_=calibrated_settings();health=visual.request('/health');selected_policy=read_policy()
                if (health.get('settings')!=settings or
                    health.get('model_hashes',{}).get(selected_policy['model']['name'])!=selected_policy['model']['sha256']):
                    blockers.append('board_visual_settings_not_frozen')
            except (OSError,ValueError,KeyError):blockers.append('board_visual_service_unavailable')
            if not args.execution_enabled:blockers.append('board_execution_not_enabled')
            state=owner.status()
            if state['status']=='blocked':blockers.append('board_control_blocked:'+str(state['reason']))
            return blockers
        @using_model
        def cycle(self,material,execute):
            check_sources()
            if execute and not args.execution_enabled:raise ValueError('board_execution_not_enabled')
            authority.begin('station-console',self.current_request_id)
            failed=True;report=None
            try:
                settings,_=calibrated_settings();visual.freeze(settings)
                extra={'height_plan':self.height_plan} if self.height_plan is not None else {}
                report=run(material,execute,object_height_mm=self.object_height_mm,**extra)
                visual.verify();failed=False
                report.update(compute_location='pynq_ps',state_and_ledger_location='pynq_ps',
                    actuator_location='pynq_ps',perception_location='pc',board_owner_epoch=owner.epoch,
                    board_port_open_attempted=authority.port_attempted,board_uart_write_attempted=authority.write_attempted)
                folder=(self.journal_dir if execute else args.root/'local-only/yoloe-recyclables/motion-preview-ledger')
                path=folder/(report['task_id']+'.json')
                if path.is_file():
                    temporary=path.with_suffix('.host.tmp')
                    temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
                    temporary.replace(path)
                return report
            except BaseException as exc:
                folder=state_root/'attempts';folder.mkdir(parents=True,exist_ok=True)
                failure=dict(kind='board-control-attempt-failure-v1',request_id=self.current_request_id,
                    compute_location='pynq_ps',reason=type(exc).__name__+':'+str(exc),
                    board_port_open_attempted=authority.port_attempted,board_uart_write_attempted=authority.write_attempted,
                    original_reports_unchanged=True,automatic_retry=False)
                path=folder/(self.current_request_id+'.json')
                path.write_text(json.dumps(failure,indent=2),encoding='utf8')
                raise
            finally:authority.finish(report,failed)
        def stop(self):
            super().stop();owner.stop('Operator stopped the unified board controller.')
        def reset(self,confirmed=False):
            result=super().reset(confirmed)
            if owner.status()['status']=='blocked':
                owner.review('Console operator confirms stationary arm, empty gripper and hands clear; jaw opening is not inferred; original failures preserved.',confirmed)
            return result
        def review_status(self):
            from .sorting_manual_review import review_status
            return review_status(self.journal_dir,self.device_lease,owner)
        def review(self,*,epoch,request_id,confirmed=False):
            self._verify();check_sources()
            current=json.loads(args.binding.read_text())
            if not any(Path(r['path']).name=='sorting_manual_review.py' for r in current['implementation_sources']):
                raise ValueError('manual_review_implementation_not_bound')
            from .sorting_manual_review import acknowledge
            return acknowledge(self.journal_dir,self.device_lease,owner,epoch=epoch,request_id=request_id,confirmed=confirmed)
        @using_model
        def home(self):
            if not args.execution_enabled:raise ValueError('board_execution_not_enabled')
            authority.begin('station-console',self.current_request_id)
            report=None;failed=True
            try:
                settings,_=calibrated_settings();visual.freeze(settings)
                report=super().home();visual.verify();failed=False
                return report
            finally:authority.finish(report,failed)
        @using_model
        def jaw_calibrate(self):
            from .sorting_neutral_return import return_neutral
            check_sources()
            if not args.execution_enabled:raise ValueError('board_execution_not_enabled')
            authority.begin('station-jaw-calibration',self.current_request_id)
            report=None;failed=True
            try:
                settings,_=calibrated_settings();visual.freeze(settings)
                report=return_neutral(args.binding,execute=True,confirmed=True,jaw_step=self.jaw_calibration_request)
                visual.verify();failed=False
                report.update(board_port_open_attempted=authority.port_attempted,
                              board_uart_write_attempted=authority.write_attempted)
                return report
            finally:authority.finish(report,failed)
    backend=Backend(args.root)
    preserved=args.root/'startup-preserved-state.json'
    if preserved.is_file():
        from .prompted_height import object_height
        previous=json.loads(preserved.read_text())
        backend.object_height_mm=object_height(previous.get('object_height_mm'))
    backend.decision_thresholds=initial_decision_thresholds(options,model_scope,calibrated_settings)
    class BoardOperations(PromptedOperations):
        def command(self,data):
            if isinstance(data,dict) and data.get('action')=='heartbeat':
                return renew_operation_heartbeat(self,data,authority)
            with self.lock:
                if models is not None and isinstance(data,dict) and data.get('action') in ('preview','start','home','jaw_calibrate'):
                    active=models.public()
                    if ('execution_profile_sha256' in data and data['execution_profile_sha256']!=active['profile_sha256'] or
                        'execution_binding_revision' in data and data['execution_binding_revision']!=active['revision']):
                        raise ValueError('selected_execution_model_binding_changed')
                previous=getattr(backend,'current_request_id',None)
                previous_jaw=getattr(backend,'jaw_calibration_request',None)
                if isinstance(data,dict) and data.get('action')=='jaw_calibrate':
                    from .jaw_calibration import request as jaw_request
                    if self.status()['busy']:raise ValueError('operation_busy_or_closed')
                    backend.jaw_calibration_request=jaw_request(data.get('jaw_calibration'))
                if isinstance(data,dict) and data.get('action') in ('preview','start','home','jaw_calibrate'):
                    backend.current_request_id=data.get('request_id')
                try:result=super().command(data)
                except Exception:
                    backend.current_request_id=previous
                    backend.jaw_calibration_request=previous_jaw
                    raise
            return result
        @using_model
        def status(self):
            with readiness_view.display():result=super().status()
            dynamic={'operator_stop_requested','arm_in_use_or_stale_device_lease','existing_physical_trial_unresolved',
                     'unified_batch_needs_review'}
            result['blockers']=[v for v in result['blockers'] if v not in dynamic and not v.startswith('board_control_blocked:')]
            if any(p.exists() for p in backend.stops):result['blockers'].append('operator_stop_requested')
            if backend.device_lease.exists():result['blockers'].append('arm_in_use_or_stale_device_lease')
            if (backend.journal_dir/'pending.json').exists():result['blockers'].append('existing_physical_trial_unresolved')
            if (backend.journal_dir/'batch-pending.json').exists():result['blockers'].append('unified_batch_needs_review')
            current=owner.status()
            if current['status']=='blocked':result['blockers'].append('board_control_blocked:'+str(current['reason']))
            result.update(compute_location='pynq_ps',state_and_ledger_location='pynq_ps',
                          actuator_location='pynq_ps',perception_location='pc',board_ownership=current,
                          board_root=str(args.root),board_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            active=current_profile()
            if active is not None:
                result['execution_model']=dict(profile_id=active['id'],name=active['name'],model_sha256=active['model_sha256'],
                    profile_sha256=active['profile_sha256'],configuration_sha256=active['upstream_configuration_sha256'],
                    settings_sha256=active['settings']['settings_sha256'],policy_sha256=active['settings']['policy_sha256'],
                    revision=active['binding_revision'],route_id=active['route_id'],physical_validation_completed=False)
                result['decision_thresholds']=dict(result['decision_thresholds'],min_confidence=active['settings']['confidence'])
            return result
    operations=BoardOperations(backend,state_root/'operations.jsonl',poll_s=2,lease_s=12,standby_s=120,
                               defer_preflight=True)
    backend.operations=operations
    def refresh_display():
        while not operations.closed.is_set():
            if authority.lease is None:
                try:backend.check()
                except Exception:
                    with readiness_view.lock:readiness_view.value=['station_binding_or_configuration_unavailable']
            operations.closed.wait(10)
    threading.Thread(target=refresh_display,daemon=True,name='readiness-display').start()
    # Route the original read-only display to the same persistent PS ledger and
    # authenticated camera service, rather than an empty candidate-local folder.
    from . import sorting_monitor_server,monitor_station_vision
    from .monitor_tasks import TaskTelemetry,current_workflow
    from .sorting_station_io import StationStatistics
    class LedgerView(TaskTelemetry):
        def __init__(self):super().__init__(backend.journal_dir)
        def status(self,*a,**kw):
            if not kw.get('task_id') and (backend.journal_dir/'pending.json').is_file():
                pending=json.loads((backend.journal_dir/'pending.json').read_text())
                kw['task_id']=pending['task_id']
            result=super().status(*a,**kw)
            result['device_lease_present']=backend.device_lease.exists()
            result['state_and_ledger_location']='pynq_ps'
            return result
    sorting_monitor_server.TaskTelemetry=LedgerView
    def station_display_request(route,body=None):
        if body is not None:authority.check()
        request=urllib.request.Request('http://127.0.0.1:8783'+route,
            data=json.dumps(body).encode() if body is not None else None,
            headers={'Authorization':'Bearer '+args.station_token_file.read_text().strip(),
                     'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=3) as reply:raw=reply.read(128001)
        if len(raw)>128000:raise ValueError('oversized_station_display_reply')
        return json.loads(raw)
    monitor_station_vision.board_request=station_display_request
    sorting_monitor_server.station_vision_status=lambda **kw:monitor_station_vision.station_vision_status(args.station_config,**kw)
    def workflow():
        result=current_workflow(config_path)
        if result.get('available'):
            result.update(geometry_backend='pynq_ps_numpy',compute_location='pynq_ps',perception_location='pc',
                          actuator_location='pynq_ps',state_and_ledger_location='pynq_ps',
                          model_sha256=policy['model']['sha256'],min_confidence=calibrated_settings()[0]['confidence'])
        return result
    sorting_monitor_server.current_workflow=workflow
    def bind_baseline(payload):
        import uuid
        with operations.lock:
            if operations.status()['busy'] or backend.reset_requires_confirmation():
                raise ValueError('board_baseline_requires_idle_reviewed_station')
            check_sources();authority.begin('station-baseline',uuid.uuid4().hex)
            try:return monitor_station_vision.bind_station_baseline(payload,args.station_config)
            finally:authority.finish()
    sorting_monitor_server.bind_station_baseline=bind_baseline
    server=create_console(args.port,upstream=args.vision_url,operations=operations)
    if server.server_address[0]!='127.0.0.1':raise ValueError('board_console_must_be_loopback')
    original=server.RequestHandlerClass;token=args.control_token_file.read_text().strip()
    class Handler(original):
        def reply(self,status,raw,headers=None):
            if isinstance(raw,bytes) and headers and headers.get('Content-Type','').startswith('text/html'):
                raw=raw.replace(b'</body>',
                    '<p class="caption">控制、规划、任务状态与机械接口由 PYNQ 处理；电脑提供视觉结果。</p><script src="/board-station.js"></script></body>'.encode())
                raw=raw.replace('GPU 识别整段'.encode(),'视觉识别整段'.encode()).replace('GPU + CPU 收尾'.encode(),'当前视觉运行时'.encode())
            if isinstance(raw,bytes) and headers and headers.get('Content-Type','').startswith('application/javascript'):
                raw=raw.replace(":'识别设备待核对',gpu?'ok':'warn');".encode(),
                    ":d.model_loaded&&d.execution?.device==='cpu'?'CPU · '+d.execution.precision.toUpperCase():'识别设备待核对',(gpu||(d.model_loaded&&d.execution?.device==='cpu'))?'ok':'warn');".encode())
            return super().reply(status,raw,headers)
        def authorized(self):return hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+token)
        def do_GET(self):
            if not self.authorized():return self.reply(403,{'error':'board_control_authentication_required'})
            from urllib.parse import urlsplit,parse_qs
            route=urlsplit(self.path)
            if route.path=='/execution-model':
                if models is None:return self.reply(404,dict(error='execution_model_follow_not_installed'))
                return self.reply(200,models.public())
            if route.path=='/board-station.js':
                return self.reply(200,Path(__file__).with_name('monitor_assets').joinpath('board_station.js').read_bytes(),
                                  {'Content-Type':'application/javascript; charset=utf-8'})
            if route.path=='/board-preview-report':
                operation_id=parse_qs(route.query).get('operation_id',[None])[0]
                status=operations.status();result=status.get('result') or {}
                if (status['busy'] or operation_id!=status['operation_id'] or result.get('preview') is not True):
                    return self.reply(409,dict(error='matching_finished_board_preview_required'))
                task=result.get('task_id')
                if not isinstance(task,str) or not task.isalnum():
                    return self.reply(409,dict(error='board_preview_rejected',reason=result.get('reason')))
                path=preview_state/(task+'.json')
                if not path.is_file():return self.reply(409,dict(error='board_preview_report_unavailable'))
                return self.reply(200,preview_display(json.loads(path.read_text())))
            if self.path=='/board-control':
                return self.reply(200,dict(compute_location='pynq_ps',mode='board_supervised_single_item',
                    execution_enabled=args.execution_enabled,ownership=owner.status(),state_and_ledger_location='pynq_ps',
                    physical_motion_on_startup=False,perception_location='pc',board_root=str(args.root),
                    board_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
            return super().do_GET()
        def do_POST(self):
            if not self.authorized():return self.reply(403,{'error':'board_control_authentication_required'})
            if self.path=='/execution-model/bind':
                if models is None:return self.reply(404,dict(error='execution_model_follow_not_installed'))
                try:
                    size=int(self.headers.get('Content-Length','0'))
                    if not 0<size<=4096 or self.headers.get('Transfer-Encoding'):raise ValueError('invalid_execution_binding_body')
                    data=json.loads(self.rfile.read(size))
                    if set(data)!={'profile_id','profile_sha256','expected_revision','request_id','route_id'}:
                        raise ValueError('registered_execution_binding_only')
                    target=copy.deepcopy(models.profiles[data['profile_id']]);target['route_id']=data['route_id']
                    if target['profile_sha256']!=data['profile_sha256']:raise ValueError('execution_profile_identity_changed')
                    check_sources()
                    with model_context(target):health=visual.request('/health')
                    if health.get('ready') is not True or health.get('settings')!=target['settings'] or health.get('execution_profile',{}).get('profile_sha256')!=target['profile_sha256']:
                        raise ValueError('selected_execution_model_not_ready')
                    def idle():
                        if operations.status()['busy'] or owner.status()['status']=='active' or backend.device_lease.exists():
                            raise ValueError('execution_model_switch_requires_idle')
                    with operations.lock:
                        value=models.activate(data['profile_id'],data['expected_revision'],data['request_id'],idle,route_id=data['route_id'])
                        operations.invalidate_height_registration('model_changed')
                    threading.Thread(target=backend.check,daemon=True,name='model-binding-readiness').start()
                    return self.reply(200,value)
                except (OSError,ValueError,KeyError,TypeError) as exc:
                    return self.reply(409,dict(error=str(exc)[:240],automatic_retry=False))
            if self.path=='/voice-transport':
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=512:return self.reply(400,{'error':'invalid_voice_transport_body'})
                payload=json.loads(self.rfile.read(size))
                if set(payload)!={'connected','device'} or type(payload['connected']) is not bool:
                    return self.reply(400,{'error':'invalid_voice_transport_state'})
                operations.voice_connection(payload['connected'])
                return self.reply(200,dict(voice_service=operations.status()['voice_service']))
            if self.path=='/prompted/plan':
                import uuid
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=524288:return self.reply(400,{'error':'invalid_board_preview_body'})
                payload=json.loads(self.rfile.read(size))
                if set(payload)-{'material','object_height_mm','height_plan','execution_profile_sha256','execution_binding_revision'}:
                    return self.reply(400,{'error':'board_preview_material_and_height_only'})
                status=operations.status();owner_id='embedded-'+uuid.uuid4().hex
                operations.command(dict(action='preview',owner=owner_id,token=status['token'],
                    request_id=uuid.uuid4().hex,material=payload.get('material','AUTO'),
                    object_height_mm=payload.get('object_height_mm'),
                    height_plan=payload.get('height_plan'),
                    **{k:payload[k] for k in ('execution_profile_sha256','execution_binding_revision') if k in payload}))
                deadline=time.monotonic()+90
                while time.monotonic()<deadline:
                    with operations.lock:
                        worker=operations.worker
                        if worker is None or not worker.is_alive():
                            result=operations.status().get('result') or {}
                            task=result.get('task_id')
                            if task and (backend.journal_dir/(task+'.json')).is_file():
                                return self.reply(200,json.loads((backend.journal_dir/(task+'.json')).read_text()))
                            # Preview journals are isolated from physical history.
                            preview=args.root/'local-only/yoloe-recyclables/motion-preview-ledger'
                            if task and (preview/(task+'.json')).is_file():
                                return self.reply(200,json.loads((preview/(task+'.json')).read_text()))
                            return self.reply(409,operations.status())
                    time.sleep(.05)
                return self.reply(503,{'error':'board_preview_deadline_no_retry'})
            return super().do_POST()
    server.RequestHandlerClass=Handler
    print(json.dumps(dict(status='board_station_ready',port=args.port,execution_enabled=args.execution_enabled,
                         physical_motion_on_startup=False)),flush=True)
    try:server.serve_forever()
    finally:
        operations.close();server.server_close();owner.close()


if __name__=='__main__':main()
