"""Use the existing console state machine with the prompted single-item runner."""
import json
import time
import uuid
from pathlib import Path
from .monitor_operation import StationBackend,Operations
from .sorting_prompted_motion import calibrated_settings,run,FRAME_MAX_AGE_MS,FRAME_TIMEOUT_S
from .prompted_motion_policy import read_motion_options
from .prompted_height import object_height
from .prompted_trial_scope import execution_scope
from .prompted_recyclables import read_policy
from .motion_sequence import scene_check_mode
from .sorting_item_heights import height_plan


class PromptedBackend(StationBackend):
    def __init__(self,root):
        super().__init__(root)
        # Existing UI shows single-item scope; shared leases/pending/stop paths
        # continue to be owned by the original station backend.
        from .prompted_batch import batch_settings
        batch=batch_settings(read_motion_options())
        self.unified=batch is not None
        self.batch_verification_mode='end_of_batch' if batch else 'per_item'
        self.skip_per_item_visual_verification=batch is not None
        self.review_tolerance_factor=1
        self.bin_edge_overlap_limit=0
        self.empty_bin_max_changed_fraction=.05
        self.object_height_mm=None
        self.height_plan=None
        from .sorting_unified_run import load_unified
        bundle,_=load_unified(self.binding)
        self.mechanical_heights_mm={m:p['object_height_mm'] for m,p in bundle[1]['profiles'].items()}

    def check(self):
        blockers=super().check()
        # Single-item UI still shares the station's batch ledger. A previous
        # real batch may require review even though this backend is not batched.
        if (self.journal_dir/'batch-pending.json').exists() and 'unified_batch_needs_review' not in blockers:
            blockers.append('unified_batch_needs_review')
        try:
            _,receipt=calibrated_settings()
            if not execution_scope(receipt,read_policy(),read_motion_options())['materials']:
                blockers.append('prompted_validation_insufficient')
        except (OSError,ValueError,KeyError):blockers.append('prompted_calibration_not_current')
        return blockers

    def cycle(self,material,execute):
        self._verify()
        extra={'height_plan':self.height_plan} if self.height_plan is not None else {}
        return run(material,execute,object_height_mm=self.object_height_mm,**extra)


def prompted_operations(root):
    backend=PromptedBackend(root)
    return PromptedOperations(backend,Path(root)/'local-only/yoloe-recyclables/motion-events.jsonl',
                      poll_s=2,lease_s=12,standby_s=120)


class PromptedOperations(Operations):
    HEIGHT_REGISTRATION_SECONDS = 300

    def invalidate_height_registration(self, reason):
        with self.lock:
            registration=getattr(self,'_height_registration',None)
            self._height_registration=None
            if registration is not None:
                self._height_registration_version=getattr(self,'_height_registration_version',0)+1
                self._record('height_registration_invalidated', reason=reason,
                             registration_id=registration['id'])

    def height_registration(self):
        with self.lock:
            registration = getattr(self, '_height_registration', None)
            if registration is not None and time.monotonic() >= registration['deadline']:
                self.invalidate_height_registration('expired')
                return None
            return registration

    def registration_command(self, data):
        with self.lock:
            if data.get('token') != self.token:raise ValueError('page_expired_reload')
            owner = data.get('owner')
            if not isinstance(owner, str) or not 8 <= len(owner) <= 80:raise ValueError('invalid_page_owner')
            current = self.height_registration()
            expected = data.get('height_registration_id')
            if expected != (current['id'] if current else None):
                raise ValueError('height_registration_changed_reload')
            if data['action'] == 'clear_heights':
                self.invalidate_height_registration('page_cancelled')
                return self.status()
            if self.closed.is_set() or self.state in self.ACTIVE or (self.worker and self.worker.is_alive()):
                raise ValueError('operation_busy_or_closed')
            request_id = data.get('request_id')
            if not isinstance(request_id, str) or not 8 <= len(request_id) <= 80:raise ValueError('invalid_request_id')
            if request_id in self.seen:raise ValueError('duplicate_request_no_retry')
            if len(self.seen) >= 1000:raise ValueError('session_request_limit_restart_required')
            if data.get('ready') is not True:raise ValueError('height_registration_scene_confirmation_required')
            plan = height_plan(data.get('height_plan'))
            if plan is None or data.get('object_height_mm') is not None:raise ValueError('instance_height_registration_required')
            state = self.status()
            if state.get('busy') or state.get('blockers'):raise ValueError('height_registration_station_not_ready')
            binding = state.get('execution_model') or {}
            if (not binding.get('profile_sha256') or type(binding.get('revision')) is not int or
                binding.get('model_sha256') != plan['model_sha256'] or
                binding.get('configuration_sha256') != plan['configuration_sha256']):
                raise ValueError('instance_height_model_or_configuration_changed')
            registration = dict(id=uuid.uuid4().hex, plan=plan, owner=owner,
                profile_sha256=binding['profile_sha256'], revision=binding['revision'],
                deadline=time.monotonic()+self.HEIGHT_REGISTRATION_SECONDS)
            self._record('height_registration_saved', registration_id=registration['id'],
                         height_plan=plan, execution_binding_revision=binding['revision'], physical=False)
            self._height_registration = registration
            self._height_registration_version=getattr(self,'_height_registration_version',0)+1
            self.seen.add(request_id)
            return self.status()

    def command(self,data):
        if isinstance(data,dict) and data.get('action') in ('register_heights','clear_heights'):
            return self.registration_command(data)
        if isinstance(data,dict) and data.get('action') in ('arm','voice_arm'):
            raise ValueError('prompted_single_manual_attempt_only')
        if isinstance(data,dict) and data.get('action') in ('preview','start'):
            height=object_height(data.get('object_height_mm'))
            plan=height_plan(data.get('height_plan'))
            if height is not None and plan is not None:raise ValueError('instance_and_shared_height_conflict')
            with self.lock:
                if data.get('token')!=self.token:raise ValueError('page_expired_reload')
                if self.state in self.ACTIVE or (self.worker is not None and self.worker.is_alive()):
                    raise ValueError('operation_busy_or_closed')
                if self.backend is None:raise ValueError('station_entry_unavailable')
                registration_id=data.get('height_registration_id')
                if str(data.get('owner','')).startswith('voice-') and not registration_id:
                    raise ValueError('voice_requires_page_height_registration')
                if registration_id is not None:
                    registration=self.height_registration()
                    if not registration or registration['id']!=registration_id:
                        raise ValueError('height_registration_missing_expired_or_consumed')
                    if height is not None or plan is not None:raise ValueError('registered_and_supplied_height_conflict')
                    binding=self.status().get('execution_model') or {}
                    if (binding.get('profile_sha256')!=registration['profile_sha256'] or
                        binding.get('revision')!=registration['revision']):
                        self.invalidate_height_registration('model_changed')
                        raise ValueError('selected_execution_model_binding_changed')
                    plan=height_plan(registration['plan'])
                previous=getattr(self.backend,'object_height_mm',None)
                previous_plan=getattr(self.backend,'height_plan',None)
                previous_operation_id=getattr(self,'operation_id',None)
                self.backend.object_height_mm=height
                self.backend.height_plan=plan
                try:
                    result=super().command(data)
                    if data['action']=='start':
                        self.invalidate_height_registration('start_consumed')
                        if isinstance(result,dict):
                            result['voice_height_registration']=None
                            result['voice_height_registration_version']=getattr(self,'_height_registration_version',0)
                    return result
                except Exception:
                    if (getattr(self,'operation_id',None)==data.get('request_id') and
                        data.get('request_id') is not None and previous_operation_id!=data['request_id']):
                        if data['action']=='start':self.invalidate_height_registration('start_accepted_reply_failed')
                        raise
                    self.backend.object_height_mm=previous
                    self.backend.height_plan=previous_plan
                    raise
        with self.lock:
            result=super().command(data)
            if isinstance(data,dict) and data.get('action') in ('stop','cancel','home','reset','jaw_calibrate','review'):
                self.invalidate_height_registration(data['action'])
                if isinstance(result,dict):
                    result['voice_height_registration']=None
                    result['voice_height_registration_version']=getattr(self,'_height_registration_version',0)
            return result

    def status(self):
        state=super().status()
        state['object_height_mm']=getattr(self.backend,'object_height_mm',None)
        state['mechanical_heights_mm']=getattr(self.backend,'mechanical_heights_mm',{})
        state['object_height_range_mm']=[1,120]
        state['height_applies_to_next_grasp']=True
        state['per_instance_heights_supported']=True
        state['per_instance_widths_supported']=True
        state['voice_height_registration_supported']=True
        with self.lock:
            registration=self.height_registration()
            if not hasattr(self,'_height_registration_session'):self._height_registration_session=uuid.uuid4().hex
            state['voice_height_registration_session']=self._height_registration_session
            state['voice_height_registration_version']=getattr(self,'_height_registration_version',0)
            state['voice_height_registration']=None if registration is None else dict(
                id=registration['id'],profile_sha256=registration['profile_sha256'],
                revision=registration['revision'],remaining_s=max(0,int(registration['deadline']-time.monotonic())),
                items=[{k:item[k] for k in ('item_id','label','object_height_mm')} for item in registration['plan']['items']])
        state['selection_policy']='clearance_then_route_v1'
        state['height_changes_grasp_z']=False
        options=read_motion_options()
        state['motion_scene_check_mode']=scene_check_mode(options)
        state['per_command_visual_checks']=state['motion_scene_check_mode']=='per_command'
        state['destination_empty_check_enabled']=options['destination_empty_check_enabled']
        state['full_scene_instance_stability_check_enabled']=options.get('full_scene_instance_stability_check_enabled',True)
        state['minimum_silhouette_separation_check_enabled']=options.get('minimum_silhouette_separation_check_enabled',True)
        state['projected_region_bounds_check_enabled']=options.get('projected_region_bounds_check_enabled',True)
        state['prompted_stable_frames']=options.get('stable_frames',3)
        if not options['destination_empty_check_enabled']:
            state['empty_bin_max_changed_fraction']=None
        try:
            settings,record=calibrated_settings()
            scope=execution_scope(record,read_policy(),options)
            state.update(prompted_trial_materials=scope['materials'],
                prompted_trial_labels=scope['official_labels'],
                prompted_scope_basis=scope['basis'],
                prompted_validated_scope=scope['validated_scope'],
                prompted_physical_validation_completed=False,
                prompted_motion_timing_overrides=options['motion_timing_overrides'],
                prompted_single_manual_attempt_only=True,
                prompted_global_confidence=settings['confidence'],
                prompted_max_frame_age_ms=options['frame_max_age_ms'],prompted_frame_timeout_s=options['frame_timeout_s'],
                prompted_thresholds=settings['material_thresholds'])
        except (OSError,ValueError,KeyError):state['prompted_trial_materials']=[]
        return state
