"""Compose measured startup work around a sealed existing board controller.

No operation admission, Ready, ownership, stop/reset, UART or FPGA programming
API is added. Live scene and final-write checks stay with the existing host.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

from . import startup_timeline as timing
from .startup_snapshot import SealedSnapshot, VerifiedBundle
from .startup_frame_evidence import FrameMarkers, digest, construct_with_initial_markers, track_camera_captures, CaptureMarkerScope

_guard = None
_installed = False
_position_proof = None
_thread = threading.local()
_backend = 'sealed_same_frame_cached_rays_v1'
_policy_installed = False
_policy_proof = None


def replace_aliases(original, replacement):
    for module in tuple(sys.modules.values()):
        if module is not None and getattr(module, '__name__', '').startswith('src.'):
            for name, value in tuple(vars(module).items()):
                if value is original:
                    setattr(module, name, replacement)


def measured(original, name):
    def call(*a, **kw):
        with timing.span(name):
            return original(*a, **kw)
    return call


def preserve_backend_height(backend_type, height, validator):
    """Restore the actual pre-update transient input after backend defaults."""
    checked = validator(height)
    original = backend_type.__init__
    def initialize(self, *a, **kw):
        original(self, *a, **kw)
        self.object_height_mm = checked
    backend_type.__init__ = initialize


def registered_confidence(profile, value):
    """An exact registered model threshold, never an arbitrary lower limit."""
    if not isinstance(profile, dict) or type(value) not in (int, float) or not .01 <= value <= .95:
        return False
    model_sha=profile.get('model_sha256')
    return (isinstance(model_sha,str) and len(model_sha)==64 and all(c in '0123456789abcdef' for c in model_sha) and
            profile.get('settings', {}).get('confidence') == value and
            profile.get('policy', {}).get('model', {}).get('sha256') == profile.get('model_sha256') and
            isinstance(profile.get('profile_sha256'), str) and len(profile['profile_sha256']) == 64 and
            profile.get('scope', {}).get('kind') == 'operator-supervised-trial-scope-v1' and
            bool(profile.get('scope', {}).get('operator_statement')))


def install_registered_policy_validation():
    """Adapt the old fixed-policy validator to the already selected catalog."""
    global _policy_installed
    if _policy_installed:raise RuntimeError('registered_policy_validation_already_installed')
    import inspect
    from . import sorting_unified_plan as plan, motion_path_runtime as motion
    from .execution_model_context import current_profile
    source_sha=hashlib.sha256(Path(plan.__file__).read_bytes()).hexdigest()
    if source_sha in (motion.AUDITED_CONFIDENCE_PLAN_SHA256, motion.AUDITED_CONFIDENCE_V2_PLAN_SHA256, motion.AUDITED_ITEM_HEIGHT_PLAN_SHA256, motion.AUDITED_CONTOUR_PLAN_SHA256):
        _policy_installed=True
        return
    if source_sha != motion.AUDITED_PLAN_SHA256:
        raise RuntimeError('registered_policy_source_not_audited')
    original=plan.validate_policy
    source=inspect.getsource(original)
    needle="        need(finite(value) and low<=value<=high,'invalid_unified_decision_threshold:'+key)\n"
    if source.count(needle)!=1:raise RuntimeError('registered_policy_guard_layout_changed')
    source=source.replace(needle,
        "        if key=='min_confidence' and _registered_active():\n"
        "            need(_registered_confidence(value),'selected_execution_confidence_changed')\n"
        "            low=min(low,value)\n"+needle)
    namespace=dict(original.__globals__,_registered_active=lambda:current_profile() is not None,
                   _registered_confidence=lambda value:registered_confidence(current_profile(),value))
    exec(compile(source,__file__+'::registered_policy_validation','exec'),namespace)
    replace_aliases(original,namespace['validate_policy'])
    _policy_installed=True


def bind_source_check(root, original):
    if _guard is None or Path(root).absolute() != _guard.root:
        raise RuntimeError('startup_source_guard_not_prepared')
    # Original host already performed its complete byte verification before
    # binding this guard. All static ancestors/entries are now sealed.
    _guard.check()
    return measured(_guard.check, 'source_integrity')


def install_runtime(root, binding, expected_sources, *, pc_warp=None):
    global _guard, _installed, _position_proof, _policy_proof
    if _installed:
        raise RuntimeError('startup_preparation_already_installed')
    root, binding = Path(root).absolute(), Path(binding).absolute()
    _guard = SealedSnapshot(root)
    for name, expected in expected_sources.items():
        if hashlib.sha256((root/'pynq_km1_vision/src'/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('startup_composition_source_changed:'+name)
    from . import board_station_host as host, sorting_prompted_motion as sm
    from . import sorting_relocated_run as relocated, sorting_unified_run as unified, sorting_known as known
    from . import empty_reference_warp as old_warp, motion_path_runtime as motion
    from . import monitor_operation, monitor_prompted_operation, station_geometry
    from .prompted_height import object_height
    preserved = _guard.read_json(root/'startup-preserved-state.json')
    preserve_backend_height(monitor_prompted_operation.PromptedBackend,
                            preserved.get('object_height_mm'), object_height)
    from .background_warp import RayTable, warp_reference
    from .execution_model_context import current_profile

    binding_loader, unified_loader = relocated.load_binding, unified.load_unified
    original_bound = known.bound_file
    def audited_bound(base_path, record):
        path = original_bound(base_path, record)
        # Every input used to build a cached proof must be inside the sealed
        # manifest. Mutable/external evidence cannot become a fixed proof.
        _guard.checked_path(path, record['sha256'])
        return path
    replace_aliases(original_bound, audited_bound)
    checked_unified = unified_loader(binding)
    bundle = checked_unified[0]
    b, c, base, camera, station, registration, scene, files = bundle
    if not _policy_installed:
        raise RuntimeError('registered_policy_composition_required_before_compiler')
    from .execution_models import digest as profile_digest
    from .execution_model_context import model_context
    from . import sorting_unified_plan as plan
    policy_cases=[]
    for profile in _guard.read_json(root/'execution-models.json')['profiles']:
        body={k:v for k,v in profile.items() if k!='profile_sha256'}
        if profile_digest(body)!=profile['profile_sha256']:
            raise RuntimeError('registered_profile_catalog_changed')
        with model_context(profile):
            policy=copy.deepcopy(checked_unified[1])
            policy['decision_thresholds']=dict(sm.read_motion_options()['decision_thresholds'],min_confidence=profile['settings']['confidence'])
            policy['threshold_operator_statement']='Selected immutable registered execution configuration.'
            plan.validate_policy(policy,c)
            invalid=copy.deepcopy(policy);invalid['decision_thresholds']['min_confidence']=.001
            try:plan.validate_policy(invalid,c)
            except ValueError:pass
            else:raise RuntimeError('unregistered_confidence_accepted')
            policy_cases.append(dict(id=profile['id'],confidence=profile['settings']['confidence']))
    _policy_proof=dict(passed=True,cases=policy_cases,unregistered_low_rejected=True,physical_motion=False)
    from .startup_calibration import CalibrationValues
    camera_loader, station_loader = station_geometry.camera_parameters, station_geometry.station_parameters
    calibration = CalibrationValues(camera, station, camera_loader, station_loader)
    replace_aliases(camera_loader, calibration.camera_parameters)
    replace_aliases(station_loader, calibration.station_parameters)
    position_policy = _guard.read_json(files['position_policy'])
    import numpy as np
    values = (c, 'paper', np.asarray(position_policy['pick_center_card_polygon_mm'], float).mean(0).tolist(),
              90, position_policy['model'])
    extension = json.loads((root/'motion-extension.json').read_text())
    client = motion.LocalMotionClient(**{k: extension[k] for k in ('token_file', 'assets', 'port')})
    client.readiness()
    _position_proof = motion.install_position_runtime(client, values)
    cached = VerifiedBundle(_guard, binding, lambda p: bundle, lambda p: checked_unified)
    replace_aliases(binding_loader, measured(lambda p: cached.load(p), 'binding_load'))
    replace_aliases(unified_loader, measured(lambda p: cached.load(p, unified=True), 'unified_load'))

    def bound_file(base_path, record):
        if not isinstance(record, dict) or not isinstance(record.get('path'), str) or not isinstance(record.get('sha256'), str):
            return original_bound(base_path, record)
        path = Path(base_path)/record['path']
        import os
        absolute = Path(os.path.abspath(path))
        if absolute in _guard.entries:
            return _guard.checked_path(absolute, record['sha256'])
        return original_bound(base_path, record)

    original_readiness = relocated.known_readiness
    initial = original_readiness(c, base, require_empty=False, require_vision_binding=False)
    replace_aliases(audited_bound, bound_file)
    fixed = [v for v in initial if v not in ('operator_stop_requested', 'arm_in_use_or_stale_device_lease')]
    configuration_key = digest(c)
    def readiness(config, config_base, require_empty=True, require_vision_binding=True):
        if require_empty or require_vision_binding or Path(config_base) != base or digest(config) != configuration_key:
            return original_readiness(config, config_base, require_empty, require_vision_binding)
        _guard.check()
        if any(v.startswith('known_configuration:') for v in fixed):
            return list(fixed)
        leading = int(bool(config.get('synthetic_fixture', False)))+int(config['known'].get('experiment_authorized') is not True)
        dynamic = []
        if any((base/p).resolve().exists() for p in config['known']['stop_files']):
            dynamic.append('operator_stop_requested')
        if (base/config['known']['device_lease']).resolve().exists():
            dynamic.append('arm_in_use_or_stale_device_lease')
        return list(fixed[:leading])+dynamic+list(fixed[leading:])
    replace_aliases(original_readiness, measured(readiness, 'live_readiness'))

    rays = RayTable()
    intrinsic, distortion = station_geometry.camera_parameters(camera)
    pc_client = None
    if pc_warp is None:
        rays.get(c['image_size'], intrinsic, distortion)
    else:
        from .background_warp_client import BackgroundClient
        pc_client = BackgroundClient(**pc_warp)
        pc_client.readiness()
    original_warp_reference = old_warp.warp_reference
    def cached_warp(image, size, regions, profiles, current, reference, roi_only=True):
        if not roi_only:
            return original_warp_reference(image, size, regions, profiles, current, reference, roi_only=False)
        with timing.span('background_warp'):
            if pc_client is None:
                result = warp_reference(image, size, regions, profiles, current, reference, rays, projector=old_warp._project)
            else:
                context = getattr(_thread, 'warp_context', None)
                if context is None:
                    raise RuntimeError('pc_background_frame_context_missing')
                result = pc_client.warp(dict(reference_bgr_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                    size=list(size), regions=regions,
                    profiles={k:dict(bin_id=p['bin_id'],bin_roi=p['bin_roi']) for k,p in profiles.items()},
                    current=[v.tolist() for v in current], reference=[v.tolist() for v in reference], context=context))
        count = size[0]*size[1]
        heights = {0.} | {float(v['height_mm']) for v in regions.values()}
        projected = count
        for name, region in regions.items():
            if name != 'source' and region['height_mm'] != 0:
                label = next(k for k,p in profiles.items() if p['bin_id'] == name)
                x0,y0,x1,y1 = profiles[label]['bin_roi']
                projected += (x1-x0)*(y1-y0)
        return result, dict(backend='pc_background_math_v1' if pc_client else 'cached_rays_native_background_v1', validated_world_points=count*len(heights),
                            projected_points=projected, floor_count=len(heights), roi_only=True,
                            projector_backend='pc_opencv_cached_rays' if pc_client else 'cpu_native_forward_projection',
                            native_library_sha256=None if pc_client else old_warp._native_library_sha256,
                            rpc_timing=None if pc_client is None else pc_client.last_timing)
    old_warp.warp_reference = cached_warp
    old_warp.BACKEND = 'pc_background_math_v1' if pc_client else 'cached_rays_native_background_v1'
    from .sorting_relocated_geometry import RelocatedGeometry
    original_geometry_warp = RelocatedGeometry.warp_empty_reference
    def geometry_warp(geometry, image):
        previous = getattr(_thread, 'warp_context', None)
        _thread.warp_context = dict(frame=copy.deepcopy(getattr(_thread, 'frame_identity', None)),
            geometry_signature=geometry.signature, current_pose_sha256=digest(geometry.pose),
            source_bgr_sha256=hashlib.sha256(geometry.reference.tobytes()).hexdigest(),
            model_version=geometry.runtime.get('model_version'), runtime_id=geometry.runtime.get('runtime_id'))
        try:return original_geometry_warp(geometry, image)
        finally:_thread.warp_context = previous
    RelocatedGeometry.warp_empty_reference = geometry_warp

    original_adapters = host.install_board_adapters
    def adapters(authority, visual, station_token, camera_parameters, station_url, arm_mapping):
        original_adapters(authority, visual, station_token, camera_parameters, station_url, arm_mapping)
        from . import sorting_station_io
        # Detector-driven CameraYoloe.next() calls _raw directly, bypassing
        # the older host's raw-only timestamp hook. Both must track captures.
        track_camera_captures(sorting_station_io.StationCamera, authority)
        capture_markers=CaptureMarkerScope(camera_parameters,lambda:visual.expected,current_profile,
                                          visual.camera_hash,station_geometry.detect_markers)
        station_geometry.detect_markers=capture_markers.detect
        original_next=sorting_station_io.StationCamera.next
        def next_frame(self,*a,**kw):
            row=original_next(self,*a,**kw)
            capture_markers.observed(row)
            return row
        sorting_station_io.StationCamera.next=next_frame
        sorting_station_io.StationCamera.observe_prepared=lambda self,row:capture_markers.observed(row)
        original_acquire=sm.PromptedMotionController.acquire
        def acquire(self,*a,**kw):
            with capture_markers.acquisition():return original_acquire(self,*a,**kw)
        sm.PromptedMotionController.acquire=acquire
        original_raw = sm.raw_prompted_window
        def raw_window(*a, **kw):
            with timing.span('initial_capture_pose'):
                result = original_raw(*a, **kw)
            raw, pose, stats = result
            _thread.markers = FrameMarkers(raw[-1], camera_parameters, visual.expected, current_profile(), visual.camera_hash)
            _thread.frame_identity = dict(stream_id=raw[-1]['observation']['source_stream_id'],
                frame_id=raw[-1]['observation']['source_frame_id'], image_sha256=_thread.markers.image_hash,
                visual_generation=visual.expected['generation'],profile_sha256=digest(current_profile()))
            return result
        sm.raw_prompted_window = raw_window
        original_geometry = sm.geometry_for_policy
        def geometry(*a, **kw):
            markers = getattr(_thread, 'markers', None)
            if markers is None:
                raise RuntimeError('initial_frame_markers_not_prepared')
            _thread.markers = None
            detector = lambda image, cam: markers.detector(image, cam, visual.expected, current_profile())
            with timing.span('geometry'):
                return construct_with_initial_markers(original_geometry, a, kw, detector)
        sm.geometry_for_policy = geometry
        original_visual_request = visual.request
        visual.request = measured(original_visual_request, 'visual_rpc')
        from . import arm_transport
        original_open = arm_transport.ArmConnection.open
        arm_transport.ArmConnection.open = measured(original_open, 'serial_open')
        original_send = arm_transport.ArmConnection.send
        def send(self, data, duration, guard=None):
            with timing.span('uart_first_guard_and_send'):
                stamp = original_send(self, data, duration, guard)
            timing.uart_written(stamp)
            return stamp
        arm_transport.ArmConnection.send = send
    host.install_board_adapters = adapters

    original_worker = monitor_operation.Operations._run
    def worker(self, *a, **kw):
        started = getattr(self, '_startup_submitted', {}).get(self.operation_id)
        with timing.transaction(self.operation_id, started):
            return original_worker(self, *a, **kw)
    monitor_operation.Operations._run = worker
    original_run = sm.run
    def run(*a, **kw):
        with timing.transaction():
            _thread.markers = None
            try:
                return original_run(*a, **kw)
            finally:
                _thread.markers = None
                _thread.frame_identity = None
    replace_aliases(original_run, run)
    original_cycle = sm.PromptedMotionController.cycle
    for name in ('observe', 'select', 'acquire', 'unchanged', 'prepare_motion'):
        setattr(sm.PromptedMotionController, name, measured(getattr(sm.PromptedMotionController, name), 'controller_'+name))
    replace_aliases(unified.configure_target_region, measured(unified.configure_target_region, 'target_region_setup'))
    def cycle(self, *a, **kw):
        timing.boundary()
        return original_cycle(self, *a, **kw)
    sm.PromptedMotionController.cycle = cycle
    original_status = monitor_prompted_operation.PromptedOperations.status
    def status(self):
        result = original_status(self)
        result['empty_reference_projector'] = 'pc_opencv_cached_rays' if pc_client else 'cpu_native_forward_projection'
        result['startup_baseline_native_library_sha256'] = old_warp._native_library_sha256
        if pc_client:
            result['startup_native_library_sha256'] = None
        result['startup_acceleration'] = dict(backend=_backend, sealed_files=len(_guard.entries),
            snapshot_faulted=_guard.faulted, ray_table_ready=True, position_policy_pl_selftest=copy.deepcopy(_position_proof),
            background_backend='pc_background_math_v1' if pc_client else 'cached_rays_native_background_v1',
            background_faulted=False if pc_client is None else pc_client.faulted,
            registered_policy_selftest=copy.deepcopy(_policy_proof),
            timeline=timing.status())
        return result
    monitor_prompted_operation.PromptedOperations.status = status
    _installed = True
    return dict(backend=_backend, sealed_files=len(_guard.entries), position_policy_pl_selftest=_position_proof)
