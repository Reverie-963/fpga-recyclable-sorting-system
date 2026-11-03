"""Three-profile, all-planar-angle route candidates with explicit model provenance.

No device access. Model reachability is not measured collision qualification.
"""
import math
import cv2
import numpy as np
from .arm import ik, packet
from .arm_route_preview import tip_rz
from .sorting_loop import LABELS, finite, fingerprint, mapped, need
from openmv_gripper.km1_arm_bridge import validate_packet
from .sorting_unified_motion import timing_policy, retime_route
from .sorting_review_policy import review_factor
from .contour_grasp import jaw_command, validate_jaw_calibration


def polygon(value):
    p = np.asarray(value, np.float32)
    need(p.ndim == 2 and p.shape[1] == 2 and len(p) >= 3 and np.isfinite(p).all()
         and cv2.isContourConvex(p) and cv2.contourArea(p) > 1, 'invalid_unified_polygon')
    return p


def validate_policy(policy, c):
    timing_policy(policy)
    need(policy.get('selection_policy','stable_track_order') in
         ('stable_track_order','clearance_then_route_v1'),'invalid_unified_selection_policy')
    factor=review_factor(policy)
    edge=policy.get('relax_bin_edge_overlap',False)
    need(type(edge) is bool and (not edge or (factor>1 and bool(policy.get('bin_edge_operator_statement')))),
         'bin_edge_review_requires_explicit_margin_record')
    need(policy.get('batch_verification_mode','per_item') in ('per_item','end_of_batch'),
         'invalid_batch_verification_mode')
    skip=policy.get('skip_per_item_visual_verification',False)
    need(type(skip) is bool and (not skip or policy.get('batch_verification_mode')=='end_of_batch'),
         'skip_item_verification_requires_batch_final')
    scale=policy.get('wait_time_scale',1)
    need(type(scale) in (int,float) and scale in (.5,1),'invalid_unified_wait_scale')
    need(scale==1 or bool(policy.get('wait_operator_statement')),'wait_scale_requires_operator_statement')
    thresholds=policy.get('decision_thresholds',{})
    limits={'min_confidence':(.4,1),'min_scene_iou':(.85,1),
            'min_erased_fraction':(.75,1),'min_persistent_fraction':(.6,1),
            'max_scene_changed_fraction':(0,.05),'max_other_bin_changed_fraction':(0,.05)}
    need(isinstance(thresholds,dict) and not set(thresholds)-set(limits),'invalid_unified_decision_thresholds')
    need(not thresholds or bool(policy.get('threshold_operator_statement')),'threshold_change_requires_operator_statement')
    from .execution_model_context import current_profile
    active=current_profile()
    for key,value in thresholds.items():
        if key=='min_confidence' and active is not None and value==active['settings']['confidence']:
            need(finite(value) and .01<=value<=.95,
                 'selected_execution_confidence_changed')
            continue
        low,high=limits[key]
        if key.startswith('min_'):low/=factor
        else:high=min(1,high*factor)
        need(finite(value) and low<=value<=high,'invalid_unified_decision_threshold:'+key)
    fraction=policy.get('min_geometry_component_fraction',.99)
    need(finite(fraction) and .9/factor<=fraction<=1,'invalid_unified_geometry_component_fraction')
    empty_limit=policy.get('empty_bin_max_changed_fraction',c['max_scene_changed_fraction'])
    need(finite(empty_limit) and 0<=empty_limit<=min(1,.2*factor),'invalid_empty_bin_change_fraction')
    need(policy.get('kind') == 'supervised-unified-sorting-v1' and
         policy.get('operator_authorized') is True and policy.get('operator_statement'),
         'unified_sorting_requires_operator_policy')
    need(policy.get('workspace_physically_qualified') is False and
         policy.get('unattended_accepted') is False, 'unified_model_cannot_claim_measured_workspace')
    need(set(policy['models']) == set(LABELS), 'unified_requires_three_explicit_profiles')
    need(type(policy['max_objects']) is int and 1 <= policy['max_objects'] <= 6,
         'invalid_unified_object_limit')
    need(type(policy.get('verification_timeout_s',5)) in (int,float) and
         2<=policy.get('verification_timeout_s',5)<=15,'invalid_unified_verification_window')
    age=policy.get('verification_max_age_ms',int(policy.get('verification_timeout_s',5)*1000))
    need(type(age) is int and c['max_age_ms']<=age<=policy.get('verification_timeout_s',5)*1000,
         'invalid_unified_post_verification_frame_age')
    if 'detection_window_px' in policy:
        need(policy['detection_window_px']==[0,0,*c['image_size']], 'unified_detection_window_must_be_full_frame')
        need(type(policy.get('max_scene_instances',32)) is int and 6<=policy.get('max_scene_instances',32)<=64,
             'invalid_unified_scene_limit')
    source, centers = polygon(policy['source_card_polygon_mm']), polygon(policy['pick_center_card_polygon_mm'])
    need(all(cv2.pointPolygonTest(source, tuple(map(float, p)), False) >= 0 for p in centers),
         'unified_center_outside_source')
    for label, model in policy['models'].items():
        p = c['profiles'][label]
        need(model['object_profile_id'] == p['object_profile_id'] and
             model['object_height_mm'] == p['object_height_mm'] and
             model['grasp_z_mm'] == p['grasp_z_mm'], 'unified_class_height_or_profile_mismatch')
        need(model['kind'] == 'unified-position-model-v1' and model.get('provenance'),
             'explicit_per_class_model_required')
        for key in ('base_pwm_per_degree', 'wrist_pwm_per_degree'):
            need(finite(model[key]) and 3 <= model[key] <= 12, 'invalid_unified_angle_calibration')
        need(type(model['base_center_pwm']) is int and 500 <= model['base_center_pwm'] <= 2500 and
             finite(model['radial_bias_mm']) and -20 <= model['radial_bias_mm'] <= 20,
             'invalid_unified_model_offset')
        need(model['xy_mode'] in ('constant_grip', 'two_height_maps'), 'invalid_unified_xy_model')
        need(1 <= len(model['source_chains']) <= 6, 'invalid_unified_pitch_candidates')
        for chain in model['source_chains']:
            need(len(chain) == 4 and all(len(pose) == 2 and all(finite(v) for v in pose)
                 for pose in chain), 'invalid_unified_source_chain')
            z = [pose[0] for pose in chain]
            need(z[0] > z[1] > z[2] > z[3] == p['grasp_z_mm'] >= 0 and
                 all(-90 <= pose[1] <= 0 for pose in chain), 'unified_chain_heights_or_pitch')
    need(finite(policy['jaw_open_span_mm']) and 20 <= policy['jaw_open_span_mm'] <= 150 and
         finite(policy['jaw_depth_mm']) and 5 <= policy['jaw_depth_mm'] <= 80 and
         finite(policy['obstacle_margin_mm']) and 0 < policy['obstacle_margin_mm'] <= 30,
         'invalid_unified_tool_clearance')
    need(type(policy.get('bin_padding_px',5)) is int and 0<=policy.get('bin_padding_px',5)<=30,
         'invalid_unified_bin_padding')
    for label in LABELS:
        validate_jaw_calibration(c['profiles'][label], policy)
        slots = policy['bin_slots'][label]
        need(slots and len({s['slot_id'] for s in slots}) == len(slots), 'invalid_bin_slots')
        for s in slots:
            r = s['roi_fraction']
            need(len(r) == 4 and all(finite(v) for v in r) and
                 0 <= r[0] < r[2] <= 1 and 0 <= r[1] < r[3] <= 1, 'invalid_bin_slot_fraction')
            need(s.get('origin') in ('legacy_qualified_profile', 'explicit_slot_record', 'candidate'),
                 'bin_slot_provenance_required')
            rect=slot_rect(c['profiles'][label],s)
            need(rect[2]-rect[0]>=8 and rect[3]-rect[1]>=8,'unified_bin_slot_too_small')
            if s['origin']=='legacy_qualified_profile':
                need(r==[0,0,1,1] and s.get('transfer_pwm',c['profiles'][label]['transfer_pwm'])==c['profiles'][label]['transfer_pwm'] and
                     s.get('retreat_pwm',c['profiles'][label]['retreat_pwm'])==c['profiles'][label]['retreat_pwm'],
                     'legacy_bin_slot_cannot_claim_new_release_path')
        need(not any(max(a['roi_fraction'][0],b['roi_fraction'][0])<min(a['roi_fraction'][2],b['roi_fraction'][2]) and
                     max(a['roi_fraction'][1],b['roi_fraction'][1])<min(a['roi_fraction'][3],b['roi_fraction'][3])
                     for i,a in enumerate(slots) for b in slots[i+1:]),'overlapping_bin_slots')
    return policy


def apply_decision_thresholds(c,policy):
    """Override only opt-in unified visual decisions, never motion guards."""
    import copy
    result=copy.deepcopy(c)
    result.update(policy.get('decision_thresholds',{}))
    return result


def slot_rect(profile, slot):
    x0, y0, x1, y1 = profile['bin_roi']
    a, b, d, e = slot['roi_fraction']
    return [math.ceil(x0+(x1-x0)*a), math.ceil(y0+(y1-y0)*b),
            math.floor(x0+(x1-x0)*d), math.floor(y0+(y1-y0)*e)]


def compile_route(c, label, card_xy, axis_deg, policy, slot, *, grasp_width_mm=None):
    validate_policy(policy, c)
    need(label in LABELS and finite(axis_deg), 'unsupported_unified_target')
    p, model = c['profiles'][label], policy['models'][label]
    closure = jaw_command(p, policy, grasp_width_mm)
    need(slot in policy['bin_slots'][label],'unified_slot_not_in_policy')
    xy = np.asarray(card_xy, float)
    need(xy.shape == (2,) and np.isfinite(xy).all() and
         cv2.pointPolygonTest(polygon(policy['pick_center_card_polygon_mm']), tuple(map(float, xy)), False) >= 0,
         'unified_pick_outside_workspace')
    grip = mapped(p['card_to_grip_mm'], [xy])[0]
    hover = mapped(p['card_to_hover_mm'], [xy])[0]

    def control(z):
        f = 0 if model['xy_mode'] == 'constant_grip' else np.clip(
            (z-p['grasp_z_mm'])/(p['hover_z_mm']-p['grasp_z_mm']), 0, 1)
        point = grip*(1-f)+hover*f
        radius = float(np.linalg.norm(point))
        need(point[1] > 0 and 100 <= radius+model['radial_bias_mm'] <= 230,
             'unified_target_outside_nominal_reach')
        return point*((radius+model['radial_bias_mm'])/radius)

    yaw = math.degrees(math.atan2(grip[0], grip[1]))
    # A parallel-jaw grasp has 180-degree symmetry. Test all equivalent rolls,
    # retaining the one nearest the qualified neutral wrist within PWM bounds.
    rolls = [90-(axis_deg % 180)-yaw+180*k for k in (-1, 0, 1)]
    wrist_options = [(abs(r), round(p['wrist_pwm']+model['wrist_pwm_per_degree']*r), r)
                     for r in rolls]
    wrist_options = [v for v in wrist_options if 500 <= v[1] <= 2500]
    need(wrist_options, 'unified_grasp_angle_unreachable')
    _, wrist, roll = min(wrist_options)
    chains, failures = [], []
    for chain_index, chain in enumerate(model['source_chains']):
        steps = []
        try:
            def append(name, pairs, ms=5000, **extra):
                data = packet(pairs, ms)
                need(validate_packet(data, [[500, 2500]]*6) == ms, 'invalid_unified_packet')
                steps.append(dict(name=name, command=data.decode(), duration_ms=ms, **extra))

            def pose(name, point):
                z, alpha = point
                target = control(z)
                joints = ik(*target, z, alpha)
                turn = math.degrees(math.atan2(target[0], target[1]))
                joints[0] = round(model['base_center_pwm']-model['base_pwm_per_degree']*turn)
                append(name, list(enumerate(joints)), xyz_nominal=target.tolist()+[z], alpha_deg=alpha)

            append('home', list(enumerate(c['known']['prepare_pwm']))+[(4,p['wrist_pwm']), (5,p['open_pwm'])])
            pose('source_high', chain[0])
            append('align_wrist', [(4,wrist)])
            for name, point in zip(('source_hover','source_mid','source_grip'), chain[1:]):pose(name, point)
            append('close', [(5,closure['close_pwm'])], 2000)
            for name, point in zip(('source_lift_mid','source_lift_hover','source_lift_high'), reversed(chain[:-1])):pose(name, point)
            append('front_high', list(enumerate(ik(0,160,190,-25))))
            append('restore_wrist_high', [(4,p['wrist_pwm'])])
            for i, joints in enumerate(slot.get('transfer_pwm', p['transfer_pwm'])):
                append('transfer_'+str(i), list(enumerate(joints)))
            append('release', [(5,p['open_pwm'])], 2000, settle_s=1)
            for i, joints in enumerate(slot.get('retreat_pwm', p['retreat_pwm'])):
                append('retreat_'+str(i), list(enumerate(joints)))
            state, audit = list(c['known']['initial_pwm']), []
            for step in steps:
                target, text = list(state), step['command']
                for offset in range(1,len(text)-1,15):
                    word=text[offset:offset+15];target[int(word[1:4])]=int(word[5:9])
                heights=[tip_rz([a+(b-a)*i/100 for a,b in zip(state,target)])[1] for i in range(101)]
                need(min(heights) >= p['grasp_z_mm']-.75, 'unified_nominal_height_floor')
                audit.append(dict(name=step['name'],start_pwm=state,target_pwm=target,
                                  nominal_z_min_mm=min(heights)))
                state=target
            need(state == c['known']['initial_pwm'], 'unified_route_must_end_neutral_open')
            cost=sum(sum(abs(a-b) for a,b in zip(s['start_pwm'],s['target_pwm'])) for s in audit)
            chains.append((chain_index, cost, steps, audit))
        except ValueError as exc:
            failures.append(str(exc))
    need(chains, 'no_complete_unified_route:'+','.join(sorted(set(failures))))
    # Prefer the first declared (previously checked) pitch family, then cost.
    chain_index, _, steps, audit = min(chains, key=lambda v:(v[0],v[1]))
    return retime_route(dict(label=label,bin_id=p['bin_id'],object_profile_id=p['object_profile_id'],
        object_height_mm=p['object_height_mm'],pick_card_xy_mm=xy.tolist(),axis_deg=axis_deg % 180,
        base_pwm=audit[5]['target_pwm'][0],wrist_pwm=wrist,roll_deg=roll,jaw_closure=closure,
        source_chain_index=chain_index,slot_id=slot['slot_id'],steps=steps,nominal_interpolation=audit,
        model_sha256=fingerprint(model),workspace_physically_qualified=False,
        collision_free_verified=False,physical_motion=False),policy)
