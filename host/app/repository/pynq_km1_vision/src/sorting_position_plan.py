"""Compile a complete position-dependent route, without opening any device.

This model preview is separate from measured workspace qualification. Inverse
kinematics and a successful packet audit do not establish physical alignment.
"""
import math
import numpy as np
import cv2
from .arm import ik, packet
from .arm_route_preview import tip_rz
from .sorting_loop import mapped, need, fingerprint
from openmv_gripper.km1_arm_bridge import validate_packet


def compile_position_route(c, label, card_xy, axis_deg, model):
    need(label == 'paper', 'position_model_requires_the_65mm_carton')
    p = c['profiles'][label]
    need(p['object_height_mm'] == 65, 'position_model_height_mismatch')
    xy = np.asarray(card_xy, float)
    need(xy.shape == (2,) and np.isfinite(xy).all() and math.isfinite(axis_deg), 'invalid_position_target')
    need(model['kind'] == 'carton-position-model-preview-v1', 'invalid_position_model')
    center, slope, bias, wrist_rate = (model[k] for k in (
        'base_center_pwm', 'base_pwm_per_degree', 'radial_bias_mm', 'wrist_pwm_per_degree'))
    need(type(center) is int and 500 <= center <= 2500 and
         all(type(v) in (int,float) and math.isfinite(v) for v in (slope,bias,wrist_rate)) and
         3 <= slope <= 12 and -20 <= bias <= 20 and 3 <= wrist_rate <= 12, 'invalid_position_model_parameters')
    grip = mapped(p['card_to_grip_mm'], [xy])[0]
    radius = float(np.linalg.norm(grip))
    need(140 <= radius+bias <= 210 and grip[1] > 0 and abs(grip[0]) <= 100, 'outside_position_preview_workspace')
    yaw = math.degrees(math.atan2(grip[0],grip[1]))
    base = round(center-slope*yaw)
    # Closing direction perpendicular to the specimen axis, compensating yaw.
    roll = (90-axis_deg-yaw+90) % 180-90
    wrist = round(p['wrist_pwm']+wrist_rate*roll)
    need(500 <= base <= 2500 and 500 <= wrist <= 2500, 'position_servo_outside_protocol')
    control = (grip*((radius+bias)/radius)).tolist()
    steps = []
    def append(name, pairs, duration=5000, **extra):
        data = packet(pairs,duration)
        need(validate_packet(data,[[500,2500]]*6)==duration,'invalid_position_packet')
        steps.append(dict(name=name,command=data.decode(),duration_ms=duration,**extra))
    def pose(name, z, alpha):
        joints=ik(*control,z,alpha);joints[0]=base
        append(name,list(enumerate(joints)),xyz_nominal=control+[z],alpha_deg=alpha)
    append('prepare_front_high',list(enumerate(c['known']['prepare_pwm']))+[(4,p['wrist_pwm']),(5,p['open_pwm'])])
    pose('source_high',130,-45)
    append('align_wrist',[(4,wrist)])
    pose('source_hover',90,-60)
    pose('source_mid',70,-65)
    pose('source_grip',50,-65)
    append('close',[(5,p['close_pwm'])],2000)
    pose('source_lift_mid',70,-65)
    pose('source_lift_hover',90,-60)
    pose('source_lift_high',130,-45)
    append('front_high',list(enumerate(ik(0,160,190,-25))))
    append('restore_wrist_high',[(4,p['wrist_pwm'])])
    for i,joints in enumerate(p['transfer_pwm']):append('transfer_'+str(i),list(enumerate(joints)))
    append('release',[(5,p['open_pwm'])],2000,settle_s=1)
    for i,joints in enumerate(p['retreat_pwm']):append('retreat_'+str(i),list(enumerate(joints)))
    state=list(c['known']['initial_pwm']);audit=[]
    for step in steps:
        target=list(state);text=step['command']
        for offset in range(1,len(text)-1,15):
            word=text[offset:offset+15];target[int(word[1:4])]=int(word[5:9])
        heights=[tip_rz([a+(b-a)*i/100 for a,b in zip(state,target)])[1] for i in range(101)]
        need(min(heights)>=49,'position_route_nominal_height_floor')
        audit.append(dict(name=step['name'],start_pwm=state,target_pwm=target,nominal_model_z_min_mm=min(heights)))
        state=target
    need(state==c['known']['initial_pwm'],'position_route_must_return_neutral_open')
    inside=all(cv2.pointPolygonTest(np.asarray(h,np.float32),tuple(map(float,xy)),False)>=0 for h in p['pick_hulls_card_mm'])
    return dict(kind='complete-position-dependent-route-preview-v1',label=label,pick_card_xy_mm=xy.tolist(),
        axis_deg=axis_deg,control_xy_nominal=control,base_yaw_deg=yaw,base_pwm=base,wrist_pwm=wrist,
        model_sha256=fingerprint(model),steps=steps,nominal_interpolation=audit,
        inside_legacy_checked_point_domain=inside,physical_motion=False,execution_permitted=False,
        calibration_status='candidate_model_requires_workspace_measurements',
        required_before_execution=['position_alignment_and_payload_clearance','bound_physical_workspace_and_route_qualification'],
        robot_mapping_is_measured=False,collision_free_verified=False)
