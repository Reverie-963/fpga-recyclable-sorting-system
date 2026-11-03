"""Shared execution options for every prompted-console preview and motion path."""
import json
from pathlib import Path
from .bin_occupancy import verification_options
from .motion_sequence import scene_check_mode

POLICY_PATH=Path(__file__).resolve().parents[1]/'config/prompted_motion_policy.json'


def read_motion_options():
    policy=json.loads(POLICY_PATH.read_text(encoding='utf-8'))
    verification_options(policy)
    if type(policy.get('fuse_precommit_baseline',False)) is not bool:raise ValueError('invalid_fused_precommit_baseline_option')
    if type(policy.get('rgb_first_verification',False)) is not bool:raise ValueError('invalid_rgb_first_verification_option')
    scene_check_mode(policy)
    if policy.get('kind')!='prompted-motion-options-v1' or type(policy.get('destination_empty_check_enabled')) is not bool:
        raise ValueError('invalid_prompted_motion_options')
    if not policy['destination_empty_check_enabled'] and not policy.get('operator_statement'):
        raise ValueError('destination_empty_override_requires_operator_statement')
    if type(policy.get('full_scene_instance_stability_check_enabled',True)) is not bool:
        raise ValueError('invalid_scene_instance_stability_option')
    if type(policy.get('minimum_silhouette_separation_check_enabled',True)) is not bool:
        raise ValueError('invalid_silhouette_separation_option')
    if type(policy.get('projected_region_bounds_check_enabled',True)) is not bool:
        raise ValueError('invalid_projection_bounds_option')
    if type(policy.get('stable_frames',3)) is not int or not 1<=policy.get('stable_frames',3)<=10:
        raise ValueError('invalid_prompted_observation_frame_count')
    if type(policy.get('frame_max_age_ms')) is not int or not 1<=policy['frame_max_age_ms']<=12000:
        raise ValueError('invalid_prompted_frame_age')
    if type(policy.get('frame_timeout_s')) not in (int,float) or not 0<policy['frame_timeout_s']<=12:
        raise ValueError('invalid_prompted_frame_timeout')
    return policy
