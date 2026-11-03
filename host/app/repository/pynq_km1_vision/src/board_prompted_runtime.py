"""Board PS three-frame observation and planning for the current 24 prompts.

Every camera operation is a GET, all motion objects are None, and all journals
belong to the candidate. This module never enables or sends physical commands.
"""
import base64
import copy
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import uuid

import cv2

from .board_detector_transport import RelayTransport
from .prompted_recyclables import read_policy, settings_for
from .sorting_loop import Journal, need
from .sorting_io import Statistics
from .sorting_instances import InstanceLedger
from .sorting_station_io import StationStatistics, StationCamera
from .sorting_position_controller import PositionWindowedCamera
from .sorting_prompted_preview import project_row, stable_official_labels, PromptedPreviewController
from .sorting_relocated_run import (motion_shift_limit, motion_marker_limit,
    initial_marker_shift_limit)
from .sorting_relocated_geometry import RelocatedGeometry, pixels_to_card
from .sorting_unified_run import load_unified, apply_geometry, per_bin_empty_images
from .sorting_review_policy import review_factor
from .station_geometry import localize_records
from .station_relocalization import legacy_domains
from .prompted_motion_policy import read_motion_options
from .prompted_observation import PromptedWindowedCamera,single_frame_pose,frame_count
from .prompted_region_geometry import geometry_for_policy,apply_prompted_geometry
from .prompted_height import effective_heights


def station_get(token, route):
    need(route in ('/health',), 'board_preview_station_write_forbidden')
    request = urllib.request.Request('http://127.0.0.1:8783'+route,
                                     headers={'Authorization':'Bearer '+token})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=1.5) as response:
        raw = response.read(128001)
    need(len(raw) <= 128000, 'oversized_board_station_reply')
    return json.loads(raw)


class BoardStationStatistics(StationStatistics):
    def __init__(self, config, token):
        self.token = token
        super().__init__(config)

    def request(self, route, body=None):
        need(body is None, 'board_preview_station_write_forbidden')
        return station_get(self.token, route)


class BoardStationCamera(StationCamera):
    def __init__(self, config, detector, statistics, token, check):
        super().__init__(config, detector, statistics)
        self.opener.addheaders = [('Authorization', 'Bearer '+token)]
        self.captured, self.owner_check = [], check

    def raw(self, timeout):
        self.owner_check()
        row = super().raw(timeout)
        self.captured.append(row)
        return row


class PromptedRelayDetector:
    def __init__(self, url, token, timeout, expected, check=lambda:None, camera_sha256=None, model_name='yoloe-11m-seg.pt'):
        self.model_name=model_name
        self.relay = RelayTransport(url, token, timeout)
        self.expected = copy.deepcopy(expected)
        need(model_name in self.expected['model_hashes'],'board_expected_model_name_missing')
        self.check = check
        self.camera_sha256 = camera_sha256
        self.validate_health(self.relay.request('health'))

    def validate_health(self, health):
        need(health.get('ready') is True, 'prompted_detector_not_ready')
        for key in ('settings', 'generation', 'model_hashes', 'execution'):
            need(health.get(key) == self.expected.get(key), 'prompted_detector_identity_changed:'+key)
        if 'station_filter_config_sha256' in self.expected:
            need(health.get('station_filter_config_sha256')==self.expected['station_filter_config_sha256'],
                 'board_station_filter_identity_changed')

    def predict(self, image, prompt_mode='text'):
        self.check()
        need(prompt_mode == 'text', 'prompted_board_text_mode_required')
        ok, data = cv2.imencode('.png', image)
        need(ok, 'cannot_encode_board_prompted_frame')
        raw = data.tobytes()
        result = self.relay.request('detect', dict(image_base64=base64.b64encode(raw).decode(), prompt_mode='text'))
        settings = self.expected['settings']
        need(result.get('settings_sha256') == settings['settings_sha256'] and
             result.get('policy_sha256') == settings['policy_sha256'] and
             result.get('generation') == self.expected['generation'] and
             result.get('model_sha256') == self.expected['model_hashes'][self.model_name],
             'prompted_board_result_identity_changed')
        need(result.get('image_size') == list(image.shape[1::-1]) and
             result.get('source_bgr_sha256') == hashlib.sha256(image.tobytes()).hexdigest(),
             'prompted_board_mask_pixels_changed')
        if 'station_filter_config_sha256' in self.expected:
            need(result.get('station_filter',{}).get('config_sha256')==self.expected['station_filter_config_sha256'] and
                 result['station_filter'].get('raw_instances_preserved') is True,'board_station_filter_result_changed')
        if self.camera_sha256 is not None:
            markers=result.get('fixed_marker_observation',{})
            need(markers.get('location')=='pc_vision' and markers.get('camera_sha256')==self.camera_sha256 and
                 markers.get('source_bgr_sha256')==result['source_bgr_sha256'] and
                 isinstance(markers.get('detections'),list),'board_fixed_marker_image_binding_changed')
        self.check()
        return dict(result, image_sha256=hashlib.sha256(raw).hexdigest())


def capture_config(config, health):
    need(health['assets'] == config['statistics']['assets'] and health['image_size'] == config['image_size'] and
         health['threshold'] == config['statistics']['threshold'], 'board_capture_identity_changed')
    result = copy.deepcopy(config)
    regions = health['regions']
    result['source_roi'] = regions['source']
    result['layout']['static_rois'] = [regions['static_a'], regions['static_b']]
    for profile in result['profiles'].values():
        profile['bin_roi'] = regions[profile['bin_id']]
    result['statistics'].update(regions=regions, url='http://127.0.0.1:8783')
    result['camera_url'] = 'http://127.0.0.1:8783/frame.jpg'
    return result


def locate(raw, binding, config, camera, station, registration, scene, camera_sha256):
    for row in raw:
        markers=row['observation'].get('fixed_marker_observation',{})
        need(markers.get('location')=='pc_vision' and markers.get('camera_sha256')==camera_sha256 and
             markers.get('source_bgr_sha256')==row['bgr_sha256'],'board_pose_marker_snapshot_changed')
    records = [dict(image_sha256=r['source_bytes_sha256'], detections=r['observation']['fixed_marker_observation']['detections'],
        source_frame_id=int(r['observation']['source_frame_id']),
        source_stream_id=r['observation']['source_stream_id']) for r in raw]
    if len(records)==1:
        return single_frame_pose(records[0],camera,station,motion_marker_limit(binding),review_factor(binding))
    profiles = {label:dict(kind='station-legacy-profile-v1', profile_id=p['object_profile_id'],
        object_height_mm=p['object_height_mm'], checked_card_polygons_mm=p['pick_hulls_card_mm'],
        card_to_grip=p['card_to_grip_mm'], card_to_hover=p['card_to_hover_mm'],
        grip_model_z_mm=p['grasp_z_mm'], hover_model_z_mm=p['hover_z_mm']) for label,p in config['profiles'].items()}
    domains = legacy_domains(registration, profiles, station)
    points = [p for domain in domains.values() for p in domain['world_xyz_mm']]
    points += [p for data in scene['scene_regions'].values() for p in data['world_xyz_mm']]
    return localize_records(records, camera, station, 'fixed_marker_task_space', points,
        min_fixed_markers=motion_marker_limit(binding), max_marker_shift_px=initial_marker_shift_limit(binding),
        review_factor=review_factor(binding))


class BoardPromptedRuntime:
    def __init__(self, binding, policy, relay_url, relay_token, station_token, artifacts, observation_timing=None):
        self.binding, self.policy = Path(binding), read_policy(policy)
        self.relay_url, self.relay_token, self.station_token = relay_url, relay_token, station_token
        self.artifacts = Path(artifacts)
        self.timing = observation_timing
        self.motion_options=read_motion_options()
        if self.timing is not None:
            need(set(self.timing)=={'max_age_ms','frame_timeout_s','source_sha256','source_module'} and
                 self.timing['source_module']=='sorting_prompted_motion.py' and
                 type(self.timing['max_age_ms']) is int and 2000<=self.timing['max_age_ms']<=6000 and
                 type(self.timing['frame_timeout_s']) in (int,float) and 2<=self.timing['frame_timeout_s']<=6 and
                 len(self.timing['source_sha256'])==64,'invalid_board_observation_timing_snapshot')

    def preview(self, expected, material='AUTO', check=lambda:None, task_id=None):
        need(material in ('AUTO','PAPER','PLASTIC','METAL'), 'invalid_board_material')
        settings = expected['settings']
        materials=('paper','plastic','metal') if material=='AUTO' else (material.lower(),)
        heights=effective_heights(settings,expected.get('object_height_mm'),materials)
        from .sorting_item_heights import ItemHeights, height_plan
        plan=height_plan(expected.get('height_plan'))
        need(plan is None or expected.get('object_height_mm') is None,'instance_and_shared_height_conflict')
        need(expected.get('motion_options')==self.motion_options,'board_motion_options_snapshot_changed')
        need(settings_for(self.policy, settings) == settings, 'invalid_board_prompt_snapshot')
        if self.policy.get('station_filter_sha256'):
            need(expected.get('station_filter_config_sha256')==self.policy['station_filter_sha256'],
                 'board_policy_station_filter_changed')
        bundle, motion_policy = load_unified(self.binding)
        b,c,base,cam,station,reg,scene,files = bundle
        item_heights=(None if plan is None else ItemHeights(plan,
            model=self.policy['model']['sha256'],configuration=expected['configuration_sha256'],image_size=c['image_size']))
        original=c
        motion_policy=copy.deepcopy(motion_policy)
        motion_policy['selection_policy']='clearance_then_route_v1'
        motion_policy['decision_thresholds']=dict(self.motion_options['decision_thresholds'],min_confidence=settings['confidence'])
        motion_policy['threshold_operator_statement']='Use the frozen current-console motion options in this read-only board preview.'
        motion_policy['min_geometry_component_fraction']=self.motion_options['min_geometry_component_fraction']
        motion_policy['batch_verification_mode']='per_item';motion_policy['skip_per_item_visual_verification']=False
        if motion_policy.get('motion_timing'):motion_policy['motion_timing'].update(self.motion_options['motion_timing_overrides'])
        if self.timing is not None:
            need(expected.get('observation_timing')==self.timing,'board_observation_timing_identity_changed')
            c=copy.deepcopy(c)
            c.update(max_age_ms=self.timing['max_age_ms'],frame_timeout_s=self.timing['frame_timeout_s'])
        c=copy.deepcopy(c);c['stable_frames']=self.motion_options.get('stable_frames',3)
        task = task_id or uuid.uuid4().hex
        folder = self.artifacts/task
        folder.mkdir(parents=True, exist_ok=False)
        report = dict(status='blocked', task_id=task, compute_location='pynq_ps',
            settings_sha256=settings['settings_sha256'], policy_sha256=settings['policy_sha256'],
            physical_motion_attempted=False, automatic_motion_allowed=False, execution_enabled=False,
            execution_blockers=['new_taxonomy_and_height_require_physical_validation'],
            frozen_snapshot=True, material_selection=material, candidates=[], rejected=[],decisions=[],frames=[],
            observation_timing=self.timing,motion_options=self.motion_options,decision_frame_count=c['stable_frames'])
        camera = None
        started = time.monotonic()
        try:
            check()
            health = station_get(self.station_token, '/health')
            capture = capture_config(c, health)
            capture['prompt_mode'] = 'text'
            camera_sha256=hashlib.sha256(files['camera'].read_bytes()).hexdigest()
            need(expected['model_hashes'].get(self.policy['model']['name'])==self.policy['model']['sha256'],
                 'board_frozen_policy_model_not_active')
            detector = PromptedRelayDetector(self.relay_url, self.relay_token, c['frame_timeout_s'], expected, check,camera_sha256,
                                            model_name=self.policy['model']['name'])
            statistics = BoardStationStatistics(capture, self.station_token)
            camera = BoardStationCamera(capture, detector, statistics, self.station_token, check)
            observer = PromptedWindowedCamera(camera)
            raw = [observer.next(c['frame_timeout_s']) for _ in range(c['stable_frames'])]
            if c['stable_frames']==1:camera.captured.extend(raw)
            observed = time.monotonic()*1000
            ages = [observed-r['observation']['capture_started_ms'] for r in raw]
            need(all(0 <= a <= c['max_age_ms'] for a in ages), 'board_three_frame_window_expired')
            report.update(observation_window_fresh=True, observation_frame_ages_ms=ages,
                observation_window_ms=(time.monotonic()-started)*1000,
                frames=[dict(stream_id=r['observation']['source_stream_id'], frame_id=r['observation']['source_frame_id'],
                             image_sha256=r['source_bytes_sha256']) for r in raw])
            pose = locate(raw,b,c,cam,station,reg,scene,camera_sha256)
            report.update(marker_detection_location='pc_vision',pose_and_planning_location='pynq_ps')
            projection_policy=copy.deepcopy(motion_policy)
            for label,height in heights.items():projection_policy['models'][label]['object_height_mm']=height
            if plan is not None:
                for item in plan['items']:
                    model=projection_policy['models'][item['label']]
                    model['object_height_mm']=max(model['object_height_mm'],item['object_height_mm'])
            def frozen_markers(image,camera_config):
                need(camera_config==cam,'board_geometry_camera_changed')
                digest=hashlib.sha256(image.tobytes()).hexdigest()
                matches=[r for r in raw if r['bgr_sha256']==digest]
                need(bool(matches),'board_geometry_image_not_in_observation')
                observation=matches[0]['observation']['fixed_marker_observation']
                need(observation['camera_sha256']==camera_sha256 and
                     observation['source_bgr_sha256']==digest,'board_geometry_markers_changed')
                return copy.deepcopy(observation['detections'])
            geometry = apply_prompted_geometry(geometry_for_policy(original,cam,station,reg,scene,pose,raw[-1]['image'],
                motion_shift_px=motion_shift_limit(b),min_fixed_markers=motion_marker_limit(b),
                review_tolerance_factor=review_factor(b),marker_detector=frozen_markers),projection_policy)
            for label, info in self.policy['materials'].items():
                need(info['bin_id'] == c['profiles'][label]['bin_id'], 'board_material_bin_binding_changed')
            geometry.map = lambda label,points:pixels_to_card(points,heights[label],pose,cam,station,reg)
            for label,height in heights.items():geometry.runtime['profiles'][label]['object_height_mm']=height
            empty = geometry.warp_empty_reference(cv2.imread(str(files['empty_image'])))
            per_bin = (per_bin_empty_images(geometry,files,raw[-1]['observation']['source_stream_id'])
                       if self.motion_options['destination_empty_check_enabled'] else {})
            rows = [project_row(self.policy,settings,r['observation'],r['image'],geometry.runtime['profiles']) for r in raw]
            report.update(decisions=rows[-1]['decisions'], heights_mm=heights if plan is None else None,supplied_object_height_mm=expected.get('object_height_mm'),
                          projected_regions=geometry.runtime['statistics']['regions'])
            row = stable_official_labels(rows)
            check()
            with Journal(folder/'ledger') as journal:
                controller = PromptedPreviewController(original,None,Statistics(),None,journal,geometry,empty,base,
                    policy=motion_policy,tracker=InstanceLedger(self.motion_options['decision_thresholds']['min_scene_iou']),mode='preview',empty_by_label=per_bin,
                    allowed_labels=('paper','plastic','metal') if material=='AUTO' else (material.lower(),),item_heights=item_heights)
                controller.c.update(self.motion_options['decision_thresholds'])
                controller.c['min_confidence']=settings['confidence']
                controller.policy.update(empty_bin_max_changed_fraction=.05,relax_bin_edge_overlap=False)
                for label,height in heights.items():
                    controller.c['profiles'][label]['object_height_mm']=height
                    controller.policy['models'][label]['object_height_mm']=height
                for other in rows[1:]:controller.unchanged(row,other)
                check()
                _,report['rejected']=controller.select(row)
                for choice,compiled_plan in controller.ranked_plans:
                    report['candidates'].append(dict(choice=choice,
                        steps=[dict(name=n,command=p.decode(),duration_ms=d) for n,p,d in compiled_plan]))
                report['candidate_order']=controller.candidate_order
                report['instance_height_plan']=expected.get('height_plan')
                report.update(status='preview_ready' if report['candidates'] else 'no_pickable_instance',slot_checks=controller.slot_checks)
            detector.validate_health(detector.relay.request('health'))
            final = station_get(self.station_token,'/health')
            need(all(final[k] == health[k] for k in ('stream_id','assets','config_sha256','baseline_frame_id')),
                 'board_station_changed_during_preview')
        except (ValueError,OSError,KeyError,TypeError) as exc:
            report.update(status='blocked',reason=str(exc),error_type=type(exc).__name__,candidates=[])
        finally:
            if camera is not None:
                for index,frame in enumerate(camera.captured):
                    cv2.imwrite(str(folder/f'frame-{index}.png'),frame['image'])
                    (folder/f'observation-{index}.json').write_text(json.dumps(frame['observation'],ensure_ascii=False))
            report['elapsed_s'] = time.monotonic()-started
            (folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        return report
