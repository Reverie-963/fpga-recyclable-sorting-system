"""Single-frame decisions with truthful pose and RGB evidence, shared with PS."""
import copy
import json
import time
import urllib.request
from .prompted_motion_policy import read_motion_options
from .sorting_loop import need
from .sorting_position_controller import PositionWindowedCamera
from .sorting_station_io import StationCamera,StationStatistics
from .station_geometry import estimate_pose,detect_markers,fingerprint


def frame_count():
    return read_motion_options().get('stable_frames',3)


class PromptedWindowedCamera(PositionWindowedCamera):
    def next(self,timeout):
        return self.source.next(timeout) if frame_count()==1 else super().next(timeout)


def single_frame_pose(record,camera,station,min_fixed_markers=3,review_factor=1):
    need(isinstance(record.get('image_sha256'),str) and len(record['image_sha256'])==64,'invalid_single_frame_hash')
    pose=estimate_pose(record['detections'],camera,station,min_fixed_markers,review_factor)
    return dict(kind='station-pose-preview-v1',**pose,camera_sha256=fingerprint(camera),
        station_sha256=fingerprint(station),camera_id=camera['camera_id'],lens_profile_id=camera['lens_profile_id'],
        image_size=camera['image_size'],world_frame_id=station['world_frame_id'],evidence=[record],
        per_frame_pose=[pose],stability_quality=dict(mode='operator_requested_single_frame',
            observed_frames=1,temporal_consistency_checked=False,minimum_fixed_markers=min_fixed_markers),
        recorded_replay_only=True,automatic_motion_allowed=False,physical_accuracy_verified=False)


def raw_prompted_window(c,camera,station,registration,scene=None,**kwargs):
    if frame_count()!=1:
        from .sorting_relocated_run import raw_window
        return raw_window(c,camera,station,registration,scene,**kwargs)
    with urllib.request.urlopen(c['statistics']['url'].rstrip('/')+'/health',timeout=3) as response:
        health=json.load(response)
    need(health['assets']==c['statistics']['assets'] and health['image_size']==c['image_size'] and
         health['threshold']==c['statistics']['threshold'],'station_capture_identity_changed')
    capture=copy.deepcopy(c);regions=health['regions']
    capture['source_roi']=regions['source'];capture['layout']['static_rois']=[regions['static_a'],regions['static_b']]
    for p in capture['profiles'].values():p['bin_roi']=regions[p['bin_id']]
    capture['statistics']['regions']=regions
    stats=StationStatistics(capture);source=StationCamera(capture,None,stats)
    row=source.raw(c['frame_timeout_s'])
    need(0<=time.monotonic()*1000-row['observation']['capture_started_ms']<=c['max_age_ms'],'single_frame_capture_expired')
    record=dict(image_sha256=row['source_bytes_sha256'],detections=detect_markers(row['image'],camera),
        source_frame_id=int(row['observation']['source_frame_id']),source_stream_id=row['observation']['source_stream_id'])
    pose=single_frame_pose(record,camera,station,kwargs.get('min_fixed_markers',3),kwargs.get('review_tolerance_factor',1))
    return [row],pose,stats


class PromptedStationStatistics(StationStatistics):
    def verify_cycle(self,after,bin_id,source,save_extra):
        if frame_count()!=1:return super().verify_cycle(after,bin_id,source,save_extra)
        need(len(after)==1,'single_frame_verification_requires_one_frame')
        evidence=self.evidence(after[0]);regions=evidence['regions']
        need(regions['source']['changed_px']>=self.c['min_bin_gain_px'] and
             regions[bin_id]['changed_px']>=self.c['min_bin_gain_px'],'station_source_or_bin_change_missing')
        for name,value in regions.items():
            if name in ('source',bin_id):continue
            limit=self.c['max_other_bin_changed_fraction'] if name.startswith('bin_') else self.c['max_scene_changed_fraction']
            need(value['changed_px']/value['area_px']<=limit,'station_unexpected_region_change:'+name)
        return dict(status='board_rgb_evidence_pass',provenance=self.provenance(),baseline=self.baseline,
            frames=[evidence],decision_frame_count=1,persistence_required=False,semantic_verified=False,
            independently_validated=False,physical_success_verified=False)
