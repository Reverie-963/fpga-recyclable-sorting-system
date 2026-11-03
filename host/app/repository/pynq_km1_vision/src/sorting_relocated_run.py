"""Prepare, review, preview and execute one calibrated known-item cycle."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

import cv2
import numpy as np

from .sorting_io import HTTPYoloe, Statistics
from .sorting_known import bound_file, file_hash, known_readiness, validate_known_config
from .sorting_loop import Journal, need
from .sorting_station_io import StationCamera, StationStatistics, station_policy
from .station_geometry import detect_markers, fingerprint, localize_records
from .station_relocalization import legacy_domains
from .sorting_relocated_geometry import RelocatedGeometry, scene_from_frame
from .sorting_relocated import RelocatedController, RelocatedMotion
from .sorting_relocated_camera import WindowedStationCamera
from .sorting_review_policy import review_factor


SOURCES=('sorting_relocated_run.py','sorting_relocated.py','sorting_relocated_geometry.py','sorting_relocated_plan.py','sorting_relocated_camera.py',
         'sorting_known.py','station_geometry.py','station_relocalization.py','sorting_station_io.py')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def motion_shift_limit(binding):
    policy=binding.get('motion_guard_policy')
    if policy is None:return 1
    need(policy.get('kind')=='operator-approved-station-motion-margin-v1' and
         policy.get('operator_statement') and policy.get('production_point_domains_changed') is False,
         'station_motion_margin_requires_explicit_operator_record')
    value=policy.get('max_displacement_px')
    need(type(value) in (int,float) and np.isfinite(value) and 1<=value<=5*review_factor(binding),
         'invalid_station_motion_shift_limit')
    return value
def write(path,data):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
def ref(path,base):return dict(path=os.path.relpath(Path(path).resolve(),Path(base).resolve()),sha256=file_hash(path))


def motion_marker_limit(binding):
    policy=binding.get('motion_guard_policy',{})
    minimum=policy.get('min_fixed_markers',3)
    need(type(minimum) is int and 2<=minimum<=4,'invalid_min_fixed_markers')
    need(minimum==3 or bool(policy.get('marker_count_operator_statement')),
         'marker_count_change_requires_operator_statement')
    return minimum


def initial_marker_shift_limit(binding):
    policy=binding.get('motion_guard_policy',{})
    limit=policy.get('initial_max_displacement_px',.5)
    need(type(limit) in (int,float) and np.isfinite(limit) and .5<=limit<=5*review_factor(binding),
         'invalid_initial_marker_shift_limit')
    need(limit==.5 or bool(policy.get('initial_localization_operator_statement')),
         'initial_marker_shift_requires_operator_statement')
    return limit


def initial_window_age_limit(binding,c):
    policy=binding.get('motion_guard_policy',{})
    value=policy.get('initial_window_max_age_ms',c['max_age_ms'])
    need(type(value) in (int,float) and np.isfinite(value) and
         c['max_age_ms']<=value<=c['max_age_ms']*1.5,'invalid_initial_window_age_limit')
    need(value==c['max_age_ms'] or bool(policy.get('initial_window_operator_statement')),
         'initial_window_age_requires_operator_statement')
    return value


def current_source_config(path):
    c=read(path);validate_known_config(c)
    need(c['statistics']['backend']=='pl_board_local','relocation_requires_board_vision')
    return c


def calibration_files(root):
    root=Path(root).resolve();session=root/'session.json'
    station=Path(__file__).parents[1]/'config/station_markers.design.json'
    # The station design is identity-bound by the completed guided session.
    files=dict(camera=root/'camera.json',registration=root/'world-registration.json',station=station)
    camera,registration,s=map(read,(files['camera'],files['registration'],station))
    need(read(session)['station_sha256']==fingerprint(s),'guided_station_identity_changed')
    checks=[]
    for path in root.glob('verify-*-registration.json'):
        d=read(path)
        if d.get('previous_registration_sha256')==fingerprint(registration) and d.get('move_check_max_mm',99)<=1.5:
            checks.append(path)
    need(checks,'completed_camera_move_verification_required')
    files['verification']=sorted(checks)[-1]
    return files,camera,s,registration


def raw_window(c,camera,station,registration,scene=None,min_fixed_markers=3,max_marker_shift_px=.5,initial_window_max_age_ms=None,review_tolerance_factor=1):
    # Region policy may already belong to the preceding relocated cycle. Read
    # its identity without claiming those pixels are the next cycle's regions.
    url=c['statistics']['url'].rstrip('/')+'/health'
    health=json.load(urllib.request.urlopen(url,timeout=c['statistics'].get('timeout_s',3)))
    need(health['assets']==c['statistics']['assets'] and health['image_size']==c['image_size'] and
         health['threshold']==c['statistics']['threshold'],'station_capture_identity_changed')
    capture=copy.deepcopy(c);rs=health['regions'];capture['source_roi']=rs['source']
    capture['layout']['static_rois']=[rs['static_a'],rs['static_b']]
    for p in capture['profiles'].values():p['bin_roi']=rs[p['bin_id']]
    capture['statistics']['regions']=rs
    stats=StationStatistics(capture);source=StationCamera(capture,None,stats)
    rows=[]
    for _ in range(3):rows.append(source.raw(c['frame_timeout_s']))
    now=time.monotonic()*1000
    window=c['max_age_ms'] if initial_window_max_age_ms is None else initial_window_max_age_ms
    need(type(window) in (int,float) and np.isfinite(window) and
         c['max_age_ms']<=window<=c['max_age_ms']*1.5,'invalid_initial_window_age_limit')
    need(all(0<=now-r['observation']['capture_started_ms']<=window for r in rows),
         'relocation_observation_window_expired')
    records=[dict(image_sha256=r['source_bytes_sha256'],detections=detect_markers(r['image'],camera),
        source_frame_id=int(r['observation']['source_frame_id']),source_stream_id=r['observation']['source_stream_id']) for r in rows]
    profiles={label:dict(kind='station-legacy-profile-v1',profile_id=p['object_profile_id'],
        object_height_mm=p['object_height_mm'],checked_card_polygons_mm=p['pick_hulls_card_mm'],
        card_to_grip=p['card_to_grip_mm'],card_to_hover=p['card_to_hover_mm'],
        grip_model_z_mm=p['grasp_z_mm'],hover_model_z_mm=p['hover_z_mm']) for label,p in c['profiles'].items()}
    domains=legacy_domains(registration,profiles,station)
    points=[point for domain in domains.values() for point in domain['world_xyz_mm']]
    if scene:points += [point for data in scene['scene_regions'].values() for point in data['world_xyz_mm']]
    pose=localize_records(records,camera,station,'fixed_marker_task_space',points,min_fixed_markers=min_fixed_markers,
                          max_marker_shift_px=max_marker_shift_px,review_factor=review_tolerance_factor)
    return rows,pose,stats


def prepare(config,calibration,output,heights):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    c=current_source_config(config);files,camera,station,registration=calibration_files(calibration)
    rows,pose,_=raw_window(c,camera,station,registration)
    scene=scene_from_frame(c,rows[-1],pose,camera,station,registration,
                           dict(zip(('bin_plastic','bin_metal','bin_paper'),heights)))
    image=output/'scene-reference.png';need(cv2.imwrite(str(image),rows[-1]['image']),'cannot_save_scene_image')
    write(output/'scene.json',scene)
    for i,row in enumerate(rows):
        need(cv2.imwrite(str(output/f'calibration-{i}.png'),row['image']),'cannot_save_calibration_image')
    drawing=rows[-1]['image'].copy()
    for name,rect in dict(source=c['source_roi'],**{p['bin_id']:p['bin_roi'] for p in c['profiles'].values()}).items():
        x0,y0,x1,y1=rect;cv2.rectangle(drawing,(x0,y0),(x1-1,y1-1),(0,210,230),2)
        cv2.putText(drawing,name,(x0+3,y0+18),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,120,220),1,cv2.LINE_AA)
    cv2.imwrite(str(output/'scene-review.png'),drawing)
    files.update(config=Path(config).resolve(),scene=output/'scene.json',empty_image=image)
    binding=dict(kind='known-relocated-integration-v1',files={k:ref(v,output) for k,v in files.items()},
        implementation_sources=[ref(Path(__file__).parent/name,output) for name in SOURCES],
        scene_geometry_confirmed=False,empty_bins_confirmed=False,physical_cycle_verified=False,
        reference_stream_id=rows[-1]['observation']['source_stream_id'],
        explicit_bin_floor_heights_mm=dict(zip(('bin_plastic','bin_metal','bin_paper'),heights)),
        mechanical_profiles_changed=False,unattended_accepted=False)
    write(output/'binding.json',binding)
    return output/'binding.json'


def load_binding(path):
    path=Path(path).resolve();b=read(path);need(b['kind']=='known-relocated-integration-v1','invalid_station_binding')
    motion_shift_limit(b)
    files={k:bound_file(path.parent,v) for k,v in b['files'].items()}
    for record in b['implementation_sources']:bound_file(path.parent,record)
    need(review_factor(b)==1 or any(Path(r['path']).name=='sorting_review_policy.py'
         for r in b['implementation_sources']),'review_policy_implementation_not_bound')
    c=current_source_config(files['config']);base=files['config'].parent
    if 'position_policy' in files:
        from .sorting_position_controller import validate_position_policy
        validate_position_policy(read(files['position_policy']),c)
        required={'sorting_position_controller.py','sorting_position_plan.py'}
        bound={Path(v['path']).name for v in b['implementation_sources']}
        need(required.issubset(bound),'position_implementation_must_be_bound')
        need({'position_trial','position_visual','position_route'}.issubset(files),
             'successful_position_trial_evidence_required')
        trial,visual,route=(read(files[k]) for k in ('position_trial','position_visual','position_route'))
        need(trial['task_id']==visual['task_id'] and
             trial['status']=='supervised_precomputed_position_cycle_visual_pass' and
             visual['status']=='supervised_visual_evidence_pass' and
             visual.get('original_neutral_pregrasp_pl_baseline') is True and
             visual.get('board',{}).get('status')=='board_rgb_evidence_pass' and
             all(s['transport_receipt']['echo_matched'] and s['transport_receipt']['completed_movement_wait']
                 for s in trial['steps']) and
             route['model_sha256']==fingerprint(read(files['position_policy'])['model']),
             'position_trial_evidence_does_not_match_model')
    camera,station,registration,scene=(read(files[k]) for k in ('camera','station','registration','scene'))
    need(scene['reference_image_sha256']==hashlib.sha256(cv2.imread(str(files['empty_image'])).tobytes()).hexdigest(),
         'scene_reference_pixels_changed')
    return b,c,base,camera,station,registration,scene,files


def integration_blockers(binding,c,base,files):
    blockers=known_readiness(c,base,require_empty=False,require_vision_binding=False)
    # The old camera's two patches are deliberately replaced by three known
    # ArUco marks. Its old layout image does not qualify the new image mapping.
    blockers=[v for v in blockers if not v.startswith('layout_reference:')]
    if not binding.get('scene_geometry_confirmed'):blockers.append('station_scene_geometry_needs_review')
    if not binding.get('empty_bins_confirmed'):blockers.append('station_empty_bins_reference_needs_confirmation')
    journal=(base/c['journal_dir']).resolve()
    if (journal/'pending.json').exists():blockers.append('existing_physical_trial_unresolved')
    for name in ('camera','station','registration'):
        if not files[name].is_file():blockers.append('station_calibration_missing:'+name)
    return blockers


def confirm(path):
    b,c,base,cam,st,reg,scene,files=load_binding(path)
    # This operation records actual user review of the annotated image and
    # empty boxes; no serial or PL calls and no old record is overwritten.
    scene=dict(scene,scene_geometry_confirmed=True,empty_bins_confirmed=True)
    scene_path=Path(path).parent/('scene-confirmed-'+str(time.time_ns())+'.json');write(scene_path,scene)
    b=copy.deepcopy(b);b['files']['scene']=ref(scene_path,Path(path).parent)
    b.update(scene_geometry_confirmed=True,empty_bins_confirmed=True,
             scene_confirmation_source='direct_operator_review_of_scene-review.png_and_empty_bins')
    result=Path(path).parent/('binding-confirmed-'+str(time.time_ns())+'.json');write(result,b)
    return result


def activate_regions(stats,geometry,c,base):
    # Shared with the existing arm owner. A pending physical task blocks even
    # if its serial port is temporarily closed while waiting for feedback.
    need(not ((base/c['journal_dir']).resolve()/'pending.json').exists(),'existing_physical_trial_unresolved')
    lease=(base/c['known']['device_lease']).resolve();lease.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(lease,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    try:
        os.write(fd,str(os.getpid()).encode());os.fsync(fd)
        need(not ((base/c['journal_dir']).resolve()/'pending.json').exists(),'existing_physical_trial_unresolved')
        current=stats.checked(stats.request('/health'))
        response=stats.request('/regions',dict(stream_id=current['stream_id'],frame_id=current['frame_id'],
            expected_config_sha256=current['config_sha256'],expected_baseline_frame_id=current['baseline_frame_id'],
            regions=geometry.runtime['statistics']['regions'],calibration_sha256=geometry.signature))
        need(response['baseline_invalidated'] and response['calibration_sha256']==geometry.signature and
             response['config_sha256']==station_policy(geometry.runtime),'station_region_activation_mismatch')
        return response
    finally:
        os.close(fd);lease.unlink()


def run(path,mode,ready=False,motion_profile=None):
    b,c,base,camera,station,registration,scene,files=load_binding(path)
    position_policy=read(files['position_policy']) if 'position_policy' in files else None
    blockers=integration_blockers(b,c,base,files)
    if motion_profile is not None:c['known']['motion_profile']=motion_profile
    health=json.load(urllib.request.urlopen(c['statistics']['url'].rstrip('/')+'/health',timeout=3))
    if 'region_binding_v1' not in health.get('capabilities',[]):blockers.append('board_region_binding_update_not_deployed')
    if mode=='preflight':return dict(status='station_integration_ready' if not blockers else 'station_integration_waiting',
        execution_blockers=blockers,physical_motion=False,physical_cycle_verified=False,
        motion_shift_limit_px=motion_shift_limit(b),
        route_mode='supervised_position_model' if position_policy else 'legacy_checked_points')
    need(mode in ('preview','execute'),'invalid_station_mode')
    if mode=='execute':need(ready and not blockers,'station_setup_not_ready:'+','.join(blockers))
    rows,pose,old_stats=raw_window(c,camera,station,registration,scene,min_fixed_markers=motion_marker_limit(b),
                                max_marker_shift_px=initial_marker_shift_limit(b),
                                initial_window_max_age_ms=initial_window_age_limit(b,c),review_tolerance_factor=review_factor(b))
    geometry=RelocatedGeometry(c,camera,station,registration,scene,pose,rows[-1]['image'],
                               motion_shift_px=motion_shift_limit(b),min_fixed_markers=motion_marker_limit(b),
                               review_tolerance_factor=review_factor(b))
    if position_policy:
        from .sorting_position_controller import apply_position_geometry
        apply_position_geometry(geometry,position_policy)
    geometry.runtime['known']['expected_camera_stream_id']=rows[-1]['observation']['source_stream_id']
    reference=cv2.imread(str(files['empty_image']))
    empty=geometry.warp_empty_reference(reference) if scene.get('empty_bins_confirmed') else geometry.reference.copy()
    # Preview never reconfigures the board, binds a baseline, opens USB or
    # resolves another trial's pending ledger.
    if mode=='execute':
        activation=activate_regions(old_stats,geometry,c,base)
        stats=StationStatistics(geometry.runtime)
        source=StationCamera(geometry.runtime,HTTPYoloe(c,None),stats)
    else:
        activation=None;stats=Statistics()
        source=StationCamera(old_stats.c,HTTPYoloe(c,None),old_stats)
    if position_policy:
        from .sorting_position_controller import PositionWindowedCamera
        source=PositionWindowedCamera(source)
    else:source=WindowedStationCamera(source)
    ledger=(base/c['journal_dir']).resolve() if mode=='execute' else Path(path).parent/'ledger'
    with Journal(ledger) as journal:
        need(journal.pending() is None,'unresolved_relocated_trial')
        if mode=='execute':
            motion=RelocatedMotion(geometry.runtime,journal,geometry,(base/c['known']['device_lease']).resolve(),
                [(base/p).resolve() for p in c['known']['stop_files']],
                (base/c['journal_dir']).resolve()/'pending.json')
        else:motion=None
        controller_class=RelocatedController;extra={}
        if position_policy:
            from .sorting_position_controller import PositionController
            controller_class=PositionController;extra['position_policy']=position_policy
        controller=controller_class(c,source,stats,motion,journal,geometry,empty,base,
            mode='known_physical' if mode=='execute' else 'preview',ready=ready,
            motion_profile=c['known'].get('motion_profile','qualified'),**extra)
        report=controller.cycle()
        report['integration_binding_sha256']=file_hash(path)
        report['region_activation']=activation
        report['execution_blockers']=blockers if mode=='preview' else report.get('execution_blockers',[])
        report['projected_regions']=geometry.runtime['statistics']['regions']
        report['physical_motion_attempted']=mode=='execute' and bool(report.get('steps'))
        journal.save(report['task_id']+'.json',report)
        return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path)
    p.add_argument('--binding',type=Path);p.add_argument('--calibration',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--bin-floor-heights',nargs=3,type=float,metavar=('PLASTIC','METAL','PAPER'))
    action=p.add_mutually_exclusive_group(required=True)
    for name in ('prepare','confirm-scene','preflight','preview','execute'):action.add_argument('--'+name,action='store_true')
    p.add_argument('--ready',action='store_true');p.add_argument('--motion-profile',choices=('qualified','compact'));args=p.parse_args()
    if args.prepare:
        need(args.config and args.calibration and args.output and args.bin_floor_heights is not None,'prepare_arguments_required')
        result=dict(status='scene_review_required',binding=str(prepare(args.config,args.calibration,args.output,args.bin_floor_heights)),physical_motion=False)
    elif args.confirm_scene:
        need(args.binding and args.ready,'explicit_scene_and_empty_bin_confirmation_required')
        result=dict(status='scene_confirmed_no_motion',binding=str(confirm(args.binding)),physical_motion=False)
    else:
        need(args.binding,'binding_required')
        if args.config:
            bound_config=bound_file(args.binding.resolve().parent,read(args.binding)['files']['config'])
            need(args.config.resolve()==bound_config,'explicit_config_does_not_match_station_binding')
        result=run(args.binding,'execute' if args.execute else 'preview' if args.preview else 'preflight',args.ready,args.motion_profile)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    return 2 if result.get('status') in ('rejected','needs_review') else 0


if __name__=='__main__':raise SystemExit(main())
