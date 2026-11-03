"""Freeze and audit existing three-item commissioning evidence; no motion I/O."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .station_calibration import read_image, write_json
from .station_geometry import import_legacy, map_xy, need
from openmv_gripper.km1_arm_bridge import validate_packet


def audit_phase(plan, execution, plan_sha, current):
    need(current == plan['start_pwm'], 'historical_phase_start_discontinuity')
    need(execution['plan_sha256'] == plan_sha and execution['status'] == 'sent_pending_observation'
         and execution['serial_closed'] is True, 'historical_execution_incomplete')
    commands = [s for stage in plan['stages'] for s in stage['steps']]
    need(len(commands) == len(execution['steps']), 'historical_step_count_mismatch')
    current = list(current)
    for source, sent in zip(commands, execution['steps']):
        payload = source['command'].encode('ascii')
        need(source['command'] == sent['command'] and sent['echo_matched'] is True
             and sent['bytes_written'] == len(payload) and payload in bytes.fromhex(sent['received_hex']),
             'historical_echo_or_command_mismatch')
        need(validate_packet(source['command'], ((500, 2500),)*6) == source['duration_ms'], 'historical_protocol_mismatch')
        for offset in range(1, len(source['command'])-1, 15):
            word = source['command'][offset:offset+15]
            current[int(word[1:4])] = int(word[5:9])
    return commands, current


def archive_three_cycles(session, output):
    session, output = Path(session).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/'sources').mkdir(); (output/'seeds').mkdir()
    frozen = {}
    def read(relative, expected=None):
        path = (session/relative).resolve()
        need(path.is_relative_to(session), 'history_path_outside_session')
        raw = path.read_bytes(); sha = hashlib.sha256(raw).hexdigest()
        need(expected is None or sha == expected, 'historical_hash_mismatch:'+str(relative))
        if sha not in frozen:
            (output/'sources'/(sha+'.json')).write_bytes(raw)
        frozen[sha] = str(path.relative_to(session))
        return json.loads(raw), sha

    def execution_file(folder):
        matches = []
        for path in (session/folder/'journal').glob('*.json'):
            row = json.loads(path.read_text(encoding='utf-8-sig'))
            if all(k in row for k in ('steps', 'serial_closed', 'plan_sha256', 'status')):
                matches.append(path)
        need(len(matches) == 1, 'expected_one_phase_execution')
        return matches[0]

    def phases(folders):
        rows, current, start = [], None, None
        for folder in folders:
            plan, sha = read(folder+'/plan.json')
            if current is None:
                current = list(plan['start_pwm']); start = list(current)
            execution, execution_sha = read(str(execution_file(folder).relative_to(session)))
            commands, current = audit_phase(plan, execution, sha, current)
            for source in commands:
                rows.append(dict(source, source_phase=folder, plan_sha256=sha, execution_sha256=execution_sha))
        return dict(kind='historical-path-only-v1', start_pwm=start, final_commanded_pwm=current,
            steps=rows, command_count=len(rows), automatic_motion_allowed=False)

    classes = {}
    carton, carton_sha = read('same-carton-calibration.json')
    can, can_sha = read('same-can-calibration-v8.json')
    for material, prefix in (('paper', 'known-carton-11s-geometry'), ('metal', 'known-can-final-review'), ('plastic', 'known-plastic-final-review')):
        review, review_sha = read(prefix+'/final-review.json')
        feedback, feedback_sha = read(prefix+'/operator-feedback.json')
        need(review['status'] == 'supervised_trial_visual_evidence_pass', 'historical_visual_cycle_not_passed')
        if material == 'paper':
            folders = ['known-carton-11s-geometry']
            need(feedback.get('source') == 'Direct user in this chat' and feedback.get('text'), 'paper_feedback_missing')
            profile = import_legacy(carton, carton_sha, 'same-carton-65mm')
            parameters = dict(height_mm=65, open_pwm=1000, close_pwm=1500, bin_id='bin_paper')
        else:
            name = 'metal56' if material == 'metal' else 'plastic53'
            record, record_sha = read(name+'/first-supervised-cycle-record.json')
            need(record['supervised_known_item_cycle_completed'] is True and record['operator_confirmed_no_problem'] is True
                 and record['final_review_sha256'] == review_sha and record['operator_feedback_sha256'] == feedback_sha,
                 'historical_cycle_record_binding_mismatch')
            folders = [p['folder'] for p in record['phase_executions']]
            for phase in record['phase_executions']:
                read(phase['folder']+'/plan.json', phase['plan_sha256'])
                read(str(execution_file(phase['folder']).relative_to(session)), phase['execution_sha256'])
            parameters = dict(height_mm=record['dimensions_mm']['diameter'], length_mm=record['dimensions_mm']['length'],
                open_pwm=record['open_pwm'], close_pwm=record['close_pwm'], bin_id=record['bin_id'])
            profile_source = copy.deepcopy(can)
            if material == 'plastic':
                # Explicitly supported by this plastic cycle's bound approach,
                # not inferred from a neighbouring height or material.
                approach, approach_sha = read('known-plastic-approach/plan.json')
                plane, plane_sha = read('plastic53/card-plane.json', approach['calibration_source_hashes']['plastic53/card-plane.json'])
                grip, grip_sha = read('metal56/grip-affine-checked.json', approach['calibration_source_hashes']['metal56/grip-affine-checked.json'])
                hover, hover_sha = read('metal56/hover-registration-checked.json', approach['calibration_source_hashes']['metal56/hover-registration-checked.json'])
                need(np.allclose(map_xy(grip['card_to_nominal'], approach['pick_card_xy_mm']), approach['grip_xyz_mm'][:2])
                     and np.allclose(map_xy(hover['card_to_nominal_hover'], approach['pick_card_xy_mm']), approach['hover_xyz_mm'][:2]),
                     'plastic_actual_plan_mapping_differs')
                profile_source.update(object_height_mm=53, card_to_grip=grip['card_to_nominal'],
                    card_to_hover=hover['card_to_nominal_hover'], grip_model_z_mm=10, hover_model_z_mm=70, working_alpha_deg=-75)
                source = dict(cycle_record=record_sha, approach=approach_sha, plane=plane_sha, grip=grip_sha, hover=hover_sha)
                profile = import_legacy(profile_source, record_sha, 'same-plastic-53mm')
                profile['actual_cycle_mapping_sources'] = source
            else:
                profile = import_legacy(profile_source, can_sha, 'same-can-56mm')
            release, release_sha = read(record['bin_release_empty_pose_file'])
            parameters.update(bin_release=release, bin_release_source_sha256=release_sha)
        route = phases(folders)
        write_json(output/(material+'-route.json'), route)
        write_json(output/(material+'-profile.json'), profile)
        classes[material] = dict(parameters=parameters, historical_commands=route['command_count'],
            route_file=material+'-route.json', profile_file=material+'-profile.json',
            final_review_sha256=review_sha, operator_feedback_sha256=feedback_sha,
            operator_text=feedback.get('original_text', feedback.get('text')),
            scope='same_known_item_one_supervised_cycle', quality_gate_passed=False, unattended_accepted=False)

    # Planar card images can seed intrinsics across camera placements ONLY when
    # the operator confirms the same lens/focus settings. Pose-specific robot
    # homographies remain separate. Decode hash removes JPEG/PNG duplicates.
    seeds, seen = [], set()
    candidates = list(session.rglob('*plane*.json'))
    for candidate in candidates:
        try:
            data = json.loads(candidate.read_text(encoding='utf-8-sig'))
            if data.get('camera_id') != 'pynq-c270' or len(data.get('corner_pixels', {})) != 20:
                continue
            path = Path(data.get('image', ''))
            if not path.is_file():
                continue
            if not path.resolve().is_relative_to(session):
                continue
            image, sha = read_image(path)
            pixel_sha = hashlib.sha256(image.tobytes()).hexdigest()
            if pixel_sha in seen:
                continue
            seen.add(pixel_sha)
            filename = 'seed-'+str(len(seeds))+path.suffix.lower()
            (output/'seeds'/filename).write_bytes(path.read_bytes())
            _, data_sha = read(str(candidate.relative_to(session)))
            seeds.append(dict(file='seeds/'+filename, file_sha256=sha, decoded_bgr_sha256=pixel_sha,
                camera_id=data['camera_id'], image_size=data['image_size'], corner_source_sha256=data_sha,
                corner_pixels=data['corner_pixels'], lens_settings_confirmed=False))
        except (KeyError, ValueError, OSError):
            continue
    # Latest unified captures may not yet have a planar JSON. They can still
    # contribute intrinsics-only image candidates after operator confirmation.
    for path in sorted(session.rglob('raised-card*')):
        if path.suffix.lower() not in ('.png', '.jpg') or not path.is_file():
            continue
        try:
            image, sha = read_image(path); pixel_sha = hashlib.sha256(image.tobytes()).hexdigest()
            if pixel_sha in seen or list(image.shape[:2][::-1]) != [640, 480]:
                continue
            ok, points = cv2.findChessboardCornersSB(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 4))
            if not ok:
                continue
            seen.add(pixel_sha); filename = 'seed-'+str(len(seeds))+path.suffix.lower()
            (output/'seeds'/filename).write_bytes(path.read_bytes())
            seeds.append(dict(file='seeds/'+filename, file_sha256=sha, decoded_bgr_sha256=pixel_sha,
                source_image=str(path.relative_to(session)), camera_id='pynq-c270', image_size=[640, 480],
                corner_order='automatic_for_intrinsics_only', lens_settings_confirmed=False))
        except (ValueError, OSError):
            continue
    report = dict(kind='three-cycle-reuse-bundle-v1', classes=classes, intrinsics_seeds=seeds,
        source_hashes=frozen, all_three_known_cycles_present=True,
        historical_command_count=sum(v['historical_commands'] for v in classes.values()),
        missing=['measured_lens_intrinsics', 'fixed_markers_to_legacy_card_registration'],
        automatic_motion_allowed=False, physical_precision_accepted=False)
    write_json(output/'bundle.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); report = archive_three_cycles(args.session, args.output)
    print(json.dumps(dict(classes=list(report['classes']), commands=report['historical_command_count'],
        camera_seed_candidates=len(report['intrinsics_seeds']), automatic_motion_allowed=False)))


if __name__ == '__main__': main()
