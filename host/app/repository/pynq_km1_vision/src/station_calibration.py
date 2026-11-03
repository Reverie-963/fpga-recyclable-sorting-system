"""Recorded-image calibration, reusable legacy profiles and printable markers.

Every command is offline. No camera capture, network, serial or execution flag.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .station_geometry import (camera_parameters, detect_markers, fingerprint,
    fit_robot_affine, fit_world_registration, import_legacy, legacy_control, localize_records, need,
    pixel_at_height, refit_recorded_plane)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def read_image(path):
    raw = Path(path).read_bytes()
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    need(image is not None, 'cannot_decode_recorded_image')
    return image, hashlib.sha256(raw).hexdigest()


def generate_kit(folder):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    station = dict(kind='station-markers-v1', world_frame_id='fixed-board-origin-center-v1',
        dictionary='DICT_4X4_50', units='mm', axes='+X paper right; +Y paper top; +Z above paper',
        physical_board_installed=False, world_registration_checked=False,
        height_range_mm=[0, 120], work_polygon_world_xy_mm=[[-35, -50], [35, -50], [35, 65], [-35, 65]],
        markers=[], note='Design geometry only. Rigidly mount the whole sheet; verify actual printed scale before use.')
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="210mm" height="297mm" viewBox="0 0 210 297">',
           '<rect width="210" height="297" fill="white"/>',
           '<text x="105" y="12" text-anchor="middle" font-size="4">FIXED STATION / PRINT 100% / DO NOT FIT PAGE</text>']
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for sid, (x, y) in zip((10, 11, 12, 13), ((20, 30), (145, 30), (20, 210), (145, 210))):
        bits = cv2.aruco.generateImageMarker(dictionary, sid, 6)
        for row in range(6):
            for col in range(6):
                if bits[row, col] == 0:
                    svg.append(f'<rect x="{x+col*7.5}" y="{y+row*7.5}" width="7.5" height="7.5"/>')
        svg.append(f'<text x="{x+22.5}" y="{y+58}" text-anchor="middle" font-size="4">ID {sid} / black edge 45 mm</text>')
        station['markers'].append(dict(id=sid, corners_world_mm=[
            [x-105, 148.5-y, 0], [x+45-105, 148.5-y, 0],
            [x+45-105, 148.5-y-45, 0], [x-105, 148.5-y-45, 0]]))
    svg += ['<path d="M100 148.5h10M105 143.5v10" stroke="black" stroke-width="0.3"/>',
            '<text x="105" y="160" text-anchor="middle" font-size="4">WORLD ORIGIN / Z UP</text>',
            '<path d="M55 280h100M55 277v6M155 277v6" stroke="black" stroke-width="0.4"/>',
            '<text x="105" y="289" text-anchor="middle" font-size="4">MEASURE THIS LINE: 100 mm</text>', '</svg>']
    (folder/'fixed-markers-a4.svg').write_text('\n'.join(svg), encoding='utf-8')
    # Same 5x4 inner-corner, 20mm spacing convention used by existing cards.
    chess = ['<svg xmlns="http://www.w3.org/2000/svg" width="210mm" height="297mm" viewBox="0 0 210 297">',
             '<rect width="210" height="297" fill="white"/>']
    for row in range(5):
        for col in range(6):
            if (row+col) % 2 == 0:
                chess.append(f'<rect x="{45+col*20}" y="{85+row*20}" width="20" height="20"/>')
    chess += ['<text x="105" y="210" text-anchor="middle" font-size="4">5 x 4 INNER CORNERS / SQUARE 20 mm</text>',
              '<text x="105" y="220" text-anchor="middle" font-size="4">PRINT 100% / MEASURE ACTUAL SQUARE</text>', '</svg>']
    (folder/'intrinsics-chessboard-a4.svg').write_text('\n'.join(chess), encoding='utf-8')
    write_json(folder/'station.design.json', station)
    write_json(folder/'camera.pending.json', dict(kind='station-camera-v1', camera_id='pynq-c270',
        lens_profile_id='c270-640x480-fixed-focus-v1', image_size=[640, 480], camera_matrix=None,
        dist_coeffs=None, status='missing_measured_intrinsics_do_not_guess'))
    write_json(folder/'intrinsics-manifest.pending.json', dict(camera_id='pynq-c270',
        lens_profile_id='c270-640x480-fixed-focus-v1', image_size=[640, 480],
        pattern_inner_corners=[5, 4], square_mm=20, fit_images=[], check_images=[]))
    write_json(folder/'world-registration.pending.json', dict(world_frame_id=station['world_frame_id'],
        world_xy_mm=[], card_xy_mm=[], check_world_xy_mm=[], check_card_xy_mm=[]))
    return dict(status='offline_kit_created', output_dir=str(folder), automatic_motion_allowed=False)


def calibrate_intrinsics(manifest, root):
    minimum = manifest.get('minimum_fit_images', 12)
    need(type(minimum) is int and 10 <= minimum <= 30, 'minimum_fit_images_must_be_10_to_30')
    need(len(manifest['fit_images']) >= minimum and len(manifest['check_images']) >= 3,
         f'need_{minimum}_fit_and_3_independent_check_images')
    model = manifest.get('calibration_model', 'brown5')
    need(model in ('brown5', 'brown4'), 'unsupported_calibration_model')
    flags = cv2.CALIB_FIX_K3 if model == 'brown4' else 0
    pattern = manifest['pattern_inner_corners']
    need(isinstance(pattern, list) and len(pattern) == 2 and all(type(n) is int and n >= 4 for n in pattern), 'invalid_chessboard_pattern')
    square = manifest['square_mm']
    need(type(square) in (int, float) and np.isfinite(square) and square > 0, 'measured_square_size_required')
    obj = np.zeros((pattern[0]*pattern[1], 3), np.float32)
    obj[:, :2] = np.mgrid[:pattern[0], :pattern[1]].T.reshape(-1, 2)*square
    seen, fit, checks = set(), [], []
    hashes = []
    for split, names in ((fit, manifest['fit_images']), (checks, manifest['check_images'])):
        for name in names:
            image, sha = read_image(root/name)
            need(sha not in seen, 'duplicate_calibration_image_or_split_leakage')
            seen.add(sha); hashes.append(sha)
            need(list(image.shape[:2][::-1]) == manifest['image_size'], 'calibration_image_size_changed')
            ok, points = cv2.findChessboardCornersSB(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), tuple(pattern))
            need(ok, 'complete_chessboard_not_detected:'+str(name))
            split.append(points.astype(np.float32))
    centers = np.asarray([p.reshape(-1, 2).mean(0) for p in fit])
    need(np.all(np.ptp(centers, axis=0) >= np.asarray(manifest['image_size'])*.2),
         'insufficient_image_field_coverage')
    rms, k, d, rotations, translations = cv2.calibrateCamera([obj]*len(fit), fit, tuple(manifest['image_size']), None, None, flags=flags)
    normals = [cv2.Rodrigues(r)[0][:, 2] for r in rotations]
    diversity = max(np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1))) for a in normals for b in normals)
    need(diversity >= 15, 'insufficient_board_tilt_diversity_do_not_use_parallel_height_photos')
    fit_errors, check_errors = [], []
    for points, r, t in zip(fit, rotations, translations):
        fit_errors.extend(np.linalg.norm(cv2.projectPoints(obj, r, t, k, d)[0].reshape(-1, 2)-points.reshape(-1, 2), axis=1))
    for points in checks:
        ok, r, t = cv2.solvePnP(obj, points, k, d)
        need(ok, 'intrinsics_check_pose_failed')
        check_errors.extend(np.linalg.norm(cv2.projectPoints(obj, r, t, k, d)[0].reshape(-1, 2)-points.reshape(-1, 2), axis=1))
    check_rms = float(np.sqrt(np.mean(np.square(check_errors))))
    need(np.isfinite(rms) and rms <= .8 and check_rms <= .8 and max(check_errors) <= 1.5, 'intrinsics_fit_or_check_error')
    stability = None
    if manifest.get('require_leave_one_out_stability') is True:
        variations = []
        for skip in range(len(fit)):
            subset = [points for i, points in enumerate(fit) if i != skip]
            _, other, _, _, _ = cv2.calibrateCamera([obj]*len(subset), subset, tuple(manifest['image_size']), None, None, flags=flags)
            variations.append([abs(other[0, 0]/k[0, 0]-1), abs(other[1, 1]/k[1, 1]-1),
                               abs(other[0, 2]-k[0, 2]), abs(other[1, 2]-k[1, 2])])
        maximum = np.max(variations, axis=0)
        need(np.isfinite(maximum).all() and max(maximum[:2]) <= .025 and max(maximum[2:]) <= 8,
             'intrinsics_unstable_add_different_tilt_or_field_positions')
        stability = dict(max_focal_relative_change=float(max(maximum[:2])), max_center_change_px=float(max(maximum[2:])))
    camera = dict(kind='station-camera-v1', camera_id=manifest['camera_id'], lens_profile_id=manifest['lens_profile_id'],
        image_size=manifest['image_size'], camera_matrix=k.tolist(), dist_coeffs=d.ravel().tolist(),
        source='recorded_chessboard_calibration', fit_rms_px=float(rms), check_rms_px=check_rms,
        check_max_px=float(max(check_errors)), board_normal_spread_deg=float(diversity),
        image_sha256=hashes, physical_accuracy_verified=False, automatic_motion_allowed=False)
    camera.update(fit_images_count=len(fit), check_images_count=len(checks), leave_one_out_stability=stability,
        calibration_model=model, calibration_flags=int(flags),
        fixed_coefficients={'k3': 0.0} if model == 'brown4' else {}, manifest_sha256=fingerprint(manifest))
    camera_parameters(camera)
    return camera


def reuse_calibrations(manifest, root, output):
    output.mkdir(parents=True, exist_ok=False)
    (output/'sources').mkdir()
    report = dict(kind='station-calibration-reuse-v1', planes=[], profiles=[], robot_fits=[],
        camera_intrinsics_available=False, automatic_motion_allowed=False)
    def snapshot(entry, index):
        raw = (root/entry['path']).read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        (output/'sources'/f'{index}-{sha[:12]}.json').write_bytes(raw)
        return json.loads(raw), sha
    for index, entry in enumerate(manifest.get('planes', [])):
        source, sha = snapshot(entry, 'plane-'+str(index))
        fitted = refit_recorded_plane(source, entry['surface_height_mm'], entry['square_mm'])
        fitted.update(source_sha256=sha, source_name=Path(entry['path']).name)
        filename = f'plane-{index}.json'; write_json(output/filename, fitted)
        report['planes'].append(dict(height_mm=fitted['surface_height_mm'], fit_max_mm=fitted['fit_max_mm'],
            check_max_mm=fitted['check_max_mm'], output=filename, source_sha256=sha))
    for index, entry in enumerate(manifest.get('profiles', [])):
        source, sha = snapshot(entry, 'profile-'+str(index))
        profile = import_legacy(source, sha, entry['profile_id'])
        filename = f'profile-{index}.json'; write_json(output/filename, profile)
        report['profiles'].append(dict(profile_id=profile['profile_id'], height_mm=profile['object_height_mm'],
            output=filename, source_sha256=sha, world_registration_pending=True))
    for index, entry in enumerate(manifest.get('robot_fits', [])):
        source, sha = snapshot(entry, 'robot-'+str(index))
        fit = fit_robot_affine(source); fit['source_sha256'] = sha
        filename = f'robot-fit-{index}.json'; write_json(output/filename, fit)
        report['robot_fits'].append(dict(name=entry['name'], output=filename, source_sha256=sha,
            check_pair_residual_max_nominal_mm=fit['check_pair_residual_max_nominal_mm'],
            physical_error_quantitatively_known=False))
    write_json(output/'reuse-report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='action', required=True)
    kit = subs.add_parser('kit'); kit.add_argument('--output-dir', type=Path, required=True)
    reuse = subs.add_parser('reuse'); reuse.add_argument('--manifest', type=Path, required=True)
    reuse.add_argument('--output-dir', type=Path, required=True)
    calibration = subs.add_parser('intrinsics'); calibration.add_argument('--manifest', type=Path, required=True)
    calibration.add_argument('--output', type=Path, required=True)
    registration = subs.add_parser('register'); registration.add_argument('--pairs', type=Path, required=True)
    registration.add_argument('--output', type=Path, required=True)
    locate = subs.add_parser('localize')
    locate.add_argument('--images', nargs='+', type=Path, required=True)
    locate.add_argument('--camera', type=Path, required=True); locate.add_argument('--station', type=Path, required=True)
    locate.add_argument('--output', type=Path, required=True)
    target = subs.add_parser('point')
    for name in ('camera', 'station', 'pose', 'output'):
        target.add_argument('--'+name, type=Path, required=True)
    target.add_argument('--pixel', type=float, nargs=2, required=True)
    target.add_argument('--height-mm', type=float, required=True)
    target.add_argument('--profile', type=Path); target.add_argument('--registration', type=Path)
    args = parser.parse_args()
    if args.action == 'kit':
        result = generate_kit(args.output_dir)
    elif args.action == 'reuse':
        result = reuse_calibrations(read_json(args.manifest), args.manifest.resolve().parent, args.output_dir)
    elif args.action == 'intrinsics':
        result = calibrate_intrinsics(read_json(args.manifest), args.manifest.resolve().parent)
        write_json(args.output, result)
    elif args.action == 'register':
        result = fit_world_registration(read_json(args.pairs)); write_json(args.output, result)
    elif args.action == 'localize':
        camera, station = read_json(args.camera), read_json(args.station)
        records = []
        for path in args.images:
            image, sha = read_image(path)
            records.append(dict(image_sha256=sha, detections=detect_markers(image, camera)))
        result = localize_records(records, camera, station); write_json(args.output, result)
    else:
        camera, station, pose = read_json(args.camera), read_json(args.station), read_json(args.pose)
        result = pixel_at_height(args.pixel, args.height_mm, pose, camera, station)
        need(args.registration is None or args.profile is not None, 'registration_requires_profile')
        if args.profile:
            profile = read_json(args.profile)
            if args.registration: profile['world_registration'] = read_json(args.registration)
            result['legacy_control_preview'] = legacy_control(result, profile)
        write_json(args.output, result)
    if args.action == 'localize':
        summary = {k: result[k] for k in ('marker_ids', 'reprojection_rms_px', 'reprojection_max_px', 'automatic_motion_allowed')}
    elif args.action == 'intrinsics':
        summary = {k: result[k] for k in ('camera_id', 'fit_rms_px', 'check_rms_px', 'automatic_motion_allowed')}
    else:
        summary = result
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__': main()
