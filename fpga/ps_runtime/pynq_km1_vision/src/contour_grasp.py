"""Planar silhouette measurements and measured jaw interpolation; no device I/O.

Pixel cells, including hole boundaries, are projected at the supplied object
height. This is not a contact-force or three-dimensional collision model.
"""
import numpy as np
from .sorting_loop import finite, fingerprint, need


def boundary_edges(mask):
    mask = np.asarray(mask, dtype=bool)
    padded = np.pad(mask, 1)
    edges = []
    for neighbor, a, b in (
        (padded[:-2, 1:-1], (-.5, -.5), (.5, -.5)),
        (padded[2:, 1:-1], (-.5, .5), (.5, .5)),
        (padded[1:-1, :-2], (-.5, -.5), (-.5, .5)),
        (padded[1:-1, 2:], (.5, -.5), (.5, .5)),
    ):
        y, x = np.nonzero(mask & ~neighbor)
        points = np.c_[x, y]
        edges.append(np.stack((points+a, points+b), axis=1))
    return np.concatenate(edges)


def contour_section(mask, map_points, pixel, axis_deg):
    """Return the material interval containing the proposed grasp, not its hull.

    A hole or concavity splits the cross-section. Never span empty space just
    because its outer bounding rectangle is narrow enough.
    """
    edges = boundary_edges(mask)
    need(len(edges) > 0, 'contour_empty_mask')
    world = np.asarray(map_points(edges.reshape(-1, 2)), float).reshape(-1, 2, 2)
    center = np.asarray(map_points([pixel]), float)[0]
    need(np.isfinite(world).all() and np.isfinite(center).all(), 'contour_invalid_projection')
    angle = np.radians(axis_deg)
    along = np.array([np.cos(angle), np.sin(angle)])
    across = np.array([-along[1], along[0]])
    local = world-center
    v, u = local@along, local@across
    # Half-open crossing rule counts a vertex once, not twice.
    crosses = (v[:, 0] > 0) != (v[:, 1] > 0)
    v, u = v[crosses], u[crosses]
    hits = np.sort(u[:, 0]-v[:, 0]*(u[:, 1]-u[:, 0])/(v[:, 1]-v[:, 0]))
    need(len(hits) >= 2 and len(hits) % 2 == 0, 'contour_section_unresolved')
    intervals = hits.reshape(-1, 2)
    owned = intervals[(intervals[:, 0] < 0) & (intervals[:, 1] > 0)]
    need(len(owned) == 1, 'contour_grasp_not_inside_material')
    lo, hi = owned[0]
    contact = np.array([center+lo*across, center+hi*across])
    return dict(kind='projected-mask-cross-section-v1', width_mm=float(hi-lo),
                center_card_xy_mm=contact.mean(axis=0).tolist(),
                contact_card_xy_mm=contact.tolist(), axis_deg=float(axis_deg % 180),
                cross_section_intervals=len(intervals),
                center_shift_mm=float(abs((lo+hi)/2)),
                physical_contact_verified=False)


def validate_jaw_calibration(profile, policy):
    calibration = profile.get('jaw_calibration')
    shared = policy.get('jaw_calibration')
    if shared is not None:
        need(isinstance(policy.get('jaw_calibration_operator_statement'),str) and
             bool(policy['jaw_calibration_operator_statement'].strip()), 'jaw_activation_statement_required')
        need(calibration is None or calibration==shared,'jaw_conflicting_calibrations')
        calibration=shared
    if calibration is None:
        return None
    need(isinstance(calibration, dict) and calibration.get('kind') == 'measured-jaw-gap-v1'
         and isinstance(calibration.get('measurement_id'), str)
         and bool(calibration['measurement_id'].strip()), 'jaw_measurement_record_required')
    samples = calibration.get('samples')
    need(isinstance(samples, list) and 2 <= len(samples) <= 32, 'jaw_invalid_samples')
    need(all(isinstance(s, dict) and finite(s.get('gap_mm')) and s['gap_mm'] >= 0
             and type(s.get('pwm')) is int and 500 <= s['pwm'] <= 2500
             for s in samples), 'jaw_invalid_sample')
    gaps = np.array([s['gap_mm'] for s in samples], float)
    pwm = np.array([s['pwm'] for s in samples], float)
    need(np.all(np.diff(gaps) > 0) and
         (np.all(np.diff(pwm) > 0) or np.all(np.diff(pwm) < 0)), 'jaw_samples_not_monotonic')
    # Keep the existing open command and its conservative collision footprint.
    need(samples[-1]['pwm'] == profile['open_pwm'] and
         gaps[-1] <= policy['jaw_open_span_mm'], 'jaw_open_geometry_mismatch')
    for key in ('compression_mm', 'opening_margin_mm'):
        need(finite(calibration.get(key)) and 0 <= calibration[key] < gaps[-1],
             'jaw_invalid_'+key)
    need(finite(calibration.get('compression_ratio',0)) and
         0 <= calibration.get('compression_ratio',0) < 1,'jaw_invalid_compression_ratio')
    need(not (calibration.get('compression_ratio',0) and calibration['compression_mm']),
         'jaw_conflicting_compression_modes')
    return calibration


def jaw_command(profile, policy, width_mm=None):
    calibration = validate_jaw_calibration(profile, policy)
    if width_mm is not None:
        need(finite(width_mm) and width_mm > 0, 'jaw_invalid_width')
    if calibration is None:
        return dict(mode='legacy_fixed_pwm', close_pwm=profile['close_pwm'],
                    observed_width_mm=width_mm, width_controls_closure=False,
                    limitation='measured_jaw_gap_calibration_missing')
    need(width_mm is not None, 'jaw_contour_width_required')
    samples = calibration['samples']
    target = width_mm*(1-calibration.get('compression_ratio',0))-calibration['compression_mm']
    extension=None
    if target < samples[0]['gap_mm']:
        extension=policy.get('jaw_gap_extension')
        need(isinstance(extension,dict) and extension.get('kind')=='operator-supervised-linear-extension-v1'
             and extension.get('operator_statement') and extension.get('physical_validation_completed') is False,
             'jaw_target_outside_measured_range')
        floor=extension.get('min_gap_mm')
        need(finite(floor) and 0<=floor<samples[0]['gap_mm'] and samples[0]['gap_mm']-floor<=1
             and floor<=target,'jaw_target_outside_requested_extension')
    need(target <= samples[-1]['gap_mm'],'jaw_target_outside_measured_range')
    need(width_mm+2*calibration['opening_margin_mm'] <= samples[-1]['gap_mm'],
         'jaw_insufficient_opening')
    pwm = int(round(float(np.interp(target, [s['gap_mm'] for s in samples],
                                   [s['pwm'] for s in samples]))))
    if extension is not None:
        low,high=samples[:2]
        pwm=int(round(low['pwm']+(target-low['gap_mm'])*(high['pwm']-low['pwm'])/(high['gap_mm']-low['gap_mm'])))
        need(500<=pwm<=2500,'jaw_estimated_pwm_outside_protocol_range')
    result=dict(mode='estimated_width_extrapolation' if extension is not None else 'measured_width_interpolation', close_pwm=pwm,
                observed_width_mm=width_mm, target_gap_mm=target,
                width_controls_closure=True, calibration_sha256=fingerprint(calibration),
                measurement_id=calibration['measurement_id'], physical_contact_verified=False)
    if extension is not None:
        result['linear_extension']=dict(min_gap_mm=extension['min_gap_mm'],source_samples=samples[:2],
            operator_statement=extension['operator_statement'],measured=False,physical_validation_completed=False)
    return result
