"""Batch sampled route-height audit; no device or overlay access on import.

The PL result bounds the old 101-point nominal model, not continuous clearance.
Only numerically ambiguous threshold cases use an explicitly recorded CPU check.
"""
import math

MAX_SEGMENTS = 64
MAGIC = 0x4D504131
SAMPLES = 101


def checked_segments(segments):
    if not isinstance(segments, (list, tuple)) or not 1 <= len(segments) <= MAX_SEGMENTS:
        raise ValueError('motion_segment_count_outside_1_64')
    result = []
    for segment in segments:
        if not isinstance(segment, (list, tuple)) or len(segment) != 2:
            raise ValueError('motion_segment_needs_start_end')
        ends = []
        for endpoint in segment:
            if (not isinstance(endpoint, (list, tuple)) or len(endpoint) != 6 or
                    any(type(v) is not int or not 500 <= v <= 2500 for v in endpoint)):
                raise ValueError('motion_pwm_must_be_six_integers_500_2500')
            ends.append(list(endpoint))
        result.append(ends)
    return result


def sampled_minimum(segment):
    start, end = segment
    minimum = math.inf
    for i in range(SAMPLES):
        pwm = [a+(b-a)*i/100 for a, b in zip(start, end)]
        shoulder = math.radians(90+(pwm[1]-1500)*270/2000)
        elbow = math.radians((pwm[2]-1500)*270/2000)
        wrist = math.radians((pwm[3]-1500)*270/2000)
        angles = shoulder, shoulder-elbow, shoulder-elbow+wrist
        z = 100+sum(n*math.sin(a) for n, a in zip((105, 88, 155), angles))
        minimum = min(minimum, z)
    return minimum


def oracle(segments):
    return [sampled_minimum(segment) for segment in checked_segments(segments)]


def decode_results(words, segments, tag):
    segments = checked_segments(segments)
    if len(words) != len(segments)*8:
        raise ValueError('motion_result_length_changed')
    def signed(v):
        return v-(1 << 32) if v & (1 << 31) else v
    result = []
    for row, (start, end) in enumerate(segments):
        w = [int(v) for v in words[row*8:row*8+8]]
        cost = sum(abs(a-b) for a, b in zip(start, end))
        if (any(not 0 <= v <= 0xffffffff for v in w) or w[0] != MAGIC or
                w[1] != 0 or w[4] > 100 or w[5] != SAMPLES or w[6] != cost or w[7] != tag):
            raise ValueError('motion_result_identity_status_or_cost_mismatch')
        lower, upper = signed(w[2])/65536, signed(w[3])/65536
        if not -248.01 <= lower <= upper <= 448.01 or not .0039 <= upper-lower <= .004:
            raise ValueError('motion_result_invalid_height_interval')
        result.append(dict(minimum_z_lower_mm=lower, minimum_z_upper_mm=upper,
                           approximate_minimum_index=w[4], samples=SAMPLES, pwm_l1_cost=cost))
    return result


def audit_segments(segments, floor_mm, backend=None):
    """Use an optional batch hardware backend without changing the old threshold.

    The backend receives the entire route in one call, never one call per sample.
    A hardware exception propagates; it does not silently fall back to CPU.
    """
    segments = checked_segments(segments)
    if type(floor_mm) not in (int, float) or not math.isfinite(floor_mm):
        raise ValueError('finite_motion_floor_required')
    if backend is None:
        values = oracle(segments)
        return dict(backend='cpu_reference', samples_per_segment=SAMPLES,
                    segment_pass=[value >= floor_mm for value in values],
                    minimum_z_mm=values, boundary_cpu_segments=[], physical_motion=False,
                    collision_free_verified=False)
    report = backend.evaluate(segments)
    rows = report['segments']
    if len(rows) != len(segments):
        raise ValueError('motion_backend_wrong_segment_count')
    passes, boundary = [], []
    for i, (segment, row) in enumerate(zip(segments, rows)):
        lower, upper = row['minimum_z_lower_mm'], row['minimum_z_upper_mm']
        if not (math.isfinite(lower) and math.isfinite(upper) and
                -248.01 <= lower <= upper <= 448.01 and upper-lower <= .004 and row['samples'] == SAMPLES):
            raise ValueError('motion_backend_invalid_interval')
        if lower >= floor_mm:
            passes.append(True)
        elif upper < floor_mm:
            passes.append(False)
        else:
            boundary.append(i)
            passes.append(sampled_minimum(segment) >= floor_mm)
    return dict(backend=report['backend'], samples_per_segment=SAMPLES, segment_pass=passes,
                intervals=rows, boundary_cpu_segments=boundary, timing=report.get('timing', {}),
                physical_motion=False, collision_free_verified=False)


def route_segments(initial_pwm, steps):
    """Consume a compiled route's complete PWM targets; never parse or send UART."""
    state = list(initial_pwm)
    segments = []
    for step in steps:
        target = step['target_pwm']
        segments.append([state, target])
        state = list(target)
    return checked_segments(segments)
