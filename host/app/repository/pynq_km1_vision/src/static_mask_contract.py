"""CPU reference and recorded-register adapter for teammate HLS mask interfaces.

No PYNQ imports. RLE is full-image row-major; HLS bbox maxima are inclusive.
"""


def check(condition, reason):
    if not condition:
        raise ValueError(reason)


def decode_mask(runs, size):
    check(isinstance(size, list) and len(size) == 2 and
          all(type(v) is int for v in size) and 1 <= size[0] <= 640 and 1 <= size[1] <= 480,
          'invalid_image_size')
    check(isinstance(runs, list) and runs and
          all(type(v) is int and v >= 0 for v in runs) and
          sum(runs) == size[0] * size[1], 'invalid_mask')
    values = []
    for i, count in enumerate(runs):
        values.extend([i % 2] * count)
    return values


def validate_pixels(mask, size):
    check(isinstance(size, list) and len(size) == 2 and
          all(type(v) is int for v in size) and 1 <= size[0] <= 640 and 1 <= size[1] <= 480,
          'invalid_image_size')
    check(isinstance(mask, list) and len(mask) == size[0] * size[1] and
          all(type(v) is int and v in (0, 1) for v in mask), 'invalid_pixels')


def bounds(indices, width):
    if not indices:
        return dict(min_row=0, max_row=0, min_col=0, max_col=0)
    rows, cols = [i // width for i in indices], [i % width for i in indices]
    return dict(min_row=min(rows), max_row=max(rows), min_col=min(cols), max_col=max(cols))


def geometry(mask, size, job_id):
    validate_pixels(mask, size)
    points = [i for i, v in enumerate(mask) if v]
    xs, ys = [i % size[0] for i in points], [i // size[0] for i in points]
    return dict(m00=len(points), m10=sum(xs), m01=sum(ys),
                m20=sum(x*x for x in xs), m02=sum(y*y for y in ys),
                m11=sum(x*y for x, y in zip(xs, ys)),
                **bounds(points, size[0]), job_id=job_id, status=1)


def visual(before, after, size, job_id):
    validate_pixels(before, size)
    validate_pixels(after, size)
    changed = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    return dict(inter=sum(a and b for a, b in zip(before, after)),
                erased=sum(a and not b for a, b in zip(before, after)),
                beyond=sum(not a and b for a, b in zip(before, after)),
                union=sum(a or b for a, b in zip(before, after)),
                **bounds(changed, size[0]), job_id=job_id, status=1)


def compare_snapshot(snapshot, expected):
    """Reject error status, another job, missing fields or inconsistent statistics."""
    check(isinstance(snapshot, dict), 'invalid_pl_snapshot')
    check(type(snapshot.get('status')) is int and snapshot['status'] == 1, 'pl_status_error')
    for key, value in expected.items():
        check(type(snapshot.get(key)) is int and snapshot[key] == value, 'pl_mismatch:' + key)


def verification(request, size, source_mask):
    """Validate correlated recorded source/bin masks, optionally with PL snapshots.

    Binary source masks must represent the selected instance in source ROI.
    Bin masks represent foreground in a separate, fixed bin ROI. This is a
    replay evidence contract, not proof of physical grasp or material identity.
    """
    evidence = request['verification_masks']
    check(isinstance(evidence, dict), 'invalid_evidence')
    for key in ('task_id', 'request_id', 'camera_id', 'model_version'):
        check(evidence.get(key) == request.get(key), 'evidence_identity_mismatch:' + key)
    check(evidence.get('image_size') == size, 'evidence_size_mismatch')
    after_id = evidence.get('after_frame_id')
    check(type(after_id) is int and after_id > request['frame_id'] and
          type(evidence.get('before_frame_id')) is int and
          evidence['before_frame_id'] == request['frame_id'], 'evidence_frame_mismatch')
    result = {}
    for phase in ('source', 'bin'):
        pair = evidence[phase]
        before = decode_mask(pair['before_rle'], size)
        after = decode_mask(pair['after_rle'], size)
        if phase == 'source':
            check(before == source_mask, 'source_mask_mismatch')
        job_id = pair['job_id']
        check(type(job_id) is int and 0 <= job_id < 2**32, 'invalid_job_id')
        expected = visual(before, after, size, job_id)
        if 'pl_stats' in pair:
            compare_snapshot(pair['pl_stats'], expected)
        result[phase] = expected
    check(evidence['source']['job_id'] != evidence['bin']['job_id'], 'duplicate_evidence_job')
    return result
