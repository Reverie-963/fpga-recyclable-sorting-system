"""Exact CPU oracle for the candidate full-frame FPGA pipeline. No devices."""
import cv2
import numpy as np

from .station_vision import pixels, validate_regions, REGION_NAMES

CONTRACT = 'station-frame-pipeline-v1'
RESULT_WORDS = 88
PLANES = ('gray', 'smooth', 'edges', 'clean_change')


def process_frame(baseline, current, regions, threshold=45, history=None, preview_enabled=True):
    """Zero borders, rounded 3x3 mean, saturated L1 Sobel, binary 3x3 opening."""
    baseline, current = pixels(baseline), pixels(current)
    size = list(current.shape[1::-1])
    validate_regions(regions, size)
    if baseline.shape != current.shape or type(threshold) is not int or not 0 <= threshold <= 765:
        raise ValueError('invalid_rgb_comparison')
    if type(preview_enabled) is not bool:
        raise ValueError('invalid_preview_flag')
    delta = np.abs(current.astype(np.int16) - baseline.astype(np.int16)).sum(axis=2, dtype=np.uint16)
    changed = delta > threshold
    if history is not None and (history.shape != changed.shape or history.dtype != np.bool_):
        raise ValueError('invalid_history')
    persistent = changed.copy() if history is None else changed & history
    b, g, r = (current[:, :, i].astype(np.uint32) for i in range(3))
    gray = ((29*b + 150*g + 77*r + 128) >> 8).astype(np.uint8)
    sums = cv2.boxFilter(gray, cv2.CV_16U, (3, 3), normalize=False,
                         borderType=cv2.BORDER_CONSTANT)
    smooth = ((sums + 4) // 9).astype(np.uint8)
    gx = cv2.Sobel(smooth, cv2.CV_16S, 1, 0, ksize=3, borderType=cv2.BORDER_CONSTANT)
    gy = cv2.Sobel(smooth, cv2.CV_16S, 0, 1, ksize=3, borderType=cv2.BORDER_CONSTANT)
    edges = np.minimum(np.abs(gx).astype(np.uint16) + np.abs(gy).astype(np.uint16), 255).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(changed.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    clean = cv2.dilate(eroded, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    summaries, raw = {}, {}
    for name in REGION_NAMES:
        x0, y0, x1, y1 = regions[name]
        region = np.s_[y0:y1, x0:x1]
        mask = changed[region]
        yy, xx = np.nonzero(mask)
        raw[name] = dict(changed_px=int(mask.sum()), persistent_px=int(persistent[region].sum()),
            delta_sum=int(delta[region].sum()), area_px=(x1-x0)*(y1-y0),
            bbox_inclusive_px=[int(xx.min())+x0, int(yy.min())+y0, int(xx.max())+x0,
                               int(yy.max())+y0] if len(xx) else [0]*4)
        summaries[name] = dict(gray_sum=int(gray[region].sum()),
            edge_px=int((edges[region] >= 64).sum()), clean_changed_px=int(clean[region].sum()),
            smooth_sum=int(smooth[region].sum()))
    return dict(contract=CONTRACT, regions=raw, processing=summaries,
        gray_histogram_16=np.bincount((gray >> 4).ravel(), minlength=16).tolist(),
        preview=np.stack((gray, smooth, edges, clean*255), axis=2) if preview_enabled else None,
        history=persistent, automatic_motion_allowed=False)


def result_words(result):
    words = []
    for name in REGION_NAMES:
        row = result['regions'][name]
        words.extend([row['changed_px'], row['persistent_px'], row['delta_sum'],
                      *row['bbox_inclusive_px'], row['area_px']])
    words.extend(result['gray_histogram_16'])
    for name in REGION_NAMES:
        row = result['processing'][name]
        words.extend(row[k] for k in ('gray_sum', 'edge_px', 'clean_changed_px', 'smooth_sum'))
    return np.asarray(words, dtype=np.uint32)


def unpack_results(words):
    a = np.asarray(words)
    if a.dtype != np.uint32 or a.shape != (RESULT_WORDS,):
        raise ValueError('expected_88_u32_frame_results')
    regions, processing = {}, {}
    for i, name in enumerate(REGION_NAMES):
        row = a[i*8:i*8+8]
        regions[name] = dict(changed_px=int(row[0]), persistent_px=int(row[1]), delta_sum=int(row[2]),
            bbox_inclusive_px=row[3:7].astype(int).tolist(), area_px=int(row[7]))
        row = a[64+i*4:68+i*4]
        processing[name] = dict(zip(('gray_sum', 'edge_px', 'clean_changed_px', 'smooth_sum'), map(int, row)))
    return dict(regions=regions, processing=processing, gray_histogram_16=a[48:64].astype(int).tolist())
