"""Exact binary moments on the existing OpenCV CPU; no device or PL claims."""
import cv2
import numpy as np


def image_stats_u8(mask, job):
    if (not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.ndim != 2 or
            not 1 <= mask.shape[0] <= 480 or not 1 <= mask.shape[1] <= 640):
        raise ValueError('binary_moments_require_bounded_bool_mask')
    # bool and uint8 have the same one-byte representation. Do not materialize
    # x/y foreground vectors or square them. Strided callers get an owned copy.
    image = np.ascontiguousarray(mask.view(np.uint8))
    moments = cv2.moments(image, binaryImage=True)
    result = {key: int(round(moments[key])) for key in ('m00', 'm10', 'm01', 'm20', 'm02', 'm11')}
    # At 640x480 each integer sum is far below 2**53. Inclusive maxima match the
    # legacy statistics ABI, rather than the exclusive detector bbox contract.
    x, y, width, height = cv2.boundingRect(image)
    result.update(min_col=x if width else 0, min_row=y if height else 0,
                  max_col=x+width-1 if width else 0, max_row=y+height-1 if height else 0,
                  job_id=job, status=1)
    return result
