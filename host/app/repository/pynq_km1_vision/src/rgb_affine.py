"""Exact uint8 RGB/BGR affine transform via a small per-channel lookup table.

The caller fits and bounds gain/bias using the existing photometric policy.
This performs the same float64 multiply, add, clip and uint8 truncation as the
full-image reference, once for each possible channel value instead of pixel.
"""
import cv2
import numpy as np


def affine_lut(gain,bias):
    g,b=np.asarray(gain,dtype=np.float64),np.asarray(bias,dtype=np.float64)
    if g.shape!=(3,) or b.shape!=(3,) or not np.isfinite(g).all() or not np.isfinite(b).all():
        raise ValueError('invalid_rgb_affine_parameters')
    values=np.arange(256,dtype=np.float64)[:,None]
    return np.clip(values*g+b,0,255).astype(np.uint8).reshape(256,1,3)


def apply_rgb_affine_u8(image,gain,bias):
    if not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3 or not image.size:
        raise ValueError('invalid_rgb_affine_image')
    return cv2.LUT(image,affine_lut(gain,bias))
