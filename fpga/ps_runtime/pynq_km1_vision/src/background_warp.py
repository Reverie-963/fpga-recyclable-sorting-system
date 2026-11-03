"""Array-only background reprojection with immutable normalized ray tables.

Keeps full-frame positive-depth and destination coverage rejection. No camera,
model, FPGA or actuator access. No map/pose/baseline caching across frames.
"""
import hashlib
import json
import threading
import cv2
import numpy as np


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


class RayTable:
    def __init__(self):
        self.key, self.rays = None, None
        self.lock = threading.RLock()

    def get(self, size, intrinsic, distortion):
        width, height = size
        k, d = np.asarray(intrinsic, dtype=np.float64), np.asarray(distortion, dtype=np.float64)
        need(type(width) is int and type(height) is int and 1 <= width <= 640 and 1 <= height <= 480,
             'unsupported_background_size')
        need(k.shape == (3, 3) and d.size in (4, 5, 8, 12, 14) and np.isfinite(k).all() and np.isfinite(d).all()
             and k[0, 0] > 0 and k[1, 1] > 0, 'invalid_background_camera')
        key = hashlib.sha256(json.dumps([width, height, cv2.__version__, 'undistortPoints_default_5_iterations',
                                        k.tolist(), d.tolist()], sort_keys=True).encode()).hexdigest()
        with self.lock:
            if key != self.key:
                yy, xx = np.indices((height, width))
                pixels = np.c_[xx.ravel(), yy.ravel()].astype(float)
                rays = np.c_[cv2.undistortPoints(pixels.reshape(-1, 1, 2), k, d).reshape(-1, 2), np.ones(len(pixels))]
                need(np.isfinite(rays).all(), 'invalid_background_rays')
                rays.flags.writeable = False
                self.key, self.rays = key, rays
            return self.rays


def checked_points(points, parameters):
    _, _, rotation, translation = parameters
    points = np.asarray(points, dtype=float)
    need(points.ndim == 2 and points.shape[1] == 3 and np.isfinite(points).all(), 'invalid_world_points')
    need(((rotation @ points.T).T+translation)[:, 2].min() > 0, 'world_points_behind_camera')
    return points


def project(points, parameters):
    intrinsic, distortion, rotation, translation = parameters
    result = np.empty((len(points), 2), dtype=np.float64)
    rvec = cv2.Rodrigues(rotation)[0]
    for start in range(0, len(points), 1024):
        block = np.ascontiguousarray(points[start:start+1024])
        result[start:start+len(block)] = cv2.projectPoints(block, rvec, translation, intrinsic, distortion)[0].reshape(-1, 2)
    return result


def warp_reference(image, size, regions, profiles, current, reference, ray_table, *, projector=project):
    width, height = size
    need(isinstance(image, np.ndarray) and image.dtype == np.uint8 and image.shape == (height, width, 3),
         'invalid_background_image')
    k, d, rotation, translation = current
    rays = ray_table.get(size, k, d)
    directions = (rotation.T @ rays.T).T
    center = -rotation.T @ translation
    result = np.zeros_like(image)
    heights = {0.}
    heights.update(float(v['height_mm']) for v in regions.values())
    need(len(heights) <= 8 and all(np.isfinite(z) for z in heights), 'invalid_background_heights')
    for z in heights:
        xyz = center+((z-center[2])/directions[:, 2])[:, None]*directions
        xyz = checked_points(xyz, reference)
        if z == 0:
            uv = np.asarray(projector(xyz, reference), np.float32).reshape(height, width, 2)
            remapped = cv2.remap(image, uv[:, :, 0], uv[:, :, 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            valid = (uv[:, :, 0] >= 0) & (uv[:, :, 0] < width-1) & (uv[:, :, 1] >= 0) & (uv[:, :, 1] < height-1)
            result = remapped
            for name, data in regions.items():
                if data['height_mm'] != z or name == 'source':
                    continue
                label = next(k for k, p in profiles.items() if p['bin_id'] == name)
                x0, y0, x1, y1 = profiles[label]['bin_roi']
                need(valid[y0:y1, x0:x1].all(), 'empty_reference_new_view_not_covered')
                result[y0:y1, x0:x1] = remapped[y0:y1, x0:x1]
        else:
            grid = xyz.reshape(height, width, 3)
            for name, data in regions.items():
                if data['height_mm'] != z or name == 'source':
                    continue
                label = next(k for k, p in profiles.items() if p['bin_id'] == name)
                x0, y0, x1, y1 = profiles[label]['bin_roi']
                uv = np.asarray(projector(grid[y0:y1, x0:x1].reshape(-1, 3), reference), np.float32).reshape(y1-y0, x1-x0, 2)
                valid = (uv[:, :, 0] >= 0) & (uv[:, :, 0] < width-1) & (uv[:, :, 1] >= 0) & (uv[:, :, 1] < height-1)
                need(valid.all(), 'empty_reference_new_view_not_covered')
                result[y0:y1, x0:x1] = cv2.remap(image, uv[:, :, 0], uv[:, :, 1], cv2.INTER_LINEAR,
                                                borderMode=cv2.BORDER_CONSTANT)
    return result
