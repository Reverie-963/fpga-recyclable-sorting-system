"""Array-only empty-background reprojection for the board startup path.

Same pixel model, full-frame depth checks, interpolation and coverage gates as
the legacy warp. Nonzero floor projections only need their destination ROIs.
No camera, model, PL, serial or state access on import.
"""
import time
import ctypes
import hashlib
from pathlib import Path
import cv2
import numpy as np

BACKEND = 'array_roi_empty_reference_v1'
PROJECT_BATCH_POINTS = 1024
_native_project = None
_native_library_sha256 = None


def load_native_projector(path, expected_sha256):
    global _native_project, _native_library_sha256
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError('empty_reference_native_library_hash_changed')
    library = ctypes.CDLL(str(path.resolve()))
    method = library.empty_reference_project
    pointer = np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    method.argtypes = [ctypes.c_uint,pointer,pointer,pointer,pointer,pointer,ctypes.c_uint,pointer]
    method.restype = ctypes.c_int
    _native_project = method
    _native_library_sha256 = expected_sha256


def need(ok, message):
    if not ok:
        raise ValueError(message)


def _checked_points(points, parameters):
    _, _, rotation, translation = parameters
    points = np.asarray(points, dtype=float)
    need(points.ndim == 2 and points.shape[1] == 3 and np.isfinite(points).all(), 'invalid_world_points')
    # Retain the old full-frame rejection, even when only a small ROI is used.
    need(((rotation @ points.T).T+translation)[:, 2].min() > 0, 'world_points_behind_camera')
    return points


def _project(points, parameters):
    intrinsic, distortion, rotation, translation = parameters
    rvec = cv2.Rodrigues(rotation)[0]
    if _native_project is not None and np.asarray(distortion).size in (4,5,8,12):
        # Match projectPoints' rotation-vector round trip before projection.
        camera_rotation = np.ascontiguousarray(cv2.Rodrigues(rvec)[0],dtype=np.float64)
        result = np.empty((len(points),2),dtype=np.float64)
        code = _native_project(len(points),np.ascontiguousarray(points,dtype=np.float64),camera_rotation,
            np.ascontiguousarray(translation,dtype=np.float64),np.ascontiguousarray(intrinsic,dtype=np.float64),
            np.ascontiguousarray(distortion,dtype=np.float64),np.asarray(distortion).size,result)
        if code != 0:
            raise RuntimeError('empty_reference_native_project_rejected')
        return result
    # Python projectPoints also materializes a 2N x (10+D) Jacobian. Bound
    # that temporary to a small block instead of allocating tens of MB on PS.
    result = np.empty((len(points), 2), dtype=np.float64)
    for start in range(0, len(points), PROJECT_BATCH_POINTS):
        block = np.ascontiguousarray(points[start:start+PROJECT_BATCH_POINTS])
        result[start:start+len(block)] = cv2.projectPoints(block, rvec,
            translation, intrinsic, distortion)[0].reshape(-1, 2)
    return result


def warp_reference(image, size, scene_regions, profiles, current_parameters,
                   reference_parameters, roi_only=True):
    """Return the old warped background without Python point lists.

    Both parameter tuples must come from the existing validated pose loader.
    roi_only=False is a full-frame array baseline for equivalence/performance.
    """
    width, height = size
    yy, xx = np.indices((height, width))
    k, d, r, t = current_parameters
    pixels = np.c_[xx.ravel(), yy.ravel()].astype(float)
    rays = np.c_[cv2.undistortPoints(pixels.reshape(-1, 1, 2), k, d).reshape(-1, 2), np.ones(len(pixels))]
    directions = (r.T @ rays.T).T
    center = -r.T @ t
    result = np.zeros_like(image)
    heights = {0.}
    heights.update(float(v['height_mm']) for v in scene_regions.values())
    projected_points = 0
    for z in heights:
        xyz = center+((z-center[2])/directions[:, 2])[:, None]*directions
        xyz = _checked_points(xyz, reference_parameters)
        if z == 0 or not roi_only:
            uv = np.asarray(_project(xyz, reference_parameters), np.float32).reshape(height, width, 2)
            projected_points += len(xyz)
            remapped = cv2.remap(image, uv[:, :, 0], uv[:, :, 1], cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT)
            valid = (uv[:, :, 0] >= 0) & (uv[:, :, 0] < width-1) & (uv[:, :, 1] >= 0) & (uv[:, :, 1] < height-1)
            if z == 0:
                result = remapped
            for name, data in scene_regions.items():
                if data['height_mm'] != z or name == 'source':
                    continue
                label = next(k for k, p in profiles.items() if p['bin_id'] == name)
                x0, y0, x1, y1 = profiles[label]['bin_roi']
                need(valid[y0:y1, x0:x1].all(), 'empty_reference_new_view_not_covered')
                result[y0:y1, x0:x1] = remapped[y0:y1, x0:x1]
        else:
            grid = xyz.reshape(height, width, 3)
            for name, data in scene_regions.items():
                if data['height_mm'] != z or name == 'source':
                    continue
                label = next(k for k, p in profiles.items() if p['bin_id'] == name)
                x0, y0, x1, y1 = profiles[label]['bin_roi']
                points = grid[y0:y1, x0:x1].reshape(-1, 3)
                uv = np.asarray(_project(points, reference_parameters), np.float32).reshape(y1-y0, x1-x0, 2)
                projected_points += len(points)
                valid = (uv[:, :, 0] >= 0) & (uv[:, :, 0] < width-1) & (uv[:, :, 1] >= 0) & (uv[:, :, 1] < height-1)
                need(valid.all(), 'empty_reference_new_view_not_covered')
                result[y0:y1, x0:x1] = cv2.remap(image, uv[:, :, 0], uv[:, :, 1], cv2.INTER_LINEAR,
                                                borderMode=cv2.BORDER_CONSTANT)
    return result, dict(backend=BACKEND if roi_only else 'array_dense_empty_reference_v1',
                        validated_world_points=len(pixels)*len(heights), projected_points=projected_points,
                        floor_count=len(heights), roi_only=roi_only,
                        projector_backend='cpu_native_forward_projection' if _native_project is not None and
                            np.asarray(reference_parameters[1]).size in (4,5,8,12) else 'opencv_chunked_projection',
                        native_library_sha256=_native_library_sha256)


def warp_empty_reference(geometry, image):
    from .station_geometry import pose_parameters
    need(geometry.scene.get('empty_bins_confirmed') is True, 'empty_bins_reference_not_confirmed')
    started = time.perf_counter_ns()
    current = pose_parameters(geometry.pose, geometry.camera, geometry.station)
    reference = pose_parameters(geometry.scene['reference_pose'], geometry.camera, geometry.station)
    result, timing = warp_reference(image, geometry.runtime['image_size'], geometry.scene['scene_regions'],
                                    geometry.runtime['profiles'], current, reference)
    timing['elapsed_ms'] = (time.perf_counter_ns()-started)/1e6
    geometry.last_empty_reference_timing = timing
    return result


def install(expected_legacy_sha256, native_library=None, native_sha256=None):
    """Explicit composition hook; the added module must be in the source manifest.

    Does not edit original files, qualification records, numeric parameters,
    motion thresholds or scene evidence. Call once before the board host starts.
    """
    from .sorting_relocated_geometry import RelocatedGeometry
    original = RelocatedGeometry.warp_empty_reference
    if original is warp_empty_reference:
        raise RuntimeError('empty_reference_backend_already_installed')
    if hashlib.sha256(Path(original.__code__.co_filename).read_bytes()).hexdigest() != expected_legacy_sha256:
        raise ValueError('empty_reference_legacy_source_changed')
    if native_library is not None:
        load_native_projector(native_library,native_sha256)
    RelocatedGeometry.warp_empty_reference = warp_empty_reference
    return original
