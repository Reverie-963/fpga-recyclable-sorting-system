"""One scoped reuse of already verified fixed markers for one exact frame."""
import copy
import hashlib
import json
import threading
from contextlib import contextmanager


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def construct_with_initial_markers(factory, args, kwargs, detector):
    if 'marker_detector' in kwargs:
        raise RuntimeError('conflicting_marker_detector_composition')
    result = factory(*args, marker_detector=detector, **kwargs)
    # The pinned upstream helper creates an ExternalMarkerGeometry subclass.
    # Only construction uses the frozen markers; future checks use its parent.
    result.check = super(type(result), result).check
    return result


def track_camera_captures(camera_type, authority):
    """Track the real capture shared by raw() and detector-driven next()."""
    original = camera_type._raw
    def capture(self, deadline):
        authority.check()
        row = original(self, deadline)
        # Preserve the returned capture timestamp. Never replace it with now.
        authority.last_capture_ms = row['observation']['capture_started_ms']
        return row
    camera_type._raw = capture


class CaptureMarkerScope:
    """Reuse markers from this acquisition's verified detector reply only."""
    def __init__(self, camera, expected, profile, camera_hash, fallback):
        self.camera,self.expected,self.profile=camera,expected,profile
        self.camera_hash,self.fallback=camera_hash,fallback
        self.local=threading.local()

    @contextmanager
    def acquisition(self):
        if getattr(self.local,'active',False):raise RuntimeError('nested_capture_marker_scope')
        self.local.active=True;self.local.markers=None
        try:yield
        finally:self.local.active=False;self.local.markers=None

    def observed(self,row):
        if getattr(self.local,'active',False):
            if self.local.markers is not None:raise RuntimeError('unconsumed_capture_markers')
            self.local.markers=FrameMarkers(row,self.camera,self.expected(),self.profile(),self.camera_hash)

    def detect(self,image,camera):
        markers=getattr(self.local,'markers',None) if getattr(self.local,'active',False) else None
        if markers is None:return self.fallback(image,camera)
        self.local.markers=None
        return markers.detector(image,camera,self.expected(),self.profile())


class FrameMarkers:
    def __init__(self, row, camera, visual_expected, profile, camera_file_sha256):
        image, observation = row['image'], row['observation']
        markers = observation['fixed_marker_observation']
        bgr_hash = hashlib.sha256(image.tobytes()).hexdigest()
        if (markers.get('source_bgr_sha256') != bgr_hash or markers.get('camera_sha256') != camera_file_sha256
                or markers.get('location') != 'pc_vision' or not isinstance(markers.get('detections'), list)
                or not isinstance(observation.get('source_stream_id'), str)
                or type(observation.get('source_frame_id')) not in (int, str)
                or visual_expected is None):
            raise ValueError('frame_marker_evidence_not_bound')
        self.image_hash, self.camera_hash = bgr_hash, digest(camera)
        self.stream, self.frame = observation['source_stream_id'], observation['source_frame_id']
        self.expected, self.profile = digest(visual_expected), digest(profile)
        self.markers = copy.deepcopy(markers['detections'])
        self.thread = threading.get_ident()
        self.used = False

    def detector(self, image, camera, visual_expected, profile):
        if (threading.get_ident() != self.thread or self.used or
                hashlib.sha256(image.tobytes()).hexdigest() != self.image_hash or
                digest(camera) != self.camera_hash or digest(visual_expected) != self.expected or
                digest(profile) != self.profile):
            raise ValueError('scoped_frame_marker_identity_changed')
        self.used = True
        return copy.deepcopy(self.markers)
