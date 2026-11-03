"""Verified immutable deployment data, with live file identity checks.

Production snapshots must reside below root-owned, non-writable ancestors.
Writable journals, owner state and camera evidence are never snapshot entries.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import threading


class SnapshotFault(RuntimeError):
    pass


def metadata(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
            value.st_ctime_ns, value.st_uid, stat.S_IMODE(value.st_mode))


def require_sealed(value, *, directory=False):
    if value.st_uid != 0 or value.st_mode & 0o022:
        raise SnapshotFault('snapshot_not_root_owned_read_only')
    if not (stat.S_ISDIR(value.st_mode) if directory else stat.S_ISREG(value.st_mode)):
        raise SnapshotFault('snapshot_symlink_or_special_node')


class SealedSnapshot:
    def __init__(self, root, manifest='startup-static-manifest.json'):
        if os.name != 'posix':
            raise SnapshotFault('sealed_snapshot_requires_posix_permissions')
        self.root = Path(root).absolute()
        self.manifest = self.root / manifest
        self.faulted = False
        self.lock = threading.RLock()
        self.entries = {}
        self.ancestors = {}
        for directory in (self.root, *self.root.parents):
            value = directory.lstat()
            require_sealed(value, directory=True)
            self.ancestors[directory] = metadata(value)
        require_sealed(self.manifest.lstat())
        self.manifest_metadata = metadata(self.manifest.lstat())
        data = json.loads(self.manifest.read_text())
        if data.get('schema') != 'sealed-startup-inputs-v1' or not isinstance(data.get('files'), dict):
            raise SnapshotFault('invalid_static_snapshot_manifest')
        for name, digest in data['files'].items():
            path = self._path(name)
            self._parents(path.parent)
            value = path.lstat()
            require_sealed(value)
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise SnapshotFault('snapshot_initial_hash_changed:'+name)
            self.entries[path] = (digest, metadata(value))
        self.json_cache = {}
        self.check()

    def _path(self, name):
        relative = Path(name)
        if relative.is_absolute() or not relative.parts or any(p in ('.', '..') for p in relative.parts):
            raise SnapshotFault('snapshot_path_escapes_root')
        return self.root / relative

    def _parents(self, directory):
        while directory.is_relative_to(self.root):
            value = directory.lstat()
            require_sealed(value, directory=True)
            self.ancestors.setdefault(directory, metadata(value))
            if directory == self.root:
                break
            directory = directory.parent

    def _fail(self, reason):
        self.faulted = True
        raise SnapshotFault(reason)

    def check(self):
        with self.lock:
            if self.faulted:
                raise SnapshotFault('snapshot_fault_latched')
            try:
                # Directory ctime/mtime change when unrelated bytecode is
                # created, so the deployment disables bytecode writes and is
                # fully sealed before these identities are captured.
                for path, expected in self.ancestors.items():
                    value = path.lstat()
                    require_sealed(value, directory=True)
                    # Ancestors outside the snapshot may receive other root
                    # deployments. Their inode/permissions must remain stable.
                    actual = metadata(value)
                    indices = (0, 1, 5, 6) if path != self.root and not path.is_relative_to(self.root) else range(7)
                    if any(actual[i] != expected[i] for i in indices):
                        self._fail('snapshot_directory_identity_changed')
                if metadata(self.manifest.lstat()) != self.manifest_metadata:
                    self._fail('snapshot_manifest_identity_changed')
                for path, (_, expected) in self.entries.items():
                    if metadata(path.lstat()) != expected:
                        self._fail('snapshot_file_identity_changed:'+str(path.relative_to(self.root)))
            except (OSError, SnapshotFault):
                self.faulted = True
                raise

    def checked_path(self, path, expected=None):
        path = Path(os.path.abspath(path))
        if path not in self.entries:
            raise SnapshotFault('file_not_in_sealed_snapshot')
        digest, identity = self.entries[path]
        if self.faulted or metadata(path.lstat()) != identity or expected is not None and expected != digest:
            self._fail('sealed_file_or_expected_hash_changed')
        return path

    def read_json(self, path):
        with self.lock:
            path = self.checked_path(path)
            if path not in self.json_cache:
                self.json_cache[path] = json.loads(path.read_text(encoding='utf-8-sig'))
            return copy.deepcopy(self.json_cache[path])


class VerifiedBundle:
    """Clone a fully validated bundle; never cache mutable readiness state."""
    def __init__(self, snapshot, binding, binding_loader, unified_loader):
        self.snapshot, self.binding = snapshot, Path(binding).absolute()
        snapshot.checked_path(self.binding)
        self.bundle = binding_loader(self.binding)
        self.unified = unified_loader(self.binding)
        self.lock = threading.RLock()

    def load(self, path, *, unified=False):
        if Path(path).absolute() != self.binding:
            raise SnapshotFault('unreviewed_binding_for_cached_loader')
        self.snapshot.check()
        with self.lock:
            return copy.deepcopy(self.unified if unified else self.bundle)
