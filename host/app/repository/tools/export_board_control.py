"""Export an execution-disabled, portable preflight bundle, preserving evidence.

Evidence paths are resolved using the consuming function's base directory.
Historical reports remain byte-for-byte evidence leaves, not executable state.
This is not an execution/cutover exporter and never enables experiment flags.
"""
import argparse
import hashlib
import json
from pathlib import Path

from stage_board_control import ROOT, PROJECT, dependency_files, sha


class Exporter:
    def __init__(self, output):
        self.output = Path(output).resolve()
        if not self.output.is_relative_to(ROOT/'local-only'):
            raise ValueError('local_only_output_required')
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output/'objects').mkdir()
        (self.output/'state').mkdir()
        self.originals, self.outputs = {}, {}

    def read(self, base, record):
        path = (Path(base)/record['path']).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or sha(path) != record['sha256']:
            raise ValueError('export_evidence_missing_or_changed:'+path.name)
        self.originals[str(path.relative_to(ROOT))] = record['sha256']
        return path

    def write(self, data, suffix):
        digest = hashlib.sha256(data).hexdigest()
        name = digest+suffix.lower()
        path = self.output/'objects'/name
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError('export_object_collision')
        else:
            path.write_bytes(data)
        self.outputs['objects/'+name] = digest
        return dict(path=name, sha256=digest)

    def source(self, base, record):
        path = self.read(base, record)
        relative = path.relative_to(PROJECT)
        if relative.parts[0] not in ('src', 'openmv_gripper') or path.suffix != '.py':
            raise ValueError('unexpected_bound_implementation_source')
        target = self.output/'pynq_km1_vision'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        destination = target.relative_to(self.output).as_posix()
        self.outputs[destination] = sha(target)
        return dict(record, path='../'+destination)

    def leaf(self, base, record):
        path = self.read(base, record)
        if path.suffix.lower() not in ('.json', '.png', '.jpg', '.jpeg'):
            raise ValueError('unexpected_evidence_file_type')
        return dict(record, **self.write(path.read_bytes(), path.suffix))

    def json_object(self, value):
        return self.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode('utf-8'), '.json')

    def qualification(self, base, record):
        path = self.read(base, record)
        value = json.loads(path.read_text(encoding='utf-8-sig'))
        # known_readiness resolves these against CONFIG base, not q's parent.
        value['evidence'] = [self.leaf(base, r) for r in value['evidence']]
        return dict(record, **self.json_object(value))

    def config(self, base, record):
        path = self.read(base, record)
        value = json.loads(path.read_text(encoding='utf-8-sig'))
        base = path.parent
        for profile in value['profiles'].values():
            profile['qualification_record'] = self.qualification(base, profile['qualification_record'])
        for key in ('layout_reference', 'empty_reference'):
            if key in value:
                value[key] = self.leaf(base, value[key])
        known = value['known']
        known['implementation_sources'] = [self.source(base, r) for r in known.get('implementation_sources', [])]
        if 'vision_binding' in known:
            binding = self.read(base, known['vision_binding'])
            vision = json.loads(binding.read_text(encoding='utf-8-sig'))
            vision['frames'] = [self.leaf(base, frame) for frame in vision['frames']]
            known['vision_binding'] = dict(known['vision_binding'], **self.json_object(vision))
        known.update(experiment_authorized=False, device_lease='../state/device.lock', stop_files=['../state/STOP'])
        value.update(journal_dir='../state/ledger', detector_url='http://127.0.0.1:1',
                     camera_url='http://127.0.0.1:1/frame.jpg', migration_execution_disabled=True)
        value['statistics']['url'] = 'http://127.0.0.1:1'
        return dict(record, **self.json_object(value))

    def special(self, base, record, role):
        path = self.read(base, record)
        value = json.loads(path.read_text(encoding='utf-8-sig'))
        if role == 'unified_policy':
            for slots in value['bin_slots'].values():
                for slot in slots:
                    if slot['origin'] == 'explicit_slot_record':
                        slot['motion_record'] = self.leaf(path.parent, slot['motion_record'])
        elif role.startswith('empty_record_'):
            value['frames'] = [dict(frame, image=self.leaf(path.parent, frame['image'])) for frame in value['frames']]
        else:
            raise ValueError('unknown_export_role')
        return dict(record, **self.json_object(value))

    def export(self, pointer):
        pointer = Path(pointer).resolve()
        pointer_sha = sha(pointer)
        active = json.loads(pointer.read_text(encoding='utf-8-sig'))
        path = self.read(pointer.parent, active)
        value = json.loads(path.read_text(encoding='utf-8-sig'))
        for role, record in list(value['files'].items()):
            if role == 'config':
                target = self.config(path.parent, record)
            elif role == 'unified_policy' or role.startswith('empty_record_'):
                target = self.special(path.parent, record, role)
            else:
                target = self.leaf(path.parent, record)
            value['files'][role] = target
        value['implementation_sources'] = [self.source(path.parent, r) for r in value['implementation_sources']]
        entry = self.json_object(value)
        # Add the import dependency closure. Bound sources must match these bytes.
        for name, source in dependency_files('src.sorting_unified_run').items():
            destination = 'pynq_km1_vision/'+name
            digest = sha(source)
            if destination in self.outputs and self.outputs[destination] != digest:
                raise ValueError('loaded_and_bound_source_mismatch')
            target = self.output/destination
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            self.outputs[destination] = digest
        if sha(pointer) != pointer_sha or any(sha(ROOT/name) != digest for name,digest in self.originals.items()):
            raise ValueError('export_sources_changed_during_snapshot')
        manifest = dict(kind='board-preflight-bundle-v1', status='exported_execution_disabled',
            source_pointer_sha256=pointer_sha, binding=dict(entry,path='objects/'+entry['path']),
            original_sha256=self.originals, output_sha256=self.outputs, execution_enabled=False,
            physical_motion=False, historical_report_contents_preserved=True,
            caveat='Historical nested report paths are evidence only; this bundle cannot execute.')
        (self.output/'bundle.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = Exporter(args.output).export(ROOT/'local-only/station-grasp-integration/active.json')
    print(json.dumps(dict(status=result['status'], files=len(result['output_sha256']),
                         binding=result['binding']['path'], execution_enabled=False)))


if __name__ == '__main__':
    main()
