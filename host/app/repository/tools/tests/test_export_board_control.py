import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).parents[1]
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location('export_board_control', TOOLS/'export_board_control.py')
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root/'config'
        self.base.mkdir()
        self.patch = patch.object(EXPORT, 'ROOT', self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.exporter = EXPORT.Exporter(self.root/'local-only/candidate')

    def save(self, relative, value):
        p = self.base/relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value))
        return dict(path=relative, sha256=EXPORT.sha(p))

    def exported_json(self, record):
        return json.loads((self.exporter.output/'objects'/record['path']).read_text())

    def test_qualification_refs_use_config_base_and_preserve_historical_bytes(self):
        leaf = self.save('evidence/report.json', dict(historical_nested_path='old-machine-only', passed=False))
        qualification = self.save('q/record.json', dict(evidence=[leaf], operator_confirmed=True))
        original = (self.base/'q/record.json').read_bytes()
        result = self.exported_json(self.exporter.qualification(self.base, qualification))
        copy = result['evidence'][0]
        self.assertEqual((self.exporter.output/'objects'/copy['path']).read_bytes(),
                         (self.base/'evidence/report.json').read_bytes())
        self.assertEqual((self.base/'q/record.json').read_bytes(), original)

    def test_per_bin_empty_frame_image_refs_use_record_directory(self):
        image = self.save('bin/frame.json', dict(synthetic_image=True))
        record = self.save('bin/record.json', dict(frames=[dict(source_frame_id='1',
            image=dict(image,path='frame.json'))]))
        value = self.exported_json(self.exporter.special(self.base, record, 'empty_record_metal'))
        self.assertEqual(value['frames'][0]['source_frame_id'], '1')
        self.assertEqual(value['frames'][0]['image']['sha256'], image['sha256'])

    def test_config_downgrades_execution_and_isolates_mutable_state(self):
        image = self.save('image.json', dict(synthetic_image=True))
        q = self.save('q/record.json', dict(evidence=[image]))
        config = self.save('control.json', dict(profiles=dict(paper=dict(qualification_record=q, grasp_z_mm=50)),
            known=dict(experiment_authorized=True, implementation_sources=[]), statistics=dict(url='http://localhost:8780'),
            layout_reference=image, empty_reference=image, arm=dict(port='COM8')))
        original = (self.base/'control.json').read_bytes()
        value = self.exported_json(self.exporter.config(self.base, config))
        self.assertFalse(value['known']['experiment_authorized'])
        self.assertTrue(value['migration_execution_disabled'])
        self.assertEqual(value['known']['device_lease'], '../state/device.lock')
        self.assertEqual(value['arm']['port'], 'COM8')
        self.assertEqual(value['profiles']['paper']['grasp_z_mm'], 50)
        self.assertEqual((self.base/'control.json').read_bytes(), original)

    def test_changed_evidence_or_outside_repository_rejected(self):
        record = self.save('evidence.json', dict(value=1))
        (self.base/'evidence.json').write_text('{}')
        with self.assertRaises(ValueError): self.exporter.leaf(self.base, record)
        with self.assertRaises(ValueError):
            self.exporter.read(self.base, dict(path='../../outside', sha256='0'*64))


if __name__ == '__main__':
    unittest.main()
