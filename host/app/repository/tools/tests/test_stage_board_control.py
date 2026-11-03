"""Check that the migration package cannot silently omit local dependencies."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('stage_board_control', Path(__file__).parents[1]/'stage_board_control.py')
STAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STAGE)


class StageBoardTests(unittest.TestCase):
    def test_transitive_control_dependencies_without_models_or_machine_settings(self):
        files = STAGE.dependency_files()
        self.assertTrue({'src/board_control_shadow.py', 'src/sorting_unified_plan.py',
                         'src/sorting_unified_run.py', 'src/monitor_operation.py',
                         'src/standby_offline.py', 'src/arm_transport.py',
                         'openmv_gripper/km1_arm_bridge.py'}.issubset(files))
        self.assertTrue(all(p.is_file() and p.suffix == '.py' and
                            p.is_relative_to(STAGE.PROJECT) for p in files.values()))
        self.assertFalse(any(n.startswith(('local-only/', 'Ultralytics/', 'sim/')) for n in files))
        self.assertFalse(any('preview_server' in n for n in files))

    def test_no_parent_escape_in_package(self):
        self.assertTrue(all('..' not in Path(n).parts and not Path(n).is_absolute()
                            for n in STAGE.dependency_files()))

    def test_production_ownership_or_pending_defers_board_checks(self):
        sys.path.insert(0, str(STAGE.PROJECT))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root/'local-only/station-grasp-integration'
            base.mkdir(parents=True)
            config = base/'config.json'
            config.write_text('{}')
            binding = base/'binding.json'
            binding.write_text(json.dumps(dict(files=dict(config=dict(path='config.json', sha256=STAGE.sha(config))))))
            pointer = base/'active.json'
            pointer.write_text(json.dumps(dict(path='binding.json', sha256=STAGE.sha(binding))))
            fixture = dict(source=dict(pointer_sha256=STAGE.sha(pointer)),
                           config=dict(journal_dir='ledger', known=dict(device_lease='device.lock')))
            with patch.object(STAGE, 'ROOT', root):
                STAGE.require_idle(fixture)
                for name in ('device.lock', 'ledger/controller.lock', 'ledger/pending.json', 'ledger/batch-pending.json'):
                    with self.subTest(name=name):
                        protected = base/name
                        protected.parent.mkdir(exist_ok=True)
                        protected.write_text('preserve this owner or evidence')
                        with self.assertRaisesRegex(ValueError, 'station_busy_or_unresolved_defer'):
                            STAGE.require_idle(fixture)
                        self.assertTrue(protected.exists())
                        protected.unlink()
                pointer.write_text('{}')
                with self.assertRaisesRegex(ValueError, 'active_binding_changed_defer'):
                    STAGE.require_idle(fixture)


if __name__ == '__main__':
    unittest.main()
