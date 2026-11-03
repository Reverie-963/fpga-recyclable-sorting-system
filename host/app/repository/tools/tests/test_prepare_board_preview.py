import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1]))
import prepare_board_preview as prepare


class OfflinePackageTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.project = self.root/'pynq_km1_vision'
        self.output = self.root/'local-only/candidate'
        self.output.mkdir(parents=True)
        self.pointer = self.save('local-only/station-grasp-integration/active.json', '{}')
        self.policy = {'stable_frames': 1, 'frame_max_age_ms': 6000, 'frame_timeout_s': 6,
                       'projected_region_bounds_check_enabled': False}
        self.save('pynq_km1_vision/config/prompted_motion_policy.json', json.dumps(self.policy))
        self.save('pynq_km1_vision/config/yoloe_taco_recyclables.json', '{"threshold":0.60}')
        self.save('local-only/yoloe-recyclables/calibration-20261030/deployment.json', '{"labels":["Drink can"]}')
        self.worker = self.save('pynq_km1_vision/src/demo.py', '# worker\n')
        for name in ('prepare_board_preview.py', 'serve_board_prompted.py',
                     'board_control_gateway.py', 'activate_board_console.py'):
            self.save('tools/'+name, '# host\n')
        for name in ('board_serial_inventory.py', 'sorting_prompted_motion.py', 'board_prompted_console.py',
                     'board_control_gateway_client.py', 'sorting_prompted_console.py',
                     'monitor_prompted_operation.py'):
            self.save('pynq_km1_vision/src/'+name, '# current height and timing implementation\n')
        # Production state and secrets must remain outside a candidate.
        self.save('local-only/ledger/pending.json', '{"status":"needs_review"}')
        self.save('local-only/secret-token.txt', 'private')
        outer = self

        class FakeExporter:
            def __init__(self, output):
                self.output = output

            def export(self, pointer):
                self.output.mkdir()
                file = self.output/'config.json'
                file.write_text('{"execution_enabled":false}')
                return dict(output_sha256={'config.json': prepare.sha(file)},
                            original_sha256={}, source_pointer_sha256=prepare.sha(pointer),
                            binding={'path': 'config.json'})

        for name, value in (('ROOT', self.root), ('PROJECT', self.project),
                            ('Exporter', FakeExporter),
                            ('dependency_files', lambda _: {'src/demo.py': outer.worker})):
            p = patch.object(prepare, name, value)
            p.start()
            self.addCleanup(p.stop)

    def save(self, name, text):
        p = self.root/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding='utf-8')
        return p

    def build(self):
        # Any accidental network connection is an immediate test failure.
        with patch('socket.socket', side_effect=AssertionError('network forbidden')):
            return prepare.freeze_preview_inputs(self.output, self.pointer)

    def test_current_conditions_and_height_code_preserved_without_live_state(self):
        result = self.build()
        manifest = prepare.verify_preview_inputs(self.output)
        self.assertEqual(manifest['motion_options'], self.policy)
        self.assertFalse(manifest['cutover_ready'])
        self.assertFalse(manifest['live_settings_verified'])
        self.assertFalse(manifest['execution_enabled'])
        self.assertEqual(result['files']['host-sources/sorting_prompted_motion.py'].read_bytes(),
                         (self.project/'src/sorting_prompted_motion.py').read_bytes())
        self.assertNotIn('pending.json', ''.join(manifest['output_sha256']))
        self.assertNotIn('token.txt', ''.join(manifest['output_sha256']))

    def test_source_changes_require_fresh_snapshot_preserving_frozen_copy(self):
        result = self.build()
        frozen = result['files']['pynq_km1_vision/src/demo.py'].read_bytes()
        self.worker.write_text('# new implementation\n')
        with self.assertRaisesRegex(ValueError, 'offline_source_changed'):
            prepare.verify_preview_inputs(self.output)
        self.assertEqual(result['files']['pynq_km1_vision/src/demo.py'].read_bytes(), frozen)

    def test_tampered_candidate_is_rejected(self):
        result = self.build()
        result['files']['prompted-policy.json'].write_text('{}')
        with self.assertRaisesRegex(ValueError, 'offline_package_changed'):
            prepare.verify_preview_inputs(self.output)

    def test_changed_binding_pointer_is_rejected(self):
        self.build()
        self.pointer.write_text('{"path":"other-binding"}')
        with self.assertRaisesRegex(ValueError, 'offline_source_changed'):
            prepare.verify_preview_inputs(self.output)

    def test_explicit_current_model_policy_and_scope_do_not_replace_default_sources(self):
        policy=self.save('local-only/new26/policy.json','{"model":"fixed26","confidence":0.30}')
        scope=self.save('local-only/new26/scope.json','{"eligible_materials":["metal"]}')
        old=(self.project/'config/yoloe_taco_recyclables.json').read_bytes()
        result=prepare.freeze_preview_inputs(self.output,self.pointer,policy,scope)
        self.assertEqual(result['files']['prompted-policy.json'].read_bytes(),policy.read_bytes())
        self.assertEqual(result['files']['prompted-trial-scope.json'].read_bytes(),scope.read_bytes())
        self.assertEqual((self.project/'config/yoloe_taco_recyclables.json').read_bytes(),old)

    def test_manifest_cannot_grant_execution_or_escape_package(self):
        self.build()
        path = self.output/'offline-package.json'
        manifest = json.loads(path.read_text())
        manifest['execution_enabled'] = True
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'invalid_offline_package_contract'):
            prepare.verify_preview_inputs(self.output)
        manifest['execution_enabled'] = False
        manifest['output_sha256']['../../outside'] = '0'*64
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'offline_package_path_escape'):
            prepare.verify_preview_inputs(self.output)


if __name__ == '__main__':
    unittest.main()
