import copy
import unittest
from src.prompted_trial_scope import execution_scope, taxonomy_sha256


class TrialScopeTests(unittest.TestCase):
    def setUp(self):
        self.policy = dict(model={'sha256': 'current'}, classes=[
            {'official_label': 'Can', 'material': 'metal'},
            {'official_label': 'Carton', 'material': 'paper'},
            {'official_label': 'Bottle', 'material': 'plastic'}])
        self.receipt = dict(eligible_materials=['metal'], eligible_official_labels=['Can'])
        self.request = dict(kind='operator-supervised-trial-scope-v1',
            operator_statement='Open the current categories for supervised trials.',
            model_sha256='current', taxonomy_sha256=taxonomy_sha256(self.policy['classes']),
            official_labels=['Can', 'Carton', 'Bottle'], single_manual_attempt_only=True,
            physical_validation_completed=False)

    def scope(self, request=None):
        return execution_scope(self.receipt, self.policy,
            {'operator_trial_scope': self.request if request is None else request})

    def test_default_keeps_prior_scope(self):
        scope = execution_scope(self.receipt, self.policy, {})
        self.assertEqual(scope['official_labels'], ['Can'])
        self.assertEqual(scope['basis'], 'existing_validation_scope')

    def test_operator_trial_opens_all_materials_without_rewriting_validation(self):
        receipt = copy.deepcopy(self.receipt)
        scope = self.scope()
        self.assertEqual(scope['materials'], ['metal', 'paper', 'plastic'])
        self.assertEqual(scope['official_labels'], ['Can', 'Carton', 'Bottle'])
        self.assertEqual(scope['validated_scope']['official_labels'], ['Can'])
        self.assertEqual(self.receipt, receipt)
        self.assertFalse(scope['physical_validation_completed'])

    def test_scope_cannot_follow_model_or_material_mapping_change(self):
        for key in ('model_sha256', 'taxonomy_sha256'):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'identity_or_basis'):
                self.scope(dict(self.request, **{key: 'changed'}))

    def test_blank_authority_or_qualification_or_continuous_mode_is_rejected(self):
        for change in ({'operator_statement': ''}, {'physical_validation_completed': True},
                       {'single_manual_attempt_only': False}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.scope(dict(self.request, **change))

    def test_unknown_duplicate_or_empty_labels_are_rejected(self):
        for labels in ([], ['Unknown'], ['Can', 'Can'], [None]):
            with self.subTest(labels=labels), self.assertRaisesRegex(ValueError, 'labels'):
                self.scope(dict(self.request, official_labels=labels))

    def test_unmapped_material_is_rejected(self):
        self.policy['classes'][0]['material'] = None
        self.request['taxonomy_sha256'] = taxonomy_sha256(self.policy['classes'])
        with self.assertRaisesRegex(ValueError, 'labels'):
            self.scope()


if __name__ == '__main__':
    unittest.main()
