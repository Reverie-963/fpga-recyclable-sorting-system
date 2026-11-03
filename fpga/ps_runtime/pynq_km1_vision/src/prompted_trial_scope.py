"""Explicit operator trials are distinct from historical validation evidence."""
import hashlib
import json


def taxonomy_sha256(classes):
    return hashlib.sha256(json.dumps(
        [(row['official_label'], row['material']) for row in classes],
        ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def execution_scope(receipt, policy, options):
    validated = dict(materials=list(receipt['eligible_materials']),
                     official_labels=list(receipt['eligible_official_labels']))
    from .execution_model_context import current_profile
    active=current_profile()
    requested = active['scope'] if active is not None else options.get('operator_trial_scope')
    if requested is None:
        return dict(**validated, basis='existing_validation_scope', validated_scope=validated,
                    physical_validation_completed=False)
    if (requested.get('kind') != 'operator-supervised-trial-scope-v1' or
            not isinstance(requested.get('operator_statement'), str) or not requested['operator_statement'].strip() or
            requested.get('physical_validation_completed') is not False or
            requested.get('single_manual_attempt_only') is not True or
            requested.get('model_sha256') != policy['model']['sha256'] or
            requested.get('taxonomy_sha256') != taxonomy_sha256(policy['classes'])):
        raise ValueError('invalid_operator_trial_scope_identity_or_basis')
    labels = requested.get('official_labels')
    mapping = {row['official_label']: row['material'] for row in policy['classes']}
    if (not isinstance(labels, list) or not labels or any(not isinstance(v, str) for v in labels) or
            len(labels) != len(set(labels)) or any(v not in mapping for v in labels) or
            any(mapping[v] not in ('paper', 'plastic', 'metal') for v in labels)):
        raise ValueError('invalid_operator_trial_scope_labels')
    return dict(materials=sorted({mapping[v] for v in labels}), official_labels=list(labels),
                basis='operator_supervised_trial', validated_scope=validated,
                physical_validation_completed=False, operator_statement=requested['operator_statement'])
