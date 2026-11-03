"""TACO original-name policy for pretrained open-vocabulary detectors.

Scores are ranking signals, not measured probabilities of material or recyclability.
This module never grants physical qualification from a class name or a threshold.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path

DEFAULT_POLICY = Path(__file__).resolve().parents[1] / 'config/yoloe_taco_recyclables.json'
MATERIALS = ('paper', 'plastic', 'metal')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def number(value, low, high, name):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(name + ': 数值须在 ' + str(low) + '–' + str(high) + ' 之间')
    return float(value)


def read_policy(path=None):
    from .execution_model_context import current_profile
    active=current_profile()
    if path is None and active is not None:return copy.deepcopy(active['policy'])
    path = path if path is not None else os.environ.get('YOLOE_RECYCLABLE_POLICY', DEFAULT_POLICY)
    policy = json.loads(Path(path).read_text(encoding='utf-8'))
    if policy['schema'] != 'prompted-recyclables-policy-v1':
        raise ValueError('不支持的分类策略')
    labels = [row['official_label'] for row in policy['classes']]
    if not labels or len(labels) > 80 or len(set(labels)) != len(labels):
        raise ValueError('官方类别为空、重复或超过 80 个')
    if set(policy['materials']) != set(MATERIALS):
        raise ValueError('必须配置纸、塑料、金属三个分区')
    if len({v['bin_id'] for v in policy['materials'].values()}) != 3:
        raise ValueError('投放区不能重复')
    for row in policy['classes']:
        if row['material'] not in (*MATERIALS, None):
            raise ValueError('未知材质分组')
        number(row['grasp_confidence'], .01, .99, '抓取阈值')
    settings_for(policy, {})
    return policy


def settings_for(policy, payload):
    """All overrides are explicit, validated and hashed together with taxonomy."""
    if payload.get('mode', 'text') != 'text':
        raise ValueError('三类分拣使用 TACO 文字提示模式')
    rows = {r['official_label']: r for r in policy['classes']}
    selected = payload.get('classes', list(rows))
    if isinstance(selected, str):
        selected = [s.strip() for s in selected.replace(',', '\n').splitlines() if s.strip()]
    if not isinstance(selected, list) or not selected or any(not isinstance(s, str) or s not in rows for s in selected):
        raise ValueError('提示词须使用类别表中的 TACO 原始名称')
    wanted = set(selected)
    names = [name for name in rows if name in wanted]
    if policy['model'].get('fixed_classes') and names != list(rows):
        raise ValueError('微调模型固定26类，不能更换或删减类别；可使用材质筛选')
    floor = number(policy['detection_floor'], .01, .95, '检测底线')
    display = number(payload.get('display_confidence', policy['display_confidence']), floor, .95, '显示阈值')
    heights = copy.deepcopy(policy['heights_mm'])
    thresholds = {m:{k:v for k,v in values.items() if k != 'bin_id'}
                  for m,values in policy['materials'].items()}
    class_thresholds = {n:r['grasp_confidence'] for n,r in rows.items()}
    for key, base in (('heights_mm', heights), ('class_thresholds', class_thresholds)):
        override = payload.get(key, {})
        if not isinstance(override, dict) or set(override) - set(base):
            raise ValueError('未知设置: ' + key)
        base.update(override)
    for label in MATERIALS:
        heights[label] = number(heights[label], *policy['height_range_mm'], '高度 mm')
    for label, value in class_thresholds.items():
        class_thresholds[label] = number(value, floor, .99, label + ' 抓取阈值')
    overrides = payload.get('material_thresholds', {})
    if not isinstance(overrides, dict) or set(overrides) - set(MATERIALS):
        raise ValueError('未知区域阈值')
    for label, values in overrides.items():
        if not isinstance(values, dict) or set(values) - {'source_confidence', 'bin_confidence'}:
            raise ValueError('区域只接受候选和落料验证阈值')
        thresholds[label].update(values)
    for values in thresholds.values():
        for key in ('source_confidence', 'bin_confidence'):
            values[key] = number(values[key], floor, .99, key)
    settings = dict(mode='text', confidence=floor, include_masks=True, classes=names,
        vocabulary_count=len(names), display_confidence=display, heights_mm=heights,
        material_thresholds=thresholds, class_thresholds=class_thresholds,
        policy_sha256=digest(policy))
    settings['settings_sha256'] = digest(settings)
    return settings


def classify(policy, settings, instance, zone='source'):
    """Keep source class intact; expose material and decision fields separately."""
    rows = {r['official_label']:r for r in policy['classes']}
    name = instance.get('official_label', instance['label'])
    score = number(instance['confidence'], 0, 1, '模型置信度')
    row = rows.get(name)
    material = row['material'] if row else None
    if instance.get('reference_assisted'):
        if not policy.get('reference_bank_sha256') or instance.get('reference_material') not in MATERIALS:
            raise ValueError('unregistered_reference_assistance')
        material=instance['reference_material']
    if zone not in ('source', *['bin_'+m for m in MATERIALS]):
        raise ValueError('未知区域')
    result = dict(instance, official_label=name, material=material,
                  sorting_bin=policy['materials'][material]['bin_id'] if material else None,
                  visible=score >= settings['display_confidence'], zone=zone,
                  obstacle=True, automatic_motion_allowed=False)
    reasons = []
    threshold = None
    if material is None:
        reasons.append('不在三类自动分拣清单')
    else:
        mt = settings['material_thresholds'][material]
        threshold = max(settings['class_thresholds'][name], mt['source_confidence' if zone == 'source' else 'bin_confidence'])
        if zone != 'source' and zone != 'bin_'+material:
            reasons.append('物体位于其他材质的投放区')
        if score < threshold:
            reasons.append('低于当前类别或区域门槛')
    result.update(decision_threshold=threshold, semantic_gate_passed=not reasons,
                  decision_reasons=reasons, physical_gate='待此类别与高度的实物验证')
    return result


def adapt(policy, settings, result):
    if result['model_sha256'] != policy['model']['sha256']:
        raise ValueError('识别权重与分类策略不匹配')
    return dict(result, schema_version='prompted-recyclables-observation-v1',
        instances=[classify(policy, settings, row) for row in result['instances']],
        settings_sha256=settings['settings_sha256'], policy_sha256=settings['policy_sha256'],
        heights_mm=settings['heights_mm'], trained_on_taco=bool(policy['model'].get('trained_on_taco', False)),
        physical_motion_allowed=False, accuracy_validated=False)
