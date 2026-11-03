"""Explicit numerical vision-review margins; no timing or identity changes."""
import math
from .sorting_loop import need


def review_factor(owner):
    policy=owner.get('visual_review_policy')
    if policy is None:return 1
    need(isinstance(policy,dict) and policy.get('kind')=='operator-vision-review-margin-v1' and
         bool(policy.get('operator_statement')) and policy.get('preserve_timings') is True,
         'vision_review_margin_requires_operator_record')
    factor=policy.get('tolerance_factor')
    need(type(factor) in (int,float) and math.isfinite(factor) and 1<=factor<=3,
         'invalid_vision_review_factor')
    return factor


def checked_factor(value):
    need(type(value) in (int,float) and math.isfinite(value) and 1<=value<=3,
         'invalid_vision_review_factor')
    return value
