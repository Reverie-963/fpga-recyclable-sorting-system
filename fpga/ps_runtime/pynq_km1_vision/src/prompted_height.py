"""Operator height input is table-relative millimetres, never robot grasp Z."""
import math


def object_height(value):
    if value is None:return None
    if type(value) not in (int,float) or not math.isfinite(value) or not 1<=value<=120:
        raise ValueError('object_height_must_be_1_to_120_mm')
    return float(value)


def effective_heights(settings,value,materials):
    height=object_height(value)
    result=dict(settings['heights_mm'])
    if height is not None:
        for material in materials:result[material]=height
    return result
