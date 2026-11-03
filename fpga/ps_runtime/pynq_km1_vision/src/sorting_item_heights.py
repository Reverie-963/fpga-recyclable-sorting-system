"""Operator measurements attached to instance masks, never material defaults."""
import copy
import re
import numpy as np
from .prompted_height import object_height
from .static_mask_contract import decode_mask
from .sorting_loop import need, iou


def height_plan(value):
    if value is None:
        return None
    need(isinstance(value, dict) and set(value) == {
        'kind', 'model_sha256', 'configuration_sha256', 'source_stream_id', 'image_size', 'items'
    } and value['kind'] == 'instance-heights-v1', 'invalid_instance_height_plan')
    for name in ('model_sha256', 'configuration_sha256'):
        need(isinstance(value[name], str) and re.fullmatch('[a-f0-9]{64}', value[name]),
             'invalid_instance_height_identity')
    need(isinstance(value['source_stream_id'], str) and 0 < len(value['source_stream_id']) <= 128,
         'invalid_instance_height_stream')
    size = value['image_size']
    need(isinstance(size, list) and len(size) == 2 and all(type(n) is int for n in size)
         and 1 <= size[0] <= 640 and 1 <= size[1] <= 480, 'invalid_instance_height_image_size')
    need(isinstance(value['items'], list) and 1 <= len(value['items']) <= 12,
         'invalid_instance_height_count')
    ids = set()
    for item in value['items']:
        required={
            'item_id', 'label', 'official_label', 'mask_rle', 'object_height_mm'
        }
        need(isinstance(item, dict) and required.issubset(item) and
             not set(item)-required-{'grasp_width_mm','measurement_profile'}, 'invalid_instance_height_item')
        need(type(item['item_id']) is int and 1 <= item['item_id'] <= 12 and item['item_id'] not in ids,
             'duplicate_instance_height_id')
        ids.add(item['item_id'])
        need(item['label'] in ('paper', 'metal', 'plastic') and
             isinstance(item['official_label'], str) and 0 < len(item['official_label']) <= 120,
             'invalid_instance_height_class')
        need(object_height(item['object_height_mm']) is not None, 'instance_height_required')
        if 'grasp_width_mm' in item:
            width=item['grasp_width_mm']
            need(type(width) in (int,float) and np.isfinite(width) and 1<=width<=150,
                 'invalid_instance_measured_width')
        if 'measurement_profile' in item:
            profile=item['measurement_profile']
            need(isinstance(profile,dict) and set(profile)=={'id','revision','name','pose'} and
                 isinstance(profile['id'],str) and re.fullmatch('[a-f0-9]{32}',profile['id']) and
                 type(profile['revision']) is int and profile['revision']>=1 and
                 all(isinstance(profile[k],str) and 0<len(profile[k])<=limit for k,limit in (('name',60),('pose',24))),
                 'invalid_instance_measurement_profile')
        runs = item['mask_rle']
        need(isinstance(runs, list) and 1 <= len(runs) <= 8192, 'invalid_instance_height_mask')
        need(any(decode_mask(runs, size)), 'empty_instance_height_mask')
    return copy.deepcopy(value)


class ItemHeights:
    def __init__(self, value, *, model, configuration, image_size):
        self.plan = height_plan(value)
        need(self.plan['model_sha256'] == model and self.plan['configuration_sha256'] == configuration,
             'instance_height_model_or_configuration_changed')
        need(self.plan['image_size'] == image_size, 'instance_height_image_size_changed')
        w, h = image_size
        self.items = [(item, np.asarray(decode_mask(item['mask_rle'], image_size), bool).reshape(h, w))
                      for item in self.plan['items']]
        self.bound = {}

    def bind(self, pairs, stream_id, minimum_iou):
        self.bound = {}
        need(stream_id == self.plan['source_stream_id'], 'instance_height_camera_stream_changed')
        bound, used = {}, set()
        for obj, mask in pairs:
            matches = [(item, ref) for item, ref in self.items
                       if item['label'] == obj['label']
                       and iou(ref, mask) >= minimum_iou]
            need(len(matches) == 1, 'instance_height_missing_or_moved' if not matches
                 else 'instance_height_ambiguous')
            item = matches[0][0]
            need(item['item_id'] not in used, 'instance_height_reused_for_two_objects')
            used.add(item['item_id'])
            bound[id(obj)] = item
        self.bound = bound

    def measurement(self, obj):
        need(id(obj) in self.bound, 'instance_height_not_bound')
        return self.bound[id(obj)]

    def check_scene(self, pairs, stream_id, minimum_iou):
        previous=self.bound
        try:
            self.bind(pairs,stream_id,minimum_iou)
        finally:
            self.bound=previous


def rank_candidates(candidates, movement_wait):
    """Safety gates run first; prefer space, then complete route time and confidence."""
    for track, target, choice, plan in candidates:
        clearance = choice.get('nominal_neighbor_clearance_mm')
        # Cap the preference: an empty table must not create an infinite score.
        space = 50.0 if clearance is None else min(50.0, clearance)
        seconds = sum(movement_wait(name, duration) for name, _, duration in plan)
        choice['selection'] = dict(policy='clearance_then_route_v1',
            clearance_preference_mm=space, complete_route_wait_s=seconds,
            confidence=float(target[0]['confidence']), stable_track_id=track,
            reason='verified_route_then_clearance_then_duration_then_confidence')
    candidates.sort(key=lambda c: (-c[2]['selection']['clearance_preference_mm'],
        c[2]['selection']['complete_route_wait_s'], -c[2]['selection']['confidence'], c[0]))
    for index, candidate in enumerate(candidates, 1):
        candidate[2]['selection']['rank'] = index
    return candidates
