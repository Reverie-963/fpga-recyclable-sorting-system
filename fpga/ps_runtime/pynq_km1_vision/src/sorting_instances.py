"""Unique one-to-one mask association, independent of detector ordering."""
from .sorting_loop import iou, need


def associate(reference, current, min_iou, allow_extra=False):
    """Return reference->current indices; reject missing or ambiguous matches."""
    need(0 < min_iou <= 1, 'invalid_instance_match_threshold')
    need(len(reference) <= len(current) and (allow_extra or len(reference) == len(current)),
         'instance_count_changed')
    graph = [[j for j, (obj, mask) in enumerate(current)
              if old['label'] == obj['label'] and iou(previous, mask) >= min_iou]
             for old, previous in reference]
    need(all(graph), 'instance_missing_split_or_moved')
    solutions = []
    order = sorted(range(len(graph)), key=lambda i: len(graph[i]))

    def visit(depth, used, result):
        if len(solutions) > 1:
            return
        if depth == len(order):
            solutions.append(result.copy())
            return
        index = order[depth]
        for candidate in graph[index]:
            if candidate not in used:
                result[index] = candidate
                visit(depth + 1, used | {candidate}, result)

    visit(0, set(), {})
    need(len(solutions) == 1, 'instance_identity_ambiguous_or_merged')
    return [solutions[0][i] for i in range(len(reference))]


class InstanceLedger:
    def __init__(self, min_iou):
        self.min_iou = min_iou
        self.live = []
        self.next_id = 1
        self.initialized = False
        self.initial = []
        self.attempted = []

    def observe(self, pairs):
        if not self.initialized:
            self.live = [(self.next_id + i, obj, mask.copy()) for i, (obj, mask) in enumerate(pairs)]
            self.next_id += len(pairs)
            self.initialized = True
            self.initial = [(track,o,m.copy()) for track,o,m in self.live]
        else:
            # A completed motion is never repeated just because its visual
            # verdict is deferred. Keep such objects as scene obstacles.
            matches = associate([(o, m) for _, o, m in self.live], pairs, self.min_iou,
                                allow_extra=bool(self.attempted))
            extras=[pair for j,pair in enumerate(pairs) if j not in matches]
            need(all(any(iou(mask,old)>=self.min_iou for _,_,old in self.attempted)
                     for _,mask in extras),'unexplained_instance_after_deferred_motion')
            self.live = [(track, pairs[j][0], pairs[j][1].copy())
                         for (track, _, _), j in zip(self.live, matches)]
        return self.live

    def defer(self, selected_id):
        selected=[item for item in self.live if item[0]==selected_id]
        need(len(selected)==1,'selected_track_not_live')
        self.attempted.extend(selected)
        self.live=[item for item in self.live if item[0]!=selected_id]

    def retire(self, selected_id, remaining):
        expected = [(track, o, m) for track, o, m in self.live if track != selected_id]
        need(len(expected) + 1 == len(self.live), 'selected_track_not_live')
        matches = associate([(o, m) for _, o, m in expected], remaining, self.min_iou)
        self.live = [(track, remaining[j][0], remaining[j][1].copy())
                     for (track, _, _), j in zip(expected, matches)]
