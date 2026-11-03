"""Build the existing 64-image/16-item support bank; never read replay images."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'pynq_km1_vision'))


def build(dataset, plan, regions, output):
    import cv2
    import numpy as np
    from src.inventory_reference import descriptor
    data = json.loads(Path(dataset, 'split_manifest.json').read_text(encoding='utf-8-sig'))
    recipe = json.loads(Path(plan).read_text(encoding='utf-8-sig'))
    region = json.loads(Path(regions).read_text(encoding='utf-8-sig'))
    if hashlib.sha256(Path(regions).read_bytes()).hexdigest() != recipe['regions_sha256']:
        raise ValueError('reference_region_not_frozen_recipe')
    rows = {row['image']: row for row in data['images']}
    references = []
    for name in recipe['reference_images']:
        row = rows[name]
        if recipe['source_roles'][name] != 'development' or len(row['object_ids']) != 1:
            raise ValueError('only_frozen_development_references_allowed')
        image_path = Path(dataset)/name
        label_path = Path(dataset)/'labels'/row['split']/(Path(name).stem+'.txt')
        raw, label = image_path.read_bytes(), label_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row['image_sha256']:
            raise ValueError('reference_image_changed')
        if row['label_sha256'] not in (hashlib.sha256(label).hexdigest(), hashlib.sha256(label.replace(b'\r\n', b'\n')).hexdigest()):
            raise ValueError('reference_label_changed')
        lines = label.decode().strip().splitlines()
        if len(lines) != 1:
            raise ValueError('single_reference_foreground_required')
        values = [float(v) for v in lines[0].split()]
        cid = int(values[0])
        if values[0] != cid or cid not in (0, 1, 2):
            raise ValueError('unknown_reference_material')
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image is None or list(image.shape[1::-1]) != region['image_size']:
            raise ValueError('reference_image_size_changed')
        mask = np.zeros(image.shape[:2], np.uint8)
        polygon = np.asarray(values[1:]).reshape(-1, 2)*region['image_size']
        cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 1)
        features = descriptor(image, mask.astype(bool))
        references.append(dict(image=name, object_id=row['object_ids'][0],
            material=['metal', 'plastic', 'paper'][cid], image_sha256=row['image_sha256'],
            image_bgr_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
            label_sha256=hashlib.sha256(label).hexdigest(), features={k: v.tolist() for k, v in features.items()}))
    counts = Counter(r['object_id'] for r in references)
    if len(references) != 64 or len(counts) != 16 or set(counts.values()) != {4}:
        raise ValueError('expected_64_references_for_16_objects')
    result = dict(schema='known-inventory-reference-v1', image_size=region['image_size'],
        workcell_id=region['workcell_id'], source_polygon=region['regions']['source'],
        reference_recipe='HSV24x8 Hellinger .6 + mean Lab .25 + log aspect/fill .15',
        source_plan_sha256=hashlib.sha256(Path(plan).read_bytes()).hexdigest(),
        reference_count=64, object_count=16, references=references, automatic_motion_allowed=False)
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT/'local-only'):
        raise ValueError('reference_bank_must_be_local_only')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return dict(reference_count=64, object_count=16, sha256=hashlib.sha256(output.read_bytes()).hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=ROOT/'local-only/recyclables-final-16-v1')
    p.add_argument('--plan', type=Path, default=ROOT/'local-only/fixed16-focus-20261031/final-two26-reference-replay/plan.json')
    p.add_argument('--regions', type=Path, default=ROOT/'pynq_km1_vision/config/yoloe_station_filter.example.json')
    p.add_argument('--output', type=Path, default=ROOT/'local-only/model-switch-console/reference-bank.json')
    a = p.parse_args()
    print(json.dumps(build(a.dataset, a.plan, a.regions, a.output)))


if __name__ == '__main__': main()
