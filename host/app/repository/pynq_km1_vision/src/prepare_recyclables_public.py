"""Build a bounded TACO training supplement while preserving local validation/test data."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import urllib.request
from urllib.parse import urlparse

import cv2
import numpy as np

from .train_segment import check_dataset, MATERIAL_LABELS

ANNOTATIONS_URL = 'https://raw.githubusercontent.com/pedropro/TACO/1.0/data/annotations.json'
# Use explicit material categories; ambiguous composite/unknown litter is excluded below.
CLASS_MAP = {
    'Drink can': 0, 'Food Can': 0,
    'Clear plastic bottle': 1, 'Other plastic bottle': 1, 'Plastic straw': 1,
    'Plastic bottle cap': 1, 'Plastic lid': 1, 'Plastic utensils': 1,
    'Other plastic': 1, 'Other plastic container': 1, 'Other plastic cup': 1,
    'Disposable plastic cup': 1, 'Other plastic wrapper': 1, 'Plastic film': 1,
    'Single-use carrier bag': 1, 'Garbage bag': 1, 'Polypropylene bag': 1,
    'Plastic glooves': 1, 'Six pack rings': 1, 'Tupperware': 1,
    'Corrugated carton': 2, 'Drink carton': 2, 'Other carton': 2,
    'Meal carton': 2, 'Egg carton': 2, 'Pizza box': 2,
}
AMBIGUOUS = {'Unlabeled litter', 'Carded blister pack', 'Crisp packet',
             'Disposable food container', 'Squeezable tube', 'Spread tub',
             'Plastified paper bag', 'Styrofoam piece', 'Foam cup', 'Foam food container'}


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def fetch(url, limit=25 * 1024 * 1024):
    parsed = urlparse(url)
    if parsed.scheme != 'https' or not (parsed.hostname == 'raw.githubusercontent.com'
            or parsed.hostname == 'olm-s3.s3.eu-west-1.amazonaws.com'
            or (parsed.hostname or '').endswith('.staticflickr.com')):
        raise ValueError('Unsupported public image host: ' + str(parsed.hostname))
    with urllib.request.urlopen(url, timeout=25) as response:
        raw = response.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('Download exceeds byte limit')
    return raw


def dhash(image):
    gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (9, 8))
    return int(''.join('1' if x else '0' for x in (gray[:, 1:] > gray[:, :-1]).flat), 2)


def polygon_line(annotation, image, category_map):
    """Keep a single COCO ring as one instance; never turn boxes/ring fragments into instances."""
    segments = annotation.get('segmentation')
    if annotation.get('iscrowd') or not isinstance(segments, list) or len(segments) != 1:
        raise ValueError('crowd, RLE or multipart polygon')
    points = segments[0]
    if len(points) < 6 or len(points) % 2 or not all(math.isfinite(v) for v in points):
        raise ValueError('invalid vertices')
    width, height = image['width'], image['height']
    coords = [v / (width if i % 2 == 0 else height) for i, v in enumerate(points)]
    if any(v < 0 or v > 1 for v in coords):
        raise ValueError('out of bounds')
    pairs = list(zip(coords[::2], coords[1::2]))
    area = abs(sum(x*y1-x1*y for (x, y), (x1, y1) in zip(pairs, pairs[1:] + pairs[:1]))) / 2
    if area < 0.001:
        raise ValueError('tiny/degenerate target; exclude the whole image')
    return str(category_map[annotation['category_id']]) + ' ' + ' '.join(f'{v:.8f}' for v in coords)


def candidates(data):
    names = {c['id']: c['name'] for c in data['categories']}
    mapping = {cid: CLASS_MAP[name] for cid, name in names.items() if name in CLASS_MAP}
    by_image = defaultdict(list)
    for ann in data['annotations']:
        by_image[ann['image_id']].append(ann)
    selected, skipped = [], Counter()
    for image in data['images']:
        anns = by_image[image['id']]
        targets = [a for a in anns if a['category_id'] in mapping]
        if not targets:
            continue
        if any(names[a['category_id']] in AMBIGUOUS for a in anns):
            skipped['ambiguous_material_or_unlabeled_litter'] += 1
            continue
        try:
            lines = [polygon_line(a, image, mapping) for a in targets]
        except ValueError:
            skipped['invalid_multipart_or_tiny_target'] += 1
            continue
        counts = Counter(mapping[a['category_id']] for a in targets)
        selected.append((image, lines, counts, targets))
    return selected, skipped


def build(base, output, cache, per_class=40, workers=4):
    base, output, cache = Path(base).resolve(), Path(output).resolve(), Path(cache).resolve()
    baseline = check_dataset(base)
    if baseline['labels'] != MATERIAL_LABELS:
        raise ValueError('Requires confirmed can/plastic/cardboard schema; do not rename bottle-only data')
    if output.exists():
        raise ValueError('Output already exists; choose a new version')
    if (output.is_relative_to(base) or base.is_relative_to(output)
            or cache.is_relative_to(output) or output.is_relative_to(cache)
            or cache.is_relative_to(base) or base.is_relative_to(cache)):
        raise ValueError('Use separate base, cache and output directories')
    cache.mkdir(parents=True, exist_ok=True)
    annotation_path = cache / 'taco-release-1.0-annotations.json'
    if not annotation_path.exists():
        annotation_path.write_bytes(fetch(ANNOTATIONS_URL))
    raw_annotations = annotation_path.read_bytes()
    data = json.loads(raw_annotations)
    pending, skipped = candidates(data)
    random.Random(7).shuffle(pending)
    manifest = json.loads((base / 'split_manifest.json').read_text(encoding='utf-8-sig'))
    output.mkdir(parents=True)
    hashes, perceptual = set(), []
    for row in manifest['images']:
        rel = Path(row['image'])
        label = Path('labels', row['split'], *rel.parts[2:]).with_suffix('.txt')
        for path in (rel, label):
            (output / path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(base / path, output / path)
        raw = (base / rel).read_bytes()
        hashes.add(sha256(raw))
        perceptual.append(dhash(cv2.imread(str(base / rel))))
    source_review = base / 'annotation-review.json'
    if source_review.exists():
        shutil.copy2(source_review, output / 'base-annotation-review.json')
    bucket_counts, instance_counts, added, failures = Counter(), Counter(), [], []

    def download(candidate):
        image = candidate[0]
        path = cache / f"taco-{image['id']:04d}.image"
        urls = list(dict.fromkeys(u for u in (image.get('flickr_640_url'), image.get('flickr_url')) if u))
        if path.exists():
            cached = path.read_bytes()
            if cv2.imdecode(np.frombuffer(cached, np.uint8), cv2.IMREAD_COLOR) is not None:
                return candidate, cached, 'cache:' + (urls[0] if urls else 'unknown')
        errors = []
        for url in urls:
            try:
                raw = fetch(url)
                if cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) is None:
                    raise ValueError('unreadable image')
                part = path.with_suffix('.part')
                part.write_bytes(raw)
                part.replace(path)
                (cache / f"taco-{image['id']:04d}-url.txt").write_text(url, encoding='utf-8')
                return candidate, raw, url
            except Exception as error:
                errors.append(type(error).__name__ + ': ' + str(error))
        return candidate, None, '; '.join(errors)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Small bounded batches avoid fetching the whole dataset, and allow per-class balancing.
        while pending and min(bucket_counts[c] for c in range(3)) < per_class:
            batch = []
            while pending and len(batch) < workers:
                item = pending.pop()
                available = [c for c in item[2] if bucket_counts[c] < per_class]
                if available:
                    bucket = min(available, key=lambda c: (bucket_counts[c], c))
                    batch.append((item, bucket))
            for (item, bucket), (_, raw, url) in zip(batch, pool.map(download, [b[0] for b in batch])):
                image, lines, counts, targets = item
                if raw is None:
                    failures.append({'image_id': image['id'], 'error': url})
                    continue
                if bucket_counts[bucket] >= per_class:
                    continue
                decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                height, width = decoded.shape[:2]
                if abs((width / height) / (image['width'] / image['height']) - 1) > 0.01:
                    skipped['aspect_ratio_mismatch'] += 1
                    continue
                fingerprint = dhash(decoded)
                if sha256(raw) in hashes or any((fingerprint ^ p).bit_count() <= 2 for p in perceptual):
                    skipped['exact_or_near_duplicate'] += 1
                    continue
                hashes.add(sha256(raw))
                perceptual.append(fingerprint)
                if max(width, height) > 640:
                    scale = 640 / max(width, height)
                    decoded = cv2.resize(decoded, (round(width*scale), round(height*scale)), interpolation=cv2.INTER_AREA)
                rel = Path('images/train') / f"taco-{image['id']:04d}.jpg"
                cv2.imwrite(str(output / rel), decoded, [cv2.IMWRITE_JPEG_QUALITY, 95])
                label = output / 'labels/train' / (rel.stem + '.txt')
                label.write_text('\n'.join(lines) + '\n', encoding='utf-8')
                cached_url = cache / f"taco-{image['id']:04d}-url.txt"
                actual_url = cached_url.read_text(encoding='utf-8') if cached_url.exists() else url.removeprefix('cache:')
                row = {'image': rel.as_posix(), 'split': 'train',
                       'object_ids': [f"taco-1.0-image-{image['id']}-annotation-{a['id']}" for a in targets],
                       'session_id': 'taco-1.0-' + image['file_name'].split('/')[0],
                       'source': 'TACO release 1.0', 'source_image_id': image['id'],
                       'source_file_name': image['file_name'], 'source_url': actual_url,
                       'source_image_sha256': sha256(raw), 'source_license': image.get('license'),
                       'image_sha256': sha256((output / rel).read_bytes()),
                       'annotation_status': 'official_source_polygons_not_individually_human_reviewed',
                       'balance_bucket': MATERIAL_LABELS[bucket]}
                manifest['images'].append(row)
                added.append(row)
                bucket_counts[bucket] += 1
                instance_counts.update(counts)
                print(json.dumps({'downloaded': len(added), 'balance_buckets': dict(bucket_counts)}), flush=True)
    manifest.update(class_names=MATERIAL_LABELS, public_source=ANNOTATIONS_URL,
                    public_annotations_sha256=sha256(raw_annotations), public_class_map=CLASS_MAP,
                    public_grouping_note='Source annotation IDs are not verified physical identities; all public data is train-only.')
    (output / 'split_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    report = check_dataset(output)
    report.update(public_images=len(added), public_instances=dict(instance_counts),
                  balance_buckets=dict(bucket_counts), requested_per_class=per_class,
                  base_dataset_sha256=baseline['dataset_sha256'], skipped=dict(skipped), failures=failures,
                  source_url=ANNOTATIONS_URL, source_annotations_sha256=sha256(raw_annotations),
                  license_note='Toolkit MIT does not establish every photo license. Preserve per-image fields; local research copy only; redistribution not cleared.',
                  validation_note='Base validation/test images and labels copied byte-for-byte; no public images in holdouts.',
                  review_note='Official source annotations with geometric checks; contact sheets require inspection; not local human approval.')
    (output / 'public-data-report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    create_previews(output, added)
    return report


def create_previews(root, rows):
    previews = root / 'previews'
    previews.mkdir(exist_ok=True)
    for cid, name in enumerate(MATERIAL_LABELS):
        chosen = [r for r in rows if r['balance_bucket'] == name][:12]
        sheet = np.full((3*200, 4*240, 3), 245, dtype=np.uint8)
        for i, row in enumerate(chosen):
            image = cv2.imread(str(root / row['image']))
            h, w = image.shape[:2]
            for line in (root / 'labels/train' / (Path(row['image']).stem + '.txt')).read_text().splitlines():
                values = [float(v) for v in line.split()]
                points = np.rint(np.array(values[1:]).reshape(-1, 2) * [w, h]).astype(np.int32)
                color = [(70, 220, 70), (240, 150, 30), (30, 100, 240)][int(values[0])]
                cv2.polylines(image, [points], True, color, 2)
            scale = min(240/w, 175/h)
            thumb = cv2.resize(image, (round(w*scale), round(h*scale)))
            y, x = (i // 4)*200, (i % 4)*240
            sheet[y:y+thumb.shape[0], x:x+thumb.shape[1]] = thumb
            cv2.putText(sheet, Path(row['image']).stem, (x+3, y+193), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1)
        cv2.imwrite(str(previews / (name + '.jpg')), sheet)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--per-class', type=int, default=40)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    if not 1 <= a.per_class <= 500 or not 1 <= a.workers <= 8:
        p.error('per-class must be 1..500 and workers 1..8')
    print(json.dumps(build(a.base, a.output, a.cache, a.per_class, a.workers), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
