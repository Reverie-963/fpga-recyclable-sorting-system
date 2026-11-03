"""Reproduce the 19-image, same-specimen workflow exercise (not a benchmark)."""
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
# Manually reviewed whole-object boxes: x1,y1,x2,y2 in the original 640x480 frame.
RECORDS = [
    (0, '1746293899356483904', 'train', [316, 163, 353, 199]),
    (0, '1746293904858634512', 'val', [273, 82, 309, 119]),
    (0, '1746293909928601138', 'train', [355, 73, 393, 111]),
    (0, '1746293914998606195', 'test', [409, 122, 449, 162]),
    (0, '1746294407406283283', 'train', [289, 42, 342, 98]),
    (0, '1746294608022067131', 'train', [344, 252, 399, 311]),
    (0, '1746294828089791197', 'val', [341, 160, 386, 206]),
    (0, '1746294902297858348', 'test', [338, 238, 383, 282]),
    (0, '1746295039357338009', 'train', [334, 255, 380, 302]),
    (1, '1746295623221693902', 'train', [368, 242, 406, 292]),
    (1, '1746295628291712534', 'val', [468, 198, 508, 249]),
    (1, '1746295633361909001', 'train', [490, 292, 530, 350]),
    (1, '1746295638467873122', 'test', [438, 389, 496, 437]),
    (1, '1746295643567956111', 'train', [522, 382, 583, 431]),
    (1, '1746295648607867532', 'train', [539, 284, 582, 342]),
    (1, '1746295653713652853', 'val', [471, 294, 525, 342]),
    (1, '1746295658783741264', 'train', [369, 279, 411, 331]),
    (1, '1746295663883768303', 'test', [391, 171, 438, 213]),
    (1, '1746295668923675268', 'train', [538, 168, 584, 216]),
]


def main():
    dest = ROOT / 'data/cap_deformation'
    names = ['cap_ok', 'cap_deformed']
    manifest = {'record_date': '2026-10-07', 'purpose': 'software_pipeline_exercise',
                'split_policy': 'same_specimen_frame_split', 'independent_test': False,
                'warning': 'One specimen per class; shared specimens across splits. Metrics are NOT independent accuracy or competition evidence.',
                'annotation': 'Manual whole-object boxes; class identities from user capture folders, visually reviewed.',
                'labels': names, 'images': []}
    cards = []
    for cls, stem, split, box in RECORDS:
        source = ROOT / f'data/defect_raw/session01/{names[cls]}_001/{stem}.jpg'
        frame = cv2.imread(str(source))
        if frame is None or frame.shape[:2] != (480, 640):
            raise ValueError(f'Unexpected image: {source}')
        x1, y1, x2, y2 = box
        assert 0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 480
        image_dest = dest / f'images/{split}/{stem}.jpg'
        label_dest = dest / f'labels/{split}/{stem}.txt'
        label = f'{cls} {(x1+x2)/1280:.6f} {(y1+y2)/960:.6f} {(x2-x1)/640:.6f} {(y2-y1)/480:.6f}\n'
        for path, content in [(image_dest, source.read_bytes()), (label_dest, label.encode())]:
            if path.exists() and path.read_bytes() != content:
                raise FileExistsError(f'Refusing to replace modified data: {path}')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        manifest['images'].append({'source': str(source.relative_to(ROOT)).replace('\\', '/'),
                                   'specimen': names[cls] + '_001', 'class_id': cls,
                                   'split': split, 'bbox_xyxy': box, 'width': 640, 'height': 480})
        color = '#00ff88' if cls == 0 else '#ff9944'
        cards.append(f'<figure><figcaption>{names[cls]} / {split} / {stem}</figcaption>'
                     f'<svg viewBox="0 0 640 480"><image href="../data/cap_deformation/images/{split}/{stem}.jpg" width="640" height="480"/>'
                     f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" fill="none" stroke="{color}" stroke-width="2"/></svg></figure>')
    report = ROOT / 'report'
    report.mkdir(exist_ok=True)
    (report / 'cap_deformation_demo_dataset_20261007.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (dest / 'DATASET_NOTE.txt').write_text(manifest['warning'] + '\nSplit: train 11, val 4, test 4. Two classes: 0 cap_ok, 1 cap_deformed.\n', encoding='utf-8')
    (report / 'cap_deformation_annotation_review.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>Cap annotation review</title>'
        '<style>body{background:#202124;color:#eee;font:14px sans-serif;margin:24px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:20px}figure{margin:0}svg{width:100%}</style>'
        '<h1>正常 / 变形：19 张整件框标注</h1><p>绿色=正常；橙色=变形。训练 11 张，验证 4 张，测试 4 张。每类仅一件实物，同件演练划分，不能作为独立准确率。</p><main>'
        + ''.join(cards) + '</main>', encoding='utf-8')
    print(json.dumps({'images': len(RECORDS), 'splits': {s: sum(r[2] == s for r in RECORDS) for s in ['train', 'val', 'test']}, 'independent_test': False}))


if __name__ == '__main__':
    main()
