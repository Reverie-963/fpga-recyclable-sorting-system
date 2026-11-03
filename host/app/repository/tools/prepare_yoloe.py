"""Fetch official YOLOE assets and cache fixed material text prompts, without training."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pynq_km1_vision'))
from src.yoloe_inference import DEFAULT_CONFIG, config_hash, file_hash, read_config

ASSETS = {
    'yoloe-11m-seg.pt': '785199b7cc75a5f4041dec15782a4b8a4105e5a34d7ffb287a0a10781724c5bd',
    'mobileclip_blt.ts': 'a67804d1b0f07b8b9a20c1761ec0847f34660f5fa338ec70e8f3fce68ed95e54',
}
BASE_URL = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/'


def prepare_visual(cfg, assets, lock, reference_manifest):
    import numpy as np
    import cv2
    import torch
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
    references = json.loads(Path(reference_manifest).read_text(encoding='utf-8'))
    if not isinstance(references, list) or not 3 <= len(references) <= 30:
        raise ValueError('Supply 3 to 30 visual references')
    if {r['category'] for r in references} != {'plastic', 'metal', 'paper'}:
        raise ValueError('References must cover all three materials')
    reference_hash = config_hash(references)
    visual_path = assets / 'yoloe-materials-visual.pt'
    if lock.get('visual_references_sha256') == reference_hash:
        if file_hash(visual_path) != lock['visual_sha256']:
            raise ValueError('Existing visual deployment hash mismatch')
        print('Existing visual prompts verified', flush=True)
        return
    if visual_path.exists():
        raise ValueError('Different visual deployment exists; preserve it before replacing references')
    model = YOLOE(str(assets / cfg['model_name'])).to('cpu')
    torch.set_num_threads(cfg['cpu_threads'])
    embeddings, prompts = [], []
    for i, reference in enumerate(references):
        path = (ROOT / reference['image']).resolve()
        if not path.is_relative_to(ROOT / 'local-only') or reference['source_split'] != 'train':
            raise ValueError('Use explicitly recorded development references from local-only')
        if file_hash(path) != reference['image_sha256']:
            raise ValueError('Reference image hash mismatch')
        frame = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        box = np.asarray(reference['bbox_xyxy'], dtype=float)
        if (frame is None or box.shape != (4,) or not np.isfinite(box).all() or
            not 0 <= box[0] < box[2] <= frame.shape[1] or
            not 0 <= box[1] < box[3] <= frame.shape[0]):
            raise ValueError('Invalid reference image or box')
        if hashlib.sha256(frame.tobytes()).hexdigest() != reference['decoded_bgr_sha256']:
            raise ValueError('Decoded reference image hash mismatch')
        predictor = YOLOEVPSegPredictor(overrides=dict(task='segment', mode='predict',
            save=False, verbose=False, batch=1, device='cpu', half=False, imgsz=640),
            _callbacks=model.callbacks)
        model.model.model[-1].nc = 1
        model.model.names = ['object0']
        predictor.set_prompts(dict(bboxes=[box.tolist()], cls=[0]))
        predictor.setup_model(model=model.model, verbose=False)
        with torch.inference_mode():
            embeddings.append(predictor.get_vpe(frame).detach())
        prompts.append(dict(text='visual_example_{:02d}'.format(i), category=reference['category'],
            description=reference['title'], reference_image_sha256=reference['image_sha256']))
        print('Encoded visual reference {}/{}'.format(i+1, len(references)), flush=True)
    model.model.set_classes([p['text'] for p in prompts], torch.cat(embeddings, dim=1))
    model.predictor = None
    model.save(str(visual_path))
    (assets / 'visual_references.json').write_text(json.dumps(references, ensure_ascii=False, indent=2), encoding='utf-8')
    lock.update(visual_sha256=file_hash(visual_path), visual_references_sha256=reference_hash,
                visual_prompts=prompts, visual_reference_count=len(references),
                visual_prompt_note='Reference embeddings only; no gradient updates or fine-tuning')
    (assets / 'deployment.json').write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding='utf-8')


def prepare(config, reference_manifest=None):
    cfg = read_config(config)
    if cfg['model_name'] != 'yoloe-11m-seg.pt':
        raise ValueError('This setup pins the official YOLOE-11m segmentation checkpoint')
    assets = ROOT / cfg['asset_dir']
    assets.mkdir(parents=True, exist_ok=True)
    provenance = []
    for name, expected in ASSETS.items():
        path = assets / name
        url = BASE_URL + name
        if not path.exists():
            temporary = path.with_suffix(path.suffix + '.download')
            print('Downloading ' + name, flush=True)
            with urllib.request.urlopen(url, timeout=60) as source, temporary.open('wb') as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
            if file_hash(temporary) != expected:
                raise ValueError('Downloaded hash mismatch: ' + str(temporary))
            temporary.replace(path)
        if file_hash(path) != expected:
            raise ValueError('Asset hash mismatch; not replacing existing file: ' + str(path))
        provenance.append(dict(file=name, url=url, sha256=expected, bytes=path.stat().st_size))
    lock_path = assets / 'deployment.json'
    prompted = assets / 'yoloe-materials-prompted.pt'
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding='utf-8'))
        if lock['config_sha256'] != config_hash(cfg) or file_hash(prompted) != lock['prompted_sha256']:
            raise ValueError('Existing deployment differs; preserve it before creating a new prompt configuration')
        print('Existing prompted deployment verified', flush=True)
        if reference_manifest:
            os.environ['YOLO_AUTOINSTALL'] = 'false'
            os.environ['YOLO_CONFIG_DIR'] = str(assets.parent / 'ultralytics-settings')
            (assets.parent / 'ultralytics-settings/Ultralytics').mkdir(parents=True, exist_ok=True)
            prepare_visual(cfg, assets, lock, reference_manifest)
        return
    if prompted.exists():
        raise ValueError('Unrecorded prompted weights exist; preserve and inspect them before retrying')
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ['YOLO_CONFIG_DIR'] = str(assets.parent / 'ultralytics-settings')
    (assets.parent / 'ultralytics-settings/Ultralytics').mkdir(parents=True, exist_ok=True)
    import torch
    import ultralytics
    from ultralytics import YOLOE
    if ultralytics.__version__ != '8.3.230' or torch.__version__ != '2.6.0+cpu':
        raise ValueError('Use the pinned dedicated CPU runtime')
    torch.set_num_threads(cfg['cpu_threads'])
    model = YOLOE(str(assets / cfg['model_name'])).to('cpu')
    prompts = [x['text'] for x in cfg['prompts']]
    # The pinned text encoder locates mobileclip_blt.ts in the working directory.
    previous = Path.cwd()
    try:
        os.chdir(assets)
        with torch.inference_mode():
            embeddings = model.get_text_pe(prompts)
            model.set_classes(prompts, embeddings)
    finally:
        os.chdir(previous)
    model.save(str(prompted))
    lock = dict(schema_version='yoloe-prompted-deployment-v1',
        created_at=datetime.now(timezone.utc).isoformat(),
        official_assets=provenance, config_sha256=config_hash(cfg),
        prompted_sha256=file_hash(prompted), prompts=cfg['prompts'],
        torch=torch.__version__, ultralytics=ultralytics.__version__,
        device='cpu', trained=False, text_embeddings_cached=True,
        automatic_motion_allowed=False)
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(lock, ensure_ascii=False, indent=2), flush=True)
    if reference_manifest:
        prepare_visual(cfg, assets, lock, reference_manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--references', type=Path, help='Optional local visual-reference manifest')
    args = parser.parse_args()
    prepare(args.config, args.references)
