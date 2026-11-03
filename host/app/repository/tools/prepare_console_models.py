"""Register existing finished checkpoints; no training, service changes or hardware."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'pynq_km1_vision'))
from src.recognition_profiles import configuration_hash


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def prepare(output, device='cuda:0'):
    if device not in ('cuda:0', 'cpu'):
        raise ValueError('unsupported_candidate_device')
    backend = 'ROCm GPU' if device == 'cuda:0' else 'CPU'
    policy = read(ROOT/'local-only/yoloe11m-recyclables26/console-deployment-20261030/policy.json')
    classes = [row['official_label'] for row in policy['classes']]
    mapping = {row['official_label']: row['material'] for row in policy['classes']}
    models = [dict(id='yoloe26', name='YOLOE-11m · 26类微调 · '+backend, kind='service', device=device,
        protocol='prompted', url='http://127.0.0.1:8824', sha256=policy['model']['sha256'], class_count=26)]
    def checkpoint(model_id, name, weights, digest, names, material_map, family):
        weights = Path(weights).resolve()
        if not weights.is_file():
            return
        if hashlib.sha256(weights.read_bytes()).hexdigest() != digest:
            raise ValueError('candidate_hash_mismatch:'+model_id)
        models.append(dict(id=model_id, name=name+' · '+backend, kind='checkpoint', weights=str(weights), device=device,
            sha256=digest, classes=names, mapping=material_map, family=family))
    for model_id, family, title in [('v8_3', 'yolov8s', 'YOLOv8s · 三类专项'), ('11s_3', 'yolo11s', 'YOLO11s · 三类专项')]:
        assets = ROOT/('local-only/'+('yolov8s' if model_id == 'v8_3' else 'yolo11s')+'-deployment/assets')
        if (assets/'deployment.json').is_file():
            lock = read(assets/'deployment.json')
            checkpoint(model_id, title, assets/'model.pt', lock['model_sha256'], lock['class_names'],
                       {'can': 'metal', 'plastic': 'plastic', 'cardboard': 'paper'}, family)
    campaign = ROOT/'local-only/public26-model-comparison-continue-20261030'
    for model_id, family, title in [('v8_26', 'yolov8s', 'YOLOv8s · 公开26类'), ('11s_26', 'yolo11s', 'YOLO11s · 公开26类')]:
        directory = campaign/('yolov8s-seg' if model_id == 'v8_26' else 'yolo11s-seg')
        if (directory/'complete.json').is_file():
            complete = read(directory/'complete.json')
            if complete.get('smoke_only') is not False:
                raise ValueError('complete_training_required:'+model_id)
            checkpoint(model_id, title, complete['best'], complete['best_sha256'], classes, mapping, family)
        else:
            models.append(dict(id=model_id, name=title, kind='unavailable', class_count=26,
                               disabled_reason='训练未完成'))
    models.append(dict(id='yoloe24', name='YOLOE-11m · 原始24类 · ROCm GPU', kind='service',
        protocol='prompted', url='http://127.0.0.1:8811', class_count=24, device='cuda:0',
        sha256='785199b7cc75a5f4041dec15782a4b8a4105e5a34d7ffb287a0a10781724c5bd'))
    latest = ROOT/'local-only/yoloe-public16-web-20261031/training/best.pt'
    latest_sha = '9d5734b34871aa81412b2fa47176d8f926283bb5adfe3a3f80cc58c5b7332534'
    if latest.is_file():
        checkpoint('yoloe26_tuned', 'YOLOE26 · 新版置信度0.10', latest, latest_sha, classes, mapping, 'yoloe')
        models[-1].update(confidence=.10, profile_name='置信度调整版', weight_version='10月31日公开素材续训best')
        bank = Path(output).resolve().parent/'reference-bank.json'
        if bank.is_file():
            checkpoint('yoloe26_reference', 'YOLOE26 · 新版参考图辅助0.10', latest, latest_sha, classes, mapping, 'yoloe')
            models[-1].update(confidence=.10, profile_name='参考图辅助版', weight_version='10月31日公开素材续训best',
                reference_bank=str(bank), reference_bank_sha256=hashlib.sha256(bank.read_bytes()).hexdigest(), reference_count=64)
    for entry in models:
        if entry['kind']=='checkpoint':entry['configuration_sha256']=configuration_hash(entry)
    config = dict(schema='switchable-console-v1', control_url='http://127.0.0.1:8810',
        bound_model='yoloe26', camera_url='http://127.0.0.1:8780/frame.jpg', worker_ports=[8831, 8832],
        device=device, worker_python=str(ROOT/('.venv-rocm/Scripts/python.exe' if device == 'cuda:0' else 'local-only/yoloe-deployment/venv/Scripts/python.exe')),
        runtime=str(ROOT/'local-only/yoloe11m-26m-comparison/runtime'), models=models)
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT/'local-only'):
        raise ValueError('catalog_must_remain_local_only')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    return [dict(id=row['id'], name=row['name'], available=row['kind'] != 'unavailable') for row in models]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'local-only/model-switch-console/models.json')
    p.add_argument('--device', choices=['cuda:0', 'cpu'], default='cuda:0')
    args = p.parse_args()
    print(json.dumps(prepare(args.output, args.device), ensure_ascii=True))


if __name__ == '__main__':
    main()
