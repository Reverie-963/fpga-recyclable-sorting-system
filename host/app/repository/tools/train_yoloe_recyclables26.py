"""Prepare and train a separate YOLOE-11m 26-class candidate; never deploy it."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['fuse','train'],required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--device',default='0')
    p.add_argument('--name',default='public26-linear')
    p.add_argument('--amp',action='store_true',help='Use AMP only after the separate training integration check passes')
    args=p.parse_args()
    dataset=args.dataset.resolve();out=args.output.resolve()
    if not out.is_relative_to(ROOT/'local-only'):raise ValueError('Use local-only output')
    out.mkdir(parents=True,exist_ok=True)
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
        os.environ[key]='1' if key=='OPENBLAS_NUM_THREADS' else '4'
    os.environ['YOLO_AUTOINSTALL']='false'
    os.environ['YOLO_CONFIG_DIR']=str(out/'settings')
    sys.path.insert(0,str(ROOT/'local-only/yoloe11m-26m-comparison/runtime'))
    sys.path.insert(0,str(ROOT/'pynq_km1_vision'))
    import torch
    import ultralytics
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer
    torch.set_num_threads(4)
    if ultralytics.__version__!='8.4.193':raise ValueError('Use private pinned Ultralytics 8.4.193')
    taxonomy=json.loads((dataset/'taxonomy.json').read_text())
    names=[r['official_label'] for r in taxonomy['classes']]
    if len(names)!=26:raise ValueError('Expected 26 classes')
    base=out/'yoloe-11m-fused26-base.pt'
    if args.stage=='fuse':
        if base.exists():raise FileExistsError(base)
        original=ROOT/'local-only/yoloe-deployment/assets/yoloe-11m-seg.pt'
        if digest(original)!='785199b7cc75a5f4041dec15782a4b8a4105e5a34d7ffb287a0a10781724c5bd':
            raise ValueError('Official base changed')
        os.chdir(original.parent)
        model=YOLOE(str(original)).to('cpu')
        with torch.inference_mode():
            embedding=model.get_text_pe(names)
            model.set_classes(names,embedding)
            model.model.eval()
            model.model.model[-1].fuse(model.model.pe)
        assert model.model.model[-1].is_fused
        assert list(model.names.values())==names
        model.save(str(base))
        save(out/'base.json',dict(source_sha256=digest(original),sha256=digest(base),names=names,
             torch=torch.__version__,ultralytics=ultralytics.__version__,trained=False,
             note='Text fused into a fixed 26-class head; arbitrary new prompts are not supported by this candidate'))
        print('Prepared fixed 26-class base without training',flush=True)
        return
    report=json.loads((dataset/'report.json').read_text())
    if not report['all_26_have_train_examples']:raise ValueError('Missing training examples for a class')
    manifest=json.loads((dataset/'manifest.json').read_text())
    for row in manifest['images']:
        image=dataset/row['image']
        label=dataset/'labels'/row['split']/Path(row['image']).with_suffix('.txt').name
        if digest(image)!=row['image_sha256'] or digest(label)!=row['label_sha256']:
            raise ValueError('Dataset image or label changed: '+row['image'])
    model=YOLOE(str(base))
    head_index=len(model.model.model)-1
    # Official linear probing: only three final classification projections learn.
    # Freeze the pretrained visual, box and mask features to limit forgetting and
    # avoid fitting rough community polygons with a changed segmentation head.
    train_prefixes=[f'model.{head_index}.cv3.{i}.2.' for i in range(3)]
    freeze=list(range(head_index))
    for child,_ in model.model.model[-1].named_children():
        if child!='cv3':freeze.append(f'{head_index}.{child}')
    for tower in range(3):
        for layer in range(2):freeze.append(f'{head_index}.cv3.{tower}.{layer}')
    actual=[name for name,_ in model.model.named_parameters() if any(name.startswith(x) for x in train_prefixes)]
    if len(actual)!=6:raise ValueError('Unexpected classification projection structure: '+repr(actual))
    torch.backends.cudnn.enabled=False
    if args.device!='cpu' and (not torch.cuda.is_available() or not torch.version.hip):
        raise ValueError('ROCm GPU unavailable')
    from src.yoloe_training_runtime import install_cpu_target_assignment
    install_cpu_target_assignment()
    configuration=dict(data=str(dataset/'data.yaml'),epochs=args.epochs,batch=args.batch,imgsz=640,
        device=args.device,workers=0,amp=args.amp,cache=False,optimizer='AdamW',lr0=.001,
        lrf=.1,weight_decay=.0005,patience=4,warmup_epochs=1,warmup_bias_lr=0.,
        seed=20261030,deterministic=True,mosaic=0.,mixup=0.,copy_paste=0.,
        degrees=15.,translate=.08,scale=.25,fliplr=.5,flipud=.1,
        project=str(out),name=args.name,exist_ok=False,plots=False,
        save=True,save_period=1,val=True,freeze=freeze)
    hashes={str(p.relative_to(dataset)):digest(p) for p in [dataset/'data.yaml',dataset/'taxonomy.json',dataset/'manifest.json']}
    plan=dict(config=configuration,base_sha256=digest(base),dataset_hashes=hashes,
        trainable_projection_parameters=actual,torch=torch.__version__,ultralytics=ultralytics.__version__,
        gpu=torch.cuda.get_device_name(0) if args.device!='cpu' else 'CPU',hip=torch.version.hip,
        taxonomy_classes=names,trained_output_is_fixed_taxonomy=True,deploy=False,
        target_assignment='original upstream algorithm on CPU; model/optimizer remain on selected device',
        public_report=report,created_at=time.time())
    planpath=out/(args.name+'-plan.json')
    if planpath.exists():raise FileExistsError(planpath)
    save(planpath,plan)
    def assert_training_scope(trainer):
        learned=[n for n,p in trainer.model.named_parameters() if p.requires_grad]
        if sorted(learned)!=sorted(actual):raise ValueError('Training scope changed: '+repr(learned))
        save(out/(args.name+'-actual-trainable.json'),dict(parameters=learned,
            count=sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)))
    model.add_callback('on_train_start',assert_training_scope)
    def progress(trainer):
        save(out/(args.name+'-progress.json'),dict(epoch=trainer.epoch+1,epochs=trainer.epochs,
             metrics={k:float(v) for k,v in trainer.metrics.items()},updated_at=time.time()))
    model.add_callback('on_fit_epoch_end',progress)
    results=model.train(trainer=YOLOEPESegTrainer,**configuration)
    after={str(p.relative_to(dataset)):digest(p) for p in [dataset/'data.yaml',dataset/'taxonomy.json',dataset/'manifest.json']}
    assert hashes==after
    save(out/(args.name+'-complete.json'),dict(completed_at=time.time(),
        metrics={k:float(v) for k,v in results.results_dict.items()},
        best=str(model.trainer.best),best_sha256=digest(model.trainer.best),
        last=str(model.trainer.last),last_sha256=digest(model.trainer.last),
        dataset_metadata_unchanged=True,deployed=False))


if __name__=='__main__':main()
