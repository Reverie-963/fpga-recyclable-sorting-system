"""Cache frozen YOLOE features once, then fit its 26-class linear projections.

The original TaskAlignedAssigner and normalized BCE classification objective are
used on CPU. Frozen box/mask terms have no gradient to these projections and are
omitted. Fixed input views replace per-epoch stochastic image augmentation.
All checkpoints remain ordinary YOLOE segmentation models with a fixed taxonomy.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    digest=hashlib.sha256()
    with Path(p).open('rb') as stream:
        while chunk:=stream.read(1024*1024):digest.update(chunk)
    return digest.hexdigest()
def save(p,v):
    p=Path(p);temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');temp.replace(p)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['cache','train'],required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=12)
    parser.add_argument('--batch',type=int,default=4)
    parser.add_argument('--limit',type=int,default=0,help='Functional cache check only; not a complete training set')
    args=parser.parse_args();out=args.output.resolve();dataset=args.dataset.resolve();base=args.base.resolve()
    if args.epochs<1 or args.batch<1 or args.limit<0:raise ValueError('Invalid epochs/batch/limit')
    if not out.is_relative_to(ROOT/'local-only'):raise ValueError('Use local-only output')
    out.mkdir(parents=True,exist_ok=True)
    os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        YOLO_AUTOINSTALL='false',YOLO_CONFIG_DIR=str(out/'settings'))
    sys.path.insert(0,str(ROOT/'local-only/yoloe11m-26m-comparison/runtime'))
    import cv2
    import numpy as np
    import torch
    from torch import nn
    from ultralytics import YOLOE
    from ultralytics.engine.model import Model as EngineModel
    from ultralytics.models.yolo.segment import SegmentationValidator
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.utils.tal import TaskAlignedAssigner,make_anchors
    torch.set_num_threads(4);torch.backends.cudnn.enabled=False
    if not torch.cuda.is_available() or not torch.version.hip:raise ValueError('ROCm GPU required for extraction/validation')
    torch.manual_seed(20261030);random.seed(20261030)
    manifest=read(dataset/'manifest.json');names=manifest['classes']
    rows=[r for r in manifest['images'] if r['split']=='train']
    if len(names)!=26:raise ValueError('Expected the authorized 26 classes')
    student=YOLOE(str(base));student.model.float().eval()
    if list(student.names.values())!=names:raise ValueError('Class order changed')
    head=student.model.model[-1]
    criterion=v8DetectionLoss(student.model)
    projections=nn.ModuleList([__import__('copy').deepcopy(head.cv3[i][2]).float().cpu() for i in range(3)])
    student.model.requires_grad_(False)
    projections.requires_grad_(True)
    if sum(p.numel() for p in projections.parameters())!=20046:raise ValueError('Unexpected classifier')
    cache=out/'features';cache.mkdir(exist_ok=True)
    specification=dict(base_sha256=sha(base),dataset_manifest_sha256=sha(dataset/'manifest.json'),
       classes=names,input_size=640,letterbox_fill=114,feature_precision='fp16',
       target_boxes='source polygon bounds transformed into the fixed letterbox',
       stochastic_per_epoch_augmentation=False,local_positive_training_images=0,
       extractor_convolutions_fused=True,cache_storage='memory-mapped fp16 features and fp32 boxes')
    if args.stage=='cache':
        if (out/'cache-complete.json').exists():raise FileExistsError('Completed cache is immutable')
        if (out/'features.fp16.dat').exists() or (out/'boxes.fp32.dat').exists():
            raise FileExistsError('Keep the interrupted cache and choose a new output directory')
        if shutil.disk_usage(out).free<16*1024**3:raise ValueError('Need at least 16 GiB free for frozen features')
        gpu=YOLOE(str(base));gpu.model.half().to('cuda:0').eval()
        gpu.model.fuse(verbose=False)
        captured={}
        hooks=[]
        for level in range(3):
            def hook(module,inputs,level=level):captured[level]=inputs[0].detach()
            hooks.append(gpu.model.model[-1].cv3[level][2].register_forward_pre_hook(hook))
        records=[];anchors=None;stride=None;max_logit_difference=0.;feature_store=None;box_store=None
        selected=rows[:args.limit] if args.limit else rows
        for offset in range(0,len(selected),args.batch):
            batch_rows=selected[offset:offset+args.batch];inputs=[];target_rows=[]
            for row in batch_rows:
                image_path=dataset/row['image'];label_path=dataset/'labels/train'/Path(row['image']).with_suffix('.txt').name
                if sha(image_path)!=row['image_sha256'] or sha(label_path)!=row['label_sha256']:
                    raise ValueError('Dataset bytes changed')
                image=cv2.imread(str(image_path));height,width=image.shape[:2]
                ratio=min(640/width,640/height);nw,nh=round(width*ratio),round(height*ratio)
                left,top=round((640-nw)/2-.1),round((640-nh)/2-.1)
                padded=np.full((640,640,3),114,np.uint8)
                padded[top:top+nh,left:left+nw]=cv2.resize(image,(nw,nh))
                inputs.append(padded[:,:,::-1].transpose(2,0,1).copy())
                targets=[]
                for line in label_path.read_text().splitlines():
                    values=[float(v) for v in line.split()];points=np.asarray(values[1:]).reshape(-1,2)*[width,height]
                    points=points*ratio+[left,top]
                    targets.append([values[0],float(points[:,0].min()),float(points[:,1].min()),
                                    float(points[:,0].max()),float(points[:,1].max())])
                target_rows.append(targets)
            tensor=torch.from_numpy(np.stack(inputs)).to('cuda:0').half()/255
            with torch.inference_mode():result=gpu.model(tensor)
            raw=result[1]
            if not isinstance(raw,dict) or not all(k in raw for k in ['boxes','scores','feats']):
                raise ValueError('Unexpected upstream head output protocol')
            features=torch.cat([captured[i].flatten(2) for i in range(3)],2).cpu().numpy()
            if anchors is None:
                spatial_shapes=[list(captured[i].shape[-2:]) for i in range(3)]
                anchors,stride=make_anchors([torch.empty((1,1,*shape)) for shape in spatial_shapes],criterion.stride,.5)
                np.savez(out/'anchors.npz',anchors=anchors.numpy(),stride=stride.numpy())
                feature_store=np.memmap(out/'features.fp16.dat',mode='w+',dtype=np.float16,
                    shape=(len(selected),features.shape[1],features.shape[2]))
                box_store=np.memmap(out/'boxes.fp32.dat',mode='w+',dtype=np.float32,
                    shape=(len(selected),features.shape[2],4))
            boxes=criterion.bbox_decode(anchors,raw['boxes'].detach().float().cpu().permute(0,2,1).contiguous())*stride
            if offset==0:
                with torch.no_grad():
                    values=[];begin=0
                    for i,shape in enumerate(spatial_shapes):
                        size=shape[0]*shape[1]
                        part=torch.from_numpy(features[:,:,begin:begin+size].copy()).float().reshape(len(batch_rows),256,*shape)
                        values.append(projections[i](part).flatten(2));begin+=size
                    reconstructed=torch.cat(values,2)
                actual=raw['scores'].detach().float().cpu()
                difference=float((reconstructed-actual).abs().max())
                if difference>.12:raise ValueError('Cached feature projection mismatch: '+str(difference))
                max_logit_difference=max(max_logit_difference,difference)
            feature_store[offset:offset+len(batch_rows)]=features
            box_store[offset:offset+len(batch_rows)]=boxes.numpy()
            for i,row in enumerate(batch_rows):
                records.append(dict(image=row['image'],feature_index=offset+i,targets=target_rows[i],
                                    image_sha256=row['image_sha256'],label_sha256=row['label_sha256']))
            completed=offset+len(batch_rows)
            if completed%100==0 or completed==len(selected):
                feature_store.flush();box_store.flush()
                feature_shape,box_shape=feature_store.shape,box_store.shape
                feature_store._mmap.close();box_store._mmap.close()
                feature_store=np.memmap(out/'features.fp16.dat',mode='r+',dtype=np.float16,shape=feature_shape)
                box_store=np.memmap(out/'boxes.fp32.dat',mode='r+',dtype=np.float32,shape=box_shape)
                save(out/'cache-progress.json',dict(completed=completed,total=len(selected),updated_at=time.time()))
                print('Cached',completed,'/',len(selected),flush=True)
        for hook in hooks:hook.remove()
        feature_store.flush();box_store.flush()
        save(out/'cache-complete.json',dict(specification=specification,records=records,
            complete_training_set=not args.limit,max_initial_logit_difference=max_logit_difference,
            spatial_shapes=spatial_shapes,feature_shape=list(feature_store.shape),box_shape=list(box_store.shape),
            arrays_sha256={name:sha(out/name) for name in ['features.fp16.dat','boxes.fp32.dat']},
            anchors_sha256=sha(out/'anchors.npz'),finished_at=time.time()))
        return

    cached=read(out/'cache-complete.json')
    if not cached['complete_training_set'] or cached['specification']!=specification:raise ValueError('Cache identity mismatch')
    if len(cached['records'])!=len(rows) or any(record['image']!=row['image'] or record['feature_index']!=i
        or record['image_sha256']!=row['image_sha256'] or record['label_sha256']!=row['label_sha256']
        for i,(record,row) in enumerate(zip(cached['records'],rows))):
        raise ValueError('Cache record order or source identity changed')
    if (out/'complete.json').exists():raise FileExistsError('Training already complete')
    if (out/'training-plan.json').exists():raise FileExistsError('Keep interrupted training record and choose a new output')
    for name,expected in cached['arrays_sha256'].items():
        if sha(out/name)!=expected:raise ValueError('Feature cache changed')
    feature_store=np.memmap(out/'features.fp16.dat',mode='r',dtype=np.float16,shape=tuple(cached['feature_shape']))
    box_store=np.memmap(out/'boxes.fp32.dat',mode='r',dtype=np.float32,shape=tuple(cached['box_shape']))
    if sha(out/'anchors.npz')!=cached['anchors_sha256']:raise ValueError('Cached anchors changed')
    anchors_file=np.load(out/'anchors.npz');anchors=torch.from_numpy(anchors_file['anchors']);stride=torch.from_numpy(anchors_file['stride'])
    matcher=TaskAlignedAssigner(topk=10,num_classes=26,alpha=.5,beta=6.,stride=criterion.stride.tolist())
    optimizer=torch.optim.AdamW(projections.parameters(),lr=.001,weight_decay=.0005)
    total_parameters=sum(p.numel() for p in projections.parameters())
    plan=dict(specification=specification,epochs=args.epochs,batch=args.batch,gradient_accumulation=16,
       optimizer='AdamW',lr0=.001,weight_decay=.0005,trainable_parameters=total_parameters,
       classification_gain=.5,target_assignment='Original upstream TaskAlignedAssigner on CPU',
       skipped_terms='Frozen box and segmentation loss terms have zero gradient to these projections',
       cache_manifest_sha256=sha(out/'cache-complete.json'),validation_split='public val only',
       checkpoint_selection='Public mask mAP50-95',patience=4,deployed=False)
    save(out/'training-plan.json',plan)
    history=[];best=-1.;best_epoch=0
    train_args=dict(student.model.args)
    train_args.update(data=str(dataset/'data.yaml'),epochs=args.epochs,task='segment',mode='train')
    student.model.args=train_args;student.ckpt['train_args']=train_args
    for epoch in range(args.epochs):
        started=time.time();order=list(range(len(cached['records'])));random.Random(20261030+epoch).shuffle(order)
        optimizer.zero_grad(set_to_none=True);losses=[]
        for step,offset in enumerate(range(0,len(order),args.batch)):
            indices=order[offset:offset+args.batch]
            features=torch.from_numpy(feature_store[indices].copy()).float()
            parts=[];begin=0
            for level,shape in enumerate(cached['spatial_shapes']):
                size=shape[0]*shape[1]
                parts.append(projections[level](features[:,:,begin:begin+size].reshape(len(indices),256,*shape)).flatten(2));begin+=size
            logits=torch.cat(parts,2).permute(0,2,1).contiguous()
            boxes=torch.from_numpy(box_store[indices].copy())
            count=max(len(cached['records'][i]['targets']) for i in indices)
            targets=torch.zeros(len(indices),count,5)
            for i,index in enumerate(indices):
                values=cached['records'][index]['targets']
                if len(values):targets[i,:len(values)]=torch.tensor(values,dtype=torch.float32)
            labels=targets[:,:,:1];gt_boxes=targets[:,:,1:];mask=gt_boxes.sum(2,keepdim=True)>0
            with torch.no_grad():
                _,_,scores,_,_=matcher(logits.detach().sigmoid(),boxes,anchors*stride,labels,gt_boxes,mask)
            loss=nn.functional.binary_cross_entropy_with_logits(logits,scores,reduction='sum')/scores.sum().clamp(min=1)
            loss=loss*.5*len(indices)
            if not torch.isfinite(loss):raise ValueError('Nonfinite classification loss')
            (loss/16).backward();losses.append(float(loss.detach()))
            if (step+1)%16==0 or offset+args.batch>=len(order):
                torch.nn.utils.clip_grad_norm_(projections.parameters(),10.)
                optimizer.step();optimizer.zero_grad(set_to_none=True)
            if step%100==0:
                save(out/'training-progress.json',dict(epoch=epoch+1,epochs=args.epochs,images=offset+len(indices),
                    total_images=len(order),mean_classification_loss=sum(losses)/len(losses),updated_at=time.time()))
                print('Epoch',epoch+1,'images',offset+len(indices),'loss',round(sum(losses)/len(losses),4),flush=True)
                # Drop mapped working-set pages regularly; tensors above own
                # copies, so this does not invalidate autograd inputs.
                feature_store._mmap.close();box_store._mmap.close()
                feature_store=np.memmap(out/'features.fp16.dat',mode='r',dtype=np.float16,shape=tuple(cached['feature_shape']))
                box_store=np.memmap(out/'boxes.fp32.dat',mode='r',dtype=np.float32,shape=tuple(cached['box_shape']))
        for i in range(3):head.cv3[i][2].load_state_dict(projections[i].state_dict())
        last=out/'last.pt';student.save(str(last))
        validation=YOLOE(str(last))
        metrics=EngineModel.val(validation,validator=SegmentationValidator,data=str(dataset/'data.yaml'),device=0,
            half=True,batch=2,imgsz=640,workers=0,plots=False,save_json=False,conf=.001,iou=.7,max_det=100,
            project=str(out/'validation'),name='epoch-%02d'%(epoch+1),verbose=False)
        values={key:float(value) for key,value in metrics.results_dict.items()}
        score=values['metrics/mAP50-95(M)']
        entry=dict(epoch=epoch+1,classification_loss=sum(losses)/len(losses),metrics=values,seconds=time.time()-started)
        history.append(entry);save(out/'history.json',history)
        if score>best:
            best=score;best_epoch=epoch+1;shutil.copyfile(last,out/'best.pt')
        save(out/'training-progress.json',dict(entry,best_epoch=best_epoch,best_public_mask_map=best,
             completed_epoch=True,updated_at=time.time()))
        print(json.dumps(entry),flush=True)
        del validation,metrics;gc.collect();torch.cuda.empty_cache()
        for group in optimizer.param_groups:group['lr']=.001*(1-.9*(epoch+1)/args.epochs)
        if epoch+1-best_epoch>=4:break
    save(out/'complete.json',dict(history=history,best_epoch=best_epoch,best_public_mask_map=best,
        best_sha256=sha(out/'best.pt'),last_sha256=sha(out/'last.pt'),
        source_base_sha256=sha(base),deployed=False,finished_at=time.time()))


if __name__=='__main__':main()
