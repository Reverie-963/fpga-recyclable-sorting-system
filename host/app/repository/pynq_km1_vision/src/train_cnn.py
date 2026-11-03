"""Small CNN training/export on a PC; separate-session validation; no PYNQ torch dependency."""
import argparse,copy,json,random
from pathlib import Path
import cv2,numpy as np
from .inference import crop_tensor

def main():
    import torch
    from torch import nn
    from torch.utils.data import Dataset,DataLoader
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/crops');p.add_argument('--output',default='data/classifier_cnn.onnx')
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--batch',type=int,default=32);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=p.parse_args()
    if a.epochs<1 or a.batch<1:p.error('epochs/batch must be positive')
    if a.device=='cuda' and not torch.cuda.is_available():p.error('CUDA unavailable; install matching PyTorch or use --device cpu')
    random.seed(7);np.random.seed(7);torch.manual_seed(7)
    root=Path(a.data);labels=sorted(d.name for d in (root/'train').iterdir() if d.is_dir())
    if len(labels)<2:raise ValueError('Need at least two classes')
    class Images(Dataset):
        def __init__(self,split):
            self.items=[]
            for idx,label in enumerate(labels):
                paths=sorted(f for f in (root/split/label).glob('*') if f.suffix.lower() in ['.png','.jpg','.jpeg'])
                if not paths:raise ValueError('Missing '+split+'/'+label)
                self.items.extend((f,idx) for f in paths)
        def __len__(self):return len(self.items)
        def __getitem__(self,i):
            path,label=self.items[i];im=cv2.imread(str(path))
            if im is None:raise ValueError('Unreadable image '+str(path))
            return torch.from_numpy(crop_tensor(im,64)[0]),label
    train=Images('train');val=Images('val')
    loader=DataLoader(train,batch_size=a.batch,shuffle=True,num_workers=0)
    validation=DataLoader(val,batch_size=a.batch,num_workers=0)
    model=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
        nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
        nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Linear(64,len(labels))).to(a.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001);criterion=nn.CrossEntropyLoss();best=-1.;best_state=None;history=[]
    for epoch in range(a.epochs):
        model.train();total_loss=0.
        for x,y in loader:
            x=x.to(a.device);y=y.to(a.device);optimizer.zero_grad();loss=criterion(model(x),y);loss.backward();optimizer.step();total_loss+=loss.item()*len(x)
        model.eval();correct=0
        with torch.no_grad():
            for x,y in validation:correct+=int((model(x.to(a.device)).argmax(1).cpu()==y).sum())
        accuracy=correct/len(val);history.append({'epoch':epoch+1,'train_loss':total_loss/len(train),'val_accuracy':accuracy})
        print(json.dumps(history[-1]),flush=True)
        if accuracy>best:best=accuracy;best_state=copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state);model=model.cpu().eval();output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    import inspect
    kwargs={'dynamo':False} if 'dynamo' in inspect.signature(torch.onnx.export).parameters else {}
    torch.onnx.export(model,torch.zeros(1,3,64,64),str(output),input_names=['image'],output_names=['logits'],opset_version=17,**kwargs)
    output.with_suffix('.meta.json').write_text(json.dumps({'labels':labels,'image_size':64,'preprocess':'rgb_nchw_0_1','output':'logits'},indent=2))
    # Compare CPU PyTorch and ONNX outputs on one held-out input. NPU validation is separate.
    import onnxruntime as ort
    session=ort.InferenceSession(str(output),providers=['CPUExecutionProvider']);sample=val[0][0][None]
    with torch.no_grad():expected=model(sample).numpy()
    actual=session.run(None,{'image':sample.numpy()})[0]
    np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=1e-5)
    output.with_suffix('.metrics.json').write_text(json.dumps({'history':history,'best_val_accuracy':best,'train_count':len(train),'val_count':len(val),
        'export_cpu_max_abs_error':float(np.max(np.abs(actual-expected))),'warning':'Validation used for checkpoint selection; collect an independent final test set.'},indent=2))
    print('Saved:',output)
if __name__=='__main__':main()
