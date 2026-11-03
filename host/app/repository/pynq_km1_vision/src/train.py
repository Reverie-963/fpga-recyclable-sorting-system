"""Train an actual ridge image classifier using explicit train/val session splits."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from .vision import feature,Classifier

def load(folder,labels):
    xs=[];ys=[]
    for i,label in enumerate(labels):
        files=sorted((folder/label).glob('*'))
        for f in files:
            im=cv2.imread(str(f))
            if im is not None: xs.append(feature(im));ys.append(i)
    if not xs: raise ValueError('No crop images in '+str(folder))
    return np.asarray(xs),np.asarray(ys)

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/crops');p.add_argument('--output',default='data/classifier.npz')
    p.add_argument('--ridge',type=float,default=10);a=p.parse_args();base=Path(a.data)
    labels=sorted(f.name for f in (base/'train').iterdir() if f.is_dir())
    if len(labels)<2: raise ValueError('Need at least two trained classes')
    x,y=load(base/'train',labels);vx,vy=load(base/'val',labels)
    if set(y)!=set(range(len(labels))) or set(vy)!=set(range(len(labels))): raise ValueError('Each class needs train and validation images')
    if a.ridge<=0: raise ValueError('ridge must be positive')
    mean=x.mean(0);scale=np.maximum(x.std(0),.05);z=np.c_[(x-mean)/scale,np.ones(len(x))]
    target=np.eye(len(labels))[y]
    # Dual solution is cheaper than 770x770 when the training set is small.
    if len(x)<z.shape[1]: w=z.T@np.linalg.solve(z@z.T+a.ridge*np.eye(len(x)),target)
    else: w=np.linalg.solve(z.T@z+a.ridge*np.eye(z.shape[1]),z.T@target)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    np.savez(a.output,mean=mean,scale=scale,weights=w,labels=np.asarray(labels))
    scores=np.c_[(vx-mean)/scale,np.ones(len(vx))]@w;pred=scores.argmax(1)
    cm=np.zeros((len(labels),len(labels)),int)
    for actual,guess in zip(vy,pred): cm[actual,guess]+=1
    margins=np.sort(scores,axis=1)[:,-1]-np.sort(scores,axis=1)[:,-2]
    report={'labels':labels,'train_count':len(x),'val_count':len(vx),'accuracy':float((pred==vy).mean()),
        'confusion_matrix_actual_rows':cm.tolist(),'margins':margins.tolist(),'correct':(pred==vy).tolist(),
        'note':'Validation is held out by directory; ensure separate capture sessions. Ridge margin is not probability.'}
    Path(a.output).with_suffix('.metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
