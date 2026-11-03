import cv2
import numpy as np


def mask_cpu(bgr,bg_bgr,threshold):
    diff=np.abs(bgr.astype(np.int16)-np.asarray(bg_bgr,dtype=np.int16))
    return (diff.sum(axis=2)>threshold).astype(np.uint8)*255


def open_mask(mask):
    return cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)


def candidates(mask,min_area,max_area,already_open=False):
    # Same PS postprocessing for BOTH backends.
    if not already_open:mask=open_mask(mask)
    n,labels,stats,centers=cv2.connectedComponentsWithStats(mask,8)
    found=[]
    for i in range(1,n):
        x,y,w,h,area=map(int,stats[i])
        if min_area <= area <= max_area and x>0 and y>0 and x+w<mask.shape[1] and y+h<mask.shape[0]:
            found.append({'bbox':[x,y,w,h],'center':centers[i].tolist(),'area':area})
    return sorted(found,key=lambda o:-o['area'])


def feature(crop):
    if crop.size==0: raise ValueError('Empty crop')
    img=cv2.resize(crop,(16,16),interpolation=cv2.INTER_AREA).astype(np.float32)/255
    # Lightweight learned classifier input: RGB spatial texture + aspect ratio.
    return np.r_[img[:,:,::-1].reshape(-1),np.clip(crop.shape[1]/crop.shape[0],0,5)].astype(np.float32)


class Classifier:
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as m:
            self.mean=m['mean'];self.scale=m['scale'];self.weights=m['weights'];self.labels=m['labels'].tolist()
    def predict(self,crop):
        f=(feature(crop)-self.mean)/self.scale
        scores=np.r_[f,1.]@self.weights
        order=np.argsort(scores)
        # Ridge scores are NOT probabilities. Margin threshold needs validation data.
        return self.labels[order[-1]],float(scores[order[-1]]-scores[order[-2]])


def map_point(point,cal):
    p=np.asarray(cal['homography'],dtype=float)@np.r_[point,1.]
    if not np.isfinite(p).all() or abs(p[2])<1e-9: raise ValueError('Invalid homography projection')
    return (p[:2]/p[2]).tolist()


class StableTarget:
    def __init__(self,frames,tolerance): self.frames=frames;self.tolerance=tolerance;self.reset()
    def reset(self): self.anchor=None;self.count=0
    def update(self,obj):
        if obj is None: self.reset();return False
        key=(obj['label'],np.asarray(obj['center']))
        if self.anchor is None or key[0]!=self.anchor[0] or np.linalg.norm(key[1]-self.anchor[1])>self.tolerance:
            self.anchor=key;self.count=1
        else: self.count+=1
        return self.count>=self.frames
