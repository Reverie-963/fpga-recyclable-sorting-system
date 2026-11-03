"""Fit planar mapping from measured pairs; reject poor fits and require independent checks."""
import argparse,json
from pathlib import Path
import cv2,numpy as np

def fit(data):
    pixels=np.asarray(data['pixels'],np.float64);world=np.asarray(data['world_mm'],np.float64)
    if pixels.shape!=world.shape or pixels.ndim!=2 or pixels.shape[1]!=2 or len(pixels)<6:
        raise ValueError('Need >=6 paired points spread across working plane')
    if not np.isfinite(pixels).all() or not np.isfinite(world).all(): raise ValueError('Nonfinite coordinates')
    hull=cv2.convexHull(pixels.astype('float32'))
    if cv2.contourArea(hull)<100: raise ValueError('Degenerate calibration coverage')
    H,mask=cv2.findHomography(pixels,world,0)
    if H is None or not np.isfinite(H).all(): raise ValueError('Cannot fit homography')
    def error(px,xy):
        px=np.asarray(px,np.float64);xy=np.asarray(xy,np.float64)
        if px.shape!=xy.shape or px.ndim!=2 or px.shape[1]!=2: raise ValueError('Invalid validation pairs')
        pred=cv2.perspectiveTransform(px[None],H)[0]
        return np.linalg.norm(pred-xy,axis=1)
    err=error(pixels,world)
    vp=data['check_pixels'];vw=data['check_world_mm']
    if len(vp)<3: raise ValueError('Need >=3 separate check points')
    ve=error(vp,vw)
    if not np.isfinite(ve).all() or max(err.max(),ve.max())>data.get('max_error_mm',3):
        raise ValueError('Calibration error exceeds tolerance')
    return {'homography':H.tolist(),'image_size':data['image_size'],'pixel_hull':hull.reshape(-1,2).tolist(),
            'fit_max_mm':float(err.max()),'check_max_mm':float(ve.max()),'plane_note':data['plane_note']}

def main():
    p=argparse.ArgumentParser();p.add_argument('pairs');p.add_argument('--output',default='data/calibration.json');a=p.parse_args()
    result=fit(json.loads(Path(a.pairs).read_text(encoding='utf-8-sig')));Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__': main()
