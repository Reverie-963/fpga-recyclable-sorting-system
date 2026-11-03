"""Fine-tune YOLO11n, preserve best weights, evaluate held-out test and export ONNX."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

def main():
    from ultralytics import YOLO
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/dataset.yaml')
    p.add_argument('--weights',default='yolo11n.pt');p.add_argument('--device',default='0')
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch',type=int,default=8)
    p.add_argument('--output-dir',default='data/models',help='Directory for best.pt, ONNX and model metadata')
    p.add_argument('--imgsz',type=int,default=640);a=p.parse_args()
    if a.epochs<1 or a.batch<1:p.error('Positive epochs/batch required')
    model=YOLO(a.weights,task='detect')
    model.train(data=a.data,epochs=a.epochs,batch=a.batch,imgsz=a.imgsz,device=a.device,
                project='runs/detect',name='km1',exist_ok=False,seed=7)
    best=Path(model.trainer.best);dest=Path(a.output_dir);dest.mkdir(parents=True,exist_ok=True)
    shutil.copy2(best,dest/'best.pt');trained=YOLO(str(dest/'best.pt'))
    test=trained.val(data=a.data,split='test',imgsz=a.imgsz,device=a.device,batch=1,
                     project='runs/evaluate',name='test',exist_ok=False)
    onnx=trained.export(format='onnx',imgsz=a.imgsz,batch=1,dynamic=False,opset=17,simplify=False,device='cpu')
    report={'architecture':'YOLO11n','source_weights':a.weights,'labels':trained.names,
            'precision':'FP32','imgsz':a.imgsz,'sha256':hashlib.sha256((dest/'best.pt').read_bytes()).hexdigest(),
            'test_metrics':{k:float(v) for k,v in test.results_dict.items()},'speed':test.speed,'onnx':str(onnx)}
    (dest/'model.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
