"""Encode official YOLOE text in the separate, pinned CPU runtime."""
import json
import os
from pathlib import Path
import sys


def main():
    os.environ['OPENBLAS_NUM_THREADS']='1'
    os.environ['OMP_NUM_THREADS']='2'
    os.environ['MKL_NUM_THREADS']='2'
    os.environ['YOLO_AUTOINSTALL']='false'
    root=Path(__file__).resolve().parents[2]
    destination=Path(sys.argv[1]).resolve()
    if not destination.is_relative_to(root/'local-only/yoloe-official/text-cache'):
        raise ValueError('Text cache must remain in its dedicated directory')
    os.chdir(root/'local-only/yoloe-deployment/assets')
    import torch
    import numpy as np
    from ultralytics import YOLOE
    torch.set_num_threads(2)
    names=json.load(sys.stdin)
    model=YOLOE('yoloe-11m-seg.pt').to('cpu')
    with torch.inference_mode():
        embedding=model.get_text_pe(names).detach().cpu().float().numpy()
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.save(stream,embedding,allow_pickle=False)
    temporary.replace(destination)


if __name__=='__main__':
    main()
