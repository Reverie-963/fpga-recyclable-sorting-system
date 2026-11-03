"""Download and hash-check the official YOLOE playground assets."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
RELEASE='https://github.com/ultralytics/assets/releases/download/v8.3.0/'
FILES=[
    ('local-only/yoloe-official/assets/yoloe-11m-seg-pf.pt', RELEASE+'yoloe-11m-seg-pf.pt',
     '0e31ac208a31b591c20c69f6e525110cc9e96a7dbc1dc2e441ae46597205bbfd'),
    ('local-only/yoloe-deployment/assets/yoloe-11m-seg.pt', RELEASE+'yoloe-11m-seg.pt',
     '785199b7cc75a5f4041dec15782a4b8a4105e5a34d7ffb287a0a10781724c5bd'),
    ('local-only/yoloe-deployment/assets/mobileclip_blt.ts', RELEASE+'mobileclip_blt.ts',
     'a67804d1b0f07b8b9a20c1761ec0847f34660f5fa338ec70e8f3fce68ed95e54'),
    ('local-only/yoloe-official/assets/bus.jpg',
     'https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg',
     'c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63'),
]


def file_hash(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        while chunk:=stream.read(1024*1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    records=[]
    for relative,url,expected in FILES:
        path=ROOT/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            temporary=path.with_suffix(path.suffix+'.download')
            print('Downloading',path.name,flush=True)
            with urllib.request.urlopen(url,timeout=60) as source,temporary.open('wb') as target:
                while chunk:=source.read(1024*1024):
                    target.write(chunk)
            if file_hash(temporary)!=expected:
                raise ValueError('Downloaded hash mismatch: '+relative)
            temporary.replace(path)
        actual=file_hash(path)
        if actual!=expected:
            raise ValueError('Existing asset differs; preserve it before replacing: '+relative)
        records.append(dict(path=relative,url=url,sha256=actual,bytes=path.stat().st_size))
        print('Verified',path.name,flush=True)
    (ROOT/'local-only/yoloe-official/assets/deployment.json').write_text(
        json.dumps(dict(official_assets=records,trained=False),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
