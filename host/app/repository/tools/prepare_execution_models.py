"""Build frozen execution identities from a registered local model catalog."""
import argparse
import json
from pathlib import Path
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'pynq_km1_vision'))
from src.execution_models import build_profiles


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def prepare(catalog,base_policy,base_settings,base_receipt):
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));services={}
    for row in catalog['models']:
        if row['kind']!='service' or row['id']==catalog['bound_model']:continue
        values={}
        for route in ('policy','health'):
            with opener.open(row['url']+'/'+route,timeout=8) as response:values[route]=json.load(response)
        services[row['id']]=dict(policy=values['policy'],settings=values['health']['settings'])
    return build_profiles(catalog,base_policy,base_settings,base_receipt,services)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('catalog','base-policy','base-settings','base-receipt','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    result=prepare(read(args.catalog),read(args.base_policy),read(args.base_settings),read(args.base_receipt))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(profiles=[dict(id=p['id'],classes=len(p['policy']['classes']),confidence=p['settings']['confidence'],
        model_sha256=p['model_sha256'],profile_sha256=p['profile_sha256']) for p in result['profiles']]),ensure_ascii=False))


if __name__=='__main__':main()
