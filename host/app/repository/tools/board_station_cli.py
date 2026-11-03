"""Thin client: task admission, planning and execution remain on PYNQ."""
import argparse
import json
import time
import urllib.request
from urllib.parse import urlsplit
import uuid


def main():
    p=argparse.ArgumentParser(description=__doc__)
    actions=p.add_mutually_exclusive_group(required=True)
    for name in ('status','preview','execute','stop','reset','home'):actions.add_argument('--'+name,action='store_true')
    p.add_argument('--url',default='http://127.0.0.1:8810')
    p.add_argument('--material',choices=['AUTO','PAPER','PLASTIC','METAL'],default='AUTO')
    p.add_argument('--height',type=float)
    p.add_argument('--ready',action='store_true',help='Operator confirms current physical scene; required for execute/reset/home.')
    args=p.parse_args();parsed=urlsplit(args.url)
    if (parsed.scheme!='http' or parsed.hostname not in ('127.0.0.1','localhost') or
        parsed.path not in ('','/') or parsed.query or parsed.fragment or parsed.username or parsed.password):
        raise ValueError('local_native_proxy_url_required')
    if (args.execute or args.reset or args.home) and not args.ready:raise ValueError('explicit_current_scene_ready_required')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def call(path,body=None):
        req=urllib.request.Request(args.url.rstrip('/')+path,
            data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'})
        with opener.open(req,timeout=8) as reply:return json.load(reply)
    status=call('/operation')
    if status.get('compute_location')!='pynq_ps':raise ValueError('native_board_control_not_active')
    if args.status:
        status.pop('token',None);print(json.dumps(status,ensure_ascii=False));return
    action=('start' if args.execute else 'preview' if args.preview else 'stop' if args.stop else 'reset' if args.reset else 'home')
    owner='cli-'+uuid.uuid4().hex
    payload=dict(action=action,owner=owner,token=status['token'],request_id=uuid.uuid4().hex,
                 material=args.material,ready=args.ready,reset_confirmed=args.ready,home_confirmed=args.ready)
    if args.height is not None:payload['object_height_mm']=args.height
    try:result=call('/operation',payload)
    except BaseException:
        # A lost admission response is unknown, never permission to submit again.
        try:call('/operation',dict(action='stop',token=status['token'],owner=owner))
        except Exception:pass
        raise
    if action in ('stop','reset'):
        result.pop('token',None);print(json.dumps(result,ensure_ascii=False));return
    operation_id=result['operation_id'];deadline=time.monotonic()+300;last=0
    try:
        while time.monotonic()<deadline:
            status=call('/operation')
            if status.get('operation_id')!=operation_id:raise ValueError('board_operation_replaced_no_retry')
            if not status.get('busy'):
                status.pop('token',None);print(json.dumps(status,ensure_ascii=False));return
            if time.monotonic()-last>=2:
                call('/operation',dict(action='heartbeat',token=status['token'],owner=owner,operation_id=operation_id))
                last=time.monotonic()
            time.sleep(.25)
        raise TimeoutError('board_operation_deadline_no_retry')
    except BaseException:
        try:call('/operation',dict(action='stop',token=status['token'],owner=owner))
        except Exception:pass
        raise


if __name__=='__main__':main()
