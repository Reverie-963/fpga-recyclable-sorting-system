"""Thin local front end to the board's control state and existing PC vision."""
import argparse
import importlib.util
import json
import threading
import urllib.request
from pathlib import Path
from board_station_links import PinnedBoardHTTP,make_control_proxy,ReverseVisualLink


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--port',type=int,default=8810)
    parser.add_argument('--voice-dimensions-port',type=int,default=18943)
    parser.add_argument('--model-catalog',type=Path,
        help='Optional local model catalog; native workbench and PS controls remain authoritative.')
    args=parser.parse_args()
    config=json.loads(args.config.read_text(encoding='utf-8'))
    if config['mode']!='board_station':raise ValueError('native_board_station_config_required')
    spec=importlib.util.spec_from_file_location('private_connection',config['connection_module'])
    private=importlib.util.module_from_spec(spec);spec.loader.exec_module(private)
    client=private.connect();control_client=None;model_selection=None;dimensions=None;reverse=None
    try:
        # A separate pinned SSH session keeps images/page polling off the
        # heartbeat/stop path. Neither session reconnects or replays requests.
        control_client=private.connect()
        board=PinnedBoardHTTP(client,config['board_port'],Path(config['control_token_file']).read_text().strip(),
                              control_client=control_client)
        status,raw,_=board.request('/board-control')
        if status!=200 or json.loads(raw).get('compute_location')!='pynq_ps':
            raise ValueError('board_station_identity_unavailable')
        server=make_control_proxy(args.port,board,config['visual_console_url'])
        catalog=args.model_catalog or Path(__file__).resolve().parents[1]/'local-only/model-switch-console/models.json'
        if catalog.is_file():
            from switchable_console import attach_to_workbench
            model_selection=attach_to_workbench(server,board,catalog.resolve())
        from voice_dimensions import Dimensions,bound_layout,make_server,fresh_observation
        from item_profile_store import ItemProfileStore
        # This authenticated reverse link only supplies measured dimensions.
        # It cannot submit, repeat or keep a robot operation alive.
        _,raw,_=board.request('/operation');current=json.loads(raw)
        root=current['board_root'];layout=bound_layout(client,root)
        from console_pickup_scope import PickupScope,attach
        def read_board_status():
            code,raw,_=board.request('/operation')
            if code!=200:raise ValueError('pickup_controller_unavailable')
            return json.loads(raw)
        attach(server,PickupScope(layout,root,read_board_status))
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        def read_local(route,payload=None):
            request=urllib.request.Request('http://127.0.0.1:'+str(args.port)+route,
                data=json.dumps(payload).encode() if payload is not None else None,headers={'Content-Type':'application/json'})
            with opener.open(request,timeout=6) as response:raw=response.read(18_000_001)
            if len(raw)>18_000_000:raise ValueError('voice_dimensions_reply_too_large')
            return json.loads(raw)
        store=ItemProfileStore(Path(__file__).resolve().parents[1]/'local-only/item-profiles/profiles.json')
        dimensions=make_server(args.voice_dimensions_port,Path(config['control_token_file']).read_text().strip(),Dimensions(read_local,store,layout,root,
            observation_loader=lambda:fresh_observation(lambda payload:read_local('/detect',payload))))
        threading.Thread(target=dimensions.serve_forever,daemon=True).start()
        reverse=ReverseVisualLink(client,args.voice_dimensions_port);reverse.start()
        print(json.dumps(dict(status='native_board_proxy_ready',port=args.port,compute_location='pynq_ps',
                              transport_contract='isolated-control-v1')),flush=True)
        try:server.serve_forever()
        finally:server.server_close()
    finally:
        if reverse is not None:reverse.close()
        if dimensions is not None:dimensions.shutdown();dimensions.server_close()
        if model_selection is not None:model_selection.close()
        if control_client is not None:control_client.close()
        client.close()


if __name__=='__main__':main()
