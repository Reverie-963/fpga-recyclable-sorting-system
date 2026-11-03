"""Read status without requesting inference, motion or review/reset actions."""
import json
import sys
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'pynq_km1_vision'))
from src.board_control_gateway_client import BoardPreviewClient


def main():
    board=BoardPreviewClient(ROOT/'local-only/board-control-integration/active.json',ROOT/'local-only')
    health=board.health()
    print(json.dumps(dict(board=health)))
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for port in (8810,8812):
        try:
            with opener.open('http://127.0.0.1:'+str(port)+'/board-control',timeout=3) as response:
                result=json.load(response)
            print(json.dumps(dict(console_port=port,board_control=result)))
        except (OSError,ValueError):print(json.dumps(dict(console_port=port,board_control='not_selected_or_unavailable')))


if __name__=='__main__':main()
