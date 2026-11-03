"""PC console client for the explicitly selected board-preview gateway."""
import json
from pathlib import Path
import urllib.error
import urllib.request

from .board_detector_transport import local_url,token_value


class BoardPreviewClient:
    def __init__(self, config, allowed_root):
        self.path=Path(config).resolve()
        self.root=Path(allowed_root).resolve()
        self.url,self.token=self.target()
        self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def target(self):
        path=self.path.resolve()
        if not path.is_relative_to(self.root):raise ValueError('board_gateway_config_outside_local_only')
        record=json.loads(path.read_text(encoding='utf-8'))
        if record.get('mode')!='preview_only':raise ValueError('board_gateway_must_be_preview_only')
        url=local_url(record['url'])
        token=(path.parent/record['token_file']).resolve()
        if not token.is_relative_to(self.root):raise ValueError('board_gateway_token_outside_local_only')
        return url,token_value(token.read_text().strip())

    def request(self,path,data=None):
        if path not in ('/health','/plan'):raise ValueError('board_gateway_has_no_motion_endpoint')
        # Each new user request reads the explicitly published target. An
        # in-flight request never retries or migrates to a replacement service.
        url,token=self.target()
        call=urllib.request.Request(url+path,data=json.dumps(data).encode() if data is not None else None,
            headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
        try:response=self.opener.open(call,timeout=95 if path=='/plan' else 3)
        except urllib.error.HTTPError as exc:response=exc
        with response:
            raw=response.read(8_500_001)
            if len(raw)>8_500_000:raise ValueError('oversized_board_gateway_reply')
            value=json.loads(raw)
            if response.status>=400:raise ValueError(value.get('error','board_preview_unavailable'))
        return value

    def health(self):return self.request('/health')
    def plan(self,payload):return self.request('/plan',payload)
