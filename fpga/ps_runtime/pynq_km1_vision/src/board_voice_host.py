"""ASRPRO on PS, routed to the same authenticated board operation state."""
import argparse
import json
from pathlib import Path
import sys
import urllib.request
from .km1_voice_dashboard import DashboardVoiceClient,VoiceCommandNotSubmitted


class SingleAttemptVoiceClient(DashboardVoiceClient):
    def _command(self,action,*a,**kw):
        if action in ('arm','voice_arm','home'):
            raise VoiceCommandNotSubmitted('voice_supports_one_explicit_start_only')
        if action in ('start','preview'):
            try:current=self._status()
            except (OSError,ValueError,KeyError) as exc:
                raise VoiceCommandNotSubmitted('voice_status_unavailable_before_start') from exc
            if current.get('busy') or current.get('blockers'):
                raise VoiceCommandNotSubmitted('voice_station_not_ready:'+','.join(current.get('blockers') or ['busy']))
            binding=current.get('execution_model')
            if not isinstance(binding,dict) or not binding.get('profile_sha256') or type(binding.get('revision')) is not int:
                raise VoiceCommandNotSubmitted('voice_execution_model_binding_unavailable')
            kw['execution_profile_sha256']=binding['profile_sha256']
            kw['execution_binding_revision']=binding['revision']
            registration=current.get('voice_height_registration')
            if (current.get('voice_height_registration_supported') is not True or
                not isinstance(registration,dict) or not registration.get('id') or
                registration.get('remaining_s',0)<=0):
                raise VoiceCommandNotSubmitted('voice_requires_page_height_registration')
            if (registration.get('profile_sha256')!=binding['profile_sha256'] or
                registration.get('revision')!=binding['revision']):
                raise VoiceCommandNotSubmitted('voice_height_registration_model_changed')
            kw['height_registration_id']=registration['id']
        return super()._command(action,*a,**kw)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--control-token-file',type=Path,required=True)
    parser.add_argument('--controller-port',type=int,default=18810)
    parser.add_argument('--voice-port',required=True)
    parser.add_argument('--state-root',type=Path)
    args=parser.parse_args()
    state_root=(args.state_root or args.root/'state').resolve()
    from . import km1_voice_live
    token=args.control_token_file.read_text().strip()
    endpoint='http://127.0.0.1:'+str(args.controller_port)
    class Client(SingleAttemptVoiceClient):
        def _status(self):
            result=self._request('/operation')
            if result.get('compute_location')!='pynq_ps' or result.get('control_version') not in (1,3):
                raise ValueError('native_voice_controller_identity_changed')
            if self.token is not None and result.get('token')!=self.token:
                raise ValueError('dashboard_restarted_voice_reconnect_required')
            self.token=result['token'];return result
        def _request(self,route,body=None):
            request=urllib.request.Request(self.endpoint+route,
                data=json.dumps(body).encode() if body is not None else None,
                headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
            with self.opener.open(request,timeout=5) as response:raw=response.read(2_000_001)
            if len(raw)>2_000_000:raise ValueError('oversized_board_voice_reply')
            return json.loads(raw)
    base=km1_voice_live.LiveControl
    class Control(base):
        def __init__(self,root,record=lambda *a,**k:None,**kwargs):
            super().__init__(root,record,candidates=lambda:[endpoint],client_factory=Client)
        def persist_stop(self):
            config,c=self.configuration()
            for value in c['known']['stop_files']:
                path=(config.parent/value).resolve()
                if not path.is_relative_to(state_root):raise ValueError('board_voice_stop_path_escape')
                path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('a',encoding='utf-8') as stream:stream.write('ASRPRO operator stop; physical review required.\n')
            # Also latch the board owner when the HTTP controller is reachable.
            try:
                client=Client(endpoint,None)
                client._command('stop')
            except (OSError,ValueError,KeyError):pass
    km1_voice_live.LiveControl=Control
    import serial
    original_serial=serial.Serial
    def connected(value,device):
        try:
            request=urllib.request.Request(endpoint+'/voice-transport',
                data=json.dumps(dict(connected=value,device=device)).encode(),
                headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=3) as response:response.read(4096)
        except (OSError,ValueError):pass
    class VoiceSerial(original_serial):
        def open(self):
            result=super().open();connected(True,self.port);return result
        def close(self):
            was_open=getattr(self,'is_open',False)
            result=super().close()
            if was_open:connected(False,self.port)
            return result
    serial.Serial=VoiceSerial
    sys.argv=['board_voice_host','--root',str(args.root),'--port',args.voice_port,
              '--log',str(state_root/'voice-events.jsonl')]
    km1_voice_live.main()


if __name__=='__main__':main()
