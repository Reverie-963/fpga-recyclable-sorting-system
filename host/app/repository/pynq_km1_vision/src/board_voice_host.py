"""ASRPRO on PS, routed to the same authenticated board operation state."""
import argparse
import json
from pathlib import Path
import sys
import threading
import uuid
import urllib.request
from .km1_voice_dashboard import DashboardVoiceClient,VoiceCommandNotSubmitted


class SingleAttemptVoiceClient(DashboardVoiceClient):
    dimensions_prepare=None

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
            if self.dimensions_prepare is not None and not current.get('voice_height_registration'):
                # A new spoken command is the confirmation. Preparing and
                # registering dimensions are read/no-motion steps; start below
                # remains a single non-retried physical submission.
                try:
                    if kw.get('ready') is not True or self.cancelled.is_set():raise ValueError('voice_explicit_request_required')
                    prepared=self.dimensions_prepare(dict(request_id=kw['request_id'],material=kw['material'],
                        profile_sha256=binding['profile_sha256'],revision=binding['revision']))
                    if (prepared.get('request_id')!=kw['request_id'] or prepared.get('profile_sha256')!=binding['profile_sha256'] or
                        prepared.get('revision')!=binding['revision']):raise ValueError('voice_dimension_reply_identity_changed')
                    plan=prepared.get('height_plan')
                    if not isinstance(plan,dict) or not plan.get('items'):raise ValueError('voice_dimensions_required')
                    # The authoritative PS registration endpoint validates the
                    # complete mask/units schema, without importing a planner
                    # or opening an actuator in this voice process.
                    latest=self._status()
                    if (latest.get('busy') or latest.get('blockers') or latest.get('execution_model')!=binding or
                        self.cancelled.is_set()):raise ValueError('voice_station_changed_during_matching')
                    if latest.get('voice_height_registration'):raise ValueError('voice_page_registration_pending')
                    saved=super()._command('register_heights',request_id=uuid.uuid4().hex,ready=True,height_plan=plan)
                    registration=saved.get('voice_height_registration')
                    if (not registration or registration.get('profile_sha256')!=binding['profile_sha256'] or
                        registration.get('revision')!=binding['revision']):raise ValueError('voice_dimensions_not_registered')
                    if self.cancelled.is_set():raise ValueError('voice_cancelled_before_start')
                    current=saved
                    self.record('voice_library_dimensions_registered',operation_id=kw['request_id'],
                        library_revision=prepared.get('library_revision'),frame_id=prepared.get('frame_id'),
                        profiles=[i.get('measurement_profile') for i in plan['items']],physical_request_sent=False)
                except (OSError,ValueError,KeyError,TypeError) as exc:
                    raise VoiceCommandNotSubmitted('voice_dimensions_rejected_before_start:'+str(exc)) from exc
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
    parser.add_argument('--dimensions-port',type=int,default=18943)
    args=parser.parse_args()
    state_root=(args.state_root or args.root/'state').resolve()
    from . import km1_voice_live
    token=args.control_token_file.read_text().strip()
    endpoint='http://127.0.0.1:'+str(args.controller_port)
    class Client(SingleAttemptVoiceClient):
        @staticmethod
        def dimensions_prepare(data):
            request=urllib.request.Request('http://127.0.0.1:'+str(args.dimensions_port)+'/prepare',
                data=json.dumps(data).encode(),headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request,timeout=8) as response:raw=response.read(2_000_001)
            if len(raw)>2_000_000:raise ValueError('voice_dimensions_reply_too_large')
            return json.loads(raw)
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
            result=super().open();connected(True,self.port)
            self._voice_done=threading.Event()
            def heartbeat():
                while not self._voice_done.wait(3):connected(True,self.port)
            self._voice_thread=threading.Thread(target=heartbeat,daemon=True);self._voice_thread.start()
            return result
        def close(self):
            was_open=getattr(self,'is_open',False)
            if hasattr(self,'_voice_done'):self._voice_done.set()
            if hasattr(self,'_voice_thread'):self._voice_thread.join(timeout=4)
            result=super().close()
            if was_open:connected(False,self.port)
            return result
    serial.Serial=VoiceSerial
    sys.argv=['board_voice_host','--root',str(args.root),'--port',args.voice_port,
              '--log',str(state_root/'voice-events.jsonl')]
    km1_voice_live.main()


if __name__=='__main__':main()
