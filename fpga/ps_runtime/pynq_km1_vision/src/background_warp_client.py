"""Pinned binary background math transport; failures latch, no implicit retry."""
import hashlib
import hmac
import http.client
import json
from pathlib import Path
import secrets
import time
import numpy as np
from .background_warp_service import encoded, signature


class BackgroundFault(RuntimeError):
    pass


class BackgroundClient:
    def __init__(self, token_file, manifest_sha256, code_sha256, port=8846):
        if type(port) is not int or not 1024 <= port <= 65535:
            raise ValueError('invalid_background_port')
        self.port,self.token=port,Path(token_file).read_text().strip()
        if len(self.token)<32:raise ValueError('background_key_too_short')
        self.manifest_sha256,self.code_sha256=manifest_sha256,dict(code_sha256)
        self.generation=None;self.faulted=False;self.last_timing=None

    def readiness(self):
        connection=http.client.HTTPConnection('127.0.0.1',self.port,timeout=5)
        try:
            connection.request('GET','/health',headers={'Authorization':'Bearer '+self.token})
            response=connection.getresponse()
            if response.status != 200:raise BackgroundFault('background_service_not_ready')
            value=json.loads(response.read(10001))
            if (value.get('backend')!='pc_background_math_v1' or value.get('ready') is not True
                    or value.get('manifest_sha256')!=self.manifest_sha256 or value.get('code_sha256')!=self.code_sha256
                    or value.get('physical_motion') is not False):
                raise BackgroundFault('background_service_identity_changed')
            generation=value['generation']
            if self.generation is not None and generation!=self.generation:
                raise BackgroundFault('background_service_restarted')
            self.generation=generation
            return value
        except Exception as exc:
            self.faulted=True
            raise BackgroundFault('background_readiness_failed') from exc
        finally:connection.close()

    def evaluate(self, request):
        if self.faulted or self.generation is None:
            raise BackgroundFault('background_client_not_ready_or_fault_latched')
        started=time.perf_counter_ns()
        payload=dict(request,nonce=secrets.token_hex(16));raw=encoded(payload)
        input_sha=hashlib.sha256(raw).hexdigest()
        if len(raw)>65536:raise ValueError('background_request_too_large')
        connection=http.client.HTTPConnection('127.0.0.1',self.port,timeout=6)
        try:
            connection.request('POST','/warp',raw,{'Authorization':'Bearer '+self.token,'Content-Type':'application/json'})
            response=connection.getresponse()
            if response.status!=200:raise BackgroundFault('background_math_unavailable')
            size=int(response.getheader('Content-Length','0'))
            if not 1 <= size <= 640*480*3:raise BackgroundFault('background_reply_size_invalid')
            body=response.read(size+1)
            body_sha=hashlib.sha256(body).hexdigest();outcome=response.getheader('X-Background-Outcome')
            expected=signature(self.token,payload['nonce'],input_sha,body_sha,self.generation,self.manifest_sha256,outcome or '')
            if (len(body)!=size or response.getheader('X-Background-Nonce')!=payload['nonce']
                    or response.getheader('X-Background-Input')!=input_sha or response.getheader('X-Background-Output')!=body_sha
                    or response.getheader('X-Background-Generation')!=self.generation
                    or response.getheader('X-Background-Manifest')!=self.manifest_sha256
                    or not hmac.compare_digest(response.getheader('X-Background-Signature',''),expected)
                    or outcome not in ('accepted','rejected')):
                raise BackgroundFault('background_reply_binding_changed')
            self.last_timing=dict(client_total_ms=(time.perf_counter_ns()-started)/1e6,
                                  server_compute_ms=float(response.getheader('X-Background-Compute','nan')))
            if outcome=='rejected':
                rejection=json.loads(body)
                if set(rejection)!={'reason'} or not isinstance(rejection['reason'],str):
                    raise BackgroundFault('invalid_background_math_rejection')
                return None,rejection['reason']
            width,height=payload['size']
            if size!=width*height*3:raise BackgroundFault('background_image_dimensions_changed')
            return np.frombuffer(body,dtype=np.uint8).reshape(height,width,3).copy(),None
        except Exception as exc:
            self.faulted=True
            raise BackgroundFault('background_call_failed_no_fallback') from exc
        finally:connection.close()

    def warp(self, request):
        result,rejection=self.evaluate(request)
        if rejection is not None:raise ValueError(rejection)
        return result
