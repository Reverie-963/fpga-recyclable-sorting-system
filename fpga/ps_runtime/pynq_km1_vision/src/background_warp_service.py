"""Authenticated loopback background math service, without actuator APIs."""
import argparse
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
import time

import cv2
import numpy as np
from .background_warp import RayTable, warp_reference


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def signature(token, nonce, input_sha, body_sha, generation, manifest_sha, outcome):
    value = ':'.join((nonce, input_sha, body_sha, generation, manifest_sha, outcome)).encode()
    return hmac.new(token.encode(), value, hashlib.sha256).hexdigest()


def parameters(value):
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError('invalid_background_parameters')
    k,d,r,t = [np.asarray(v, dtype=np.float64) for v in value]
    if (k.shape != (3,3) or r.shape != (3,3) or t.shape != (3,) or d.ndim != 1
            or d.size not in (4,5,8,12,14) or not all(np.isfinite(v).all() for v in (k,d,r,t))
            or k[0,0] <= 0 or k[1,1] <= 0 or not np.allclose(r@r.T,np.eye(3),atol=1e-6) or np.linalg.det(r) <= .999):
        raise ValueError('invalid_background_parameters')
    return k,d,r,t


class BackgroundMath:
    def __init__(self, manifest_path, token):
        raw = Path(manifest_path).read_bytes()
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        self.manifest = json.loads(raw)
        if self.manifest.get('schema') != 'background-math-manifest-v1' or len(token) < 32:
            raise ValueError('invalid_background_manifest_or_key')
        self.token, self.generation = token, secrets.token_hex(16)
        self.size = self.manifest['camera']['image_size']
        self.k = np.asarray(self.manifest['camera']['camera_matrix'], np.float64)
        self.d = np.asarray(self.manifest['camera']['dist_coeffs'], np.float64)
        self.rays = RayTable(); self.rays.get(self.size, self.k, self.d)
        self.references = {}
        for ref in self.manifest['references']:
            raw = Path(ref['path']).read_bytes()
            if hashlib.sha256(raw).hexdigest() != ref['file_sha256']:
                raise ValueError('background_reference_file_changed')
            image = cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
            if image is None or list(image.shape[1::-1]) != self.size:
                raise ValueError('background_reference_size_changed')
            sha = hashlib.sha256(image.tobytes()).hexdigest()
            if sha != ref['bgr_sha256'] or sha in self.references:
                raise ValueError('background_reference_pixels_changed')
            image.flags.writeable = False
            self.references[sha] = (image, ref['reference_parameters'])
        if not 1 <= len(self.references) <= 4:
            raise ValueError('bounded_background_references_required')
        self.code_sha256 = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                           (Path(__file__), Path(__file__).with_name('background_warp.py'))}

    def health(self):
        return dict(backend='pc_background_math_v1', ready=True, generation=self.generation,
                    manifest_sha256=self.manifest_sha256, code_sha256=self.code_sha256,
                    opencv_version=cv2.__version__, physical_motion=False)

    def evaluate(self, payload):
        required = {'nonce','reference_bgr_sha256','size','regions','profiles','current','reference','context'}
        if set(payload) != required or payload['size'] != self.size:
            raise ValueError('invalid_background_request_fields')
        nonce = payload['nonce']
        if not isinstance(nonce,str) or len(nonce) != 32 or any(c not in '0123456789abcdef' for c in nonce):
            raise ValueError('invalid_background_request_nonce')
        image, expected = self.references[payload['reference_bgr_sha256']]
        if payload['reference'] != expected:
            raise ValueError('background_reference_pose_changed')
        current, reference = parameters(payload['current']), parameters(payload['reference'])
        if not np.array_equal(current[0],self.k) or not np.array_equal(current[1],self.d):
            raise ValueError('background_camera_changed')
        regions, profiles = payload['regions'], payload['profiles']
        if not isinstance(regions,dict) or not 1 <= len(regions) <= 8 or not isinstance(profiles,dict):
            raise ValueError('invalid_background_regions')
        width,height = self.size
        for name, row in profiles.items():
            roi=row['bin_roi']
            if (not isinstance(roi,list) or len(roi) != 4 or not all(type(v) is int for v in roi)
                    or not 0 <= roi[0] < roi[2] <= width or not 0 <= roi[1] < roi[3] <= height):
                raise ValueError('invalid_background_roi')
        return warp_reference(image,self.size,regions,profiles,current,reference,self.rays)


def make_server(port, math):
    gate = threading.BoundedSemaphore(1)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*a):pass
        def authorized(self):return hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+math.token)
        def plain(self, code, value):
            raw=encoded(value)
            self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)))
            self.end_headers();self.wfile.write(raw)
        def do_GET(self):
            if not self.authorized():return self.plain(403,dict(error='authentication_required'))
            if self.path != '/health':return self.plain(404,dict(error='background_math_only'))
            return self.plain(200,math.health())
        def do_POST(self):
            if not self.authorized():return self.plain(403,dict(error='authentication_required'))
            if self.path != '/warp':return self.plain(404,dict(error='background_math_only'))
            if not gate.acquire(blocking=False):return self.plain(409,dict(error='background_math_busy'))
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0 < size <= 65536 or self.headers.get('Transfer-Encoding'):
                    return self.plain(400,dict(error='bounded_background_body_required'))
                raw=self.rfile.read(size)
                if len(raw) != size:return self.plain(400,dict(error='truncated_background_body'))
                payload=json.loads(raw);nonce=payload['nonce'];input_sha=hashlib.sha256(raw).hexdigest()
                if not isinstance(nonce,str) or len(nonce)!=32 or any(c not in '0123456789abcdef' for c in nonce):
                    return self.plain(400,dict(error='invalid_background_request_nonce'))
                started=time.perf_counter_ns()
                try:
                    image=math.evaluate(payload);body=image.tobytes();outcome='accepted'
                except ValueError as exc:
                    body=encoded(dict(reason=str(exc)));outcome='rejected'
                body_sha=hashlib.sha256(body).hexdigest()
                self.send_response(200)
                self.send_header('Content-Type','application/octet-stream')
                self.send_header('Content-Length',str(len(body)))
                self.send_header('Cache-Control','no-store')
                for name,value in dict(Nonce=nonce,Input=input_sha,Output=body_sha,Generation=math.generation,
                        Manifest=math.manifest_sha256,Outcome=outcome,
                        Signature=signature(math.token,nonce,input_sha,body_sha,math.generation,math.manifest_sha256,outcome),
                        Compute=str((time.perf_counter_ns()-started)/1e6)).items():
                    self.send_header('X-Background-'+name,value)
                self.end_headers();self.wfile.write(body)
            except (OSError,ValueError,KeyError,TypeError):
                try:self.plain(400,dict(error='invalid_background_request'))
                except OSError:pass
            finally:gate.release()
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--token-file',type=Path,required=True)
    parser.add_argument('--port',type=int,default=8846)
    args=parser.parse_args()
    if not 1024 <= args.port <= 65535:raise ValueError('invalid_loopback_port')
    cv2.setNumThreads(1)
    math=BackgroundMath(args.manifest,args.token_file.read_text().strip())
    server=make_server(args.port,math)
    print(json.dumps(dict(port=args.port,**math.health())),flush=True)
    server.serve_forever()


if __name__=='__main__':main()
