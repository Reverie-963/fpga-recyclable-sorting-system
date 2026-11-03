"""Loopback HTTP bridge to the board-local service through an existing SSH account.

Only image/evidence and explicit baseline routes are forwarded. Never starts a
remote process, loads an overlay, changes wiring or opens an arm interface.
"""
import argparse
import http.client
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import threading
from urllib.parse import urlsplit
from paramiko import SSHException


class _BorrowedChannelSocket:
    """HTTP/1.0 closes its socket before body read; retain the SSH channel.

    Paramiko ChannelFile does not retain the channel like socket.makefile does.
    BoardTransport closes the channel explicitly after consuming the response.
    """
    def __init__(self,channel):self.channel=channel
    def __getattr__(self,name):return getattr(self.channel,name)
    def close(self):pass


class BoardTransport:
    def __init__(self,connect,token,port=8783):
        if len(token)<32:raise ValueError('token_too_short')
        self.connect=connect;self.token=token;self.port=port;self.client=None;self.lock=threading.Lock()

    def request(self,path,body=None):
        with self.lock:
            transport=self.client.get_transport() if self.client is not None else None
            if transport is None or not transport.is_active():
                if self.client is not None:self.client.close()
                self.client=self.connect();transport=self.client.get_transport()
            try:channel=transport.open_channel('direct-tcpip',('127.0.0.1',self.port),('127.0.0.1',0),timeout=3)
            except SSHException as exc:raise ConnectionError('board_service_unavailable') from exc
        channel.settimeout(3)
        connection=http.client.HTTPConnection('127.0.0.1',self.port,timeout=3);connection.sock=_BorrowedChannelSocket(channel)
        try:
            connection.request('POST' if body is not None else 'GET',path,body=body,
                headers={'Authorization':'Bearer '+self.token,'Content-Type':'application/json'})
            response=connection.getresponse();raw=response.read(8_500_001)
            if len(raw)>8_500_000:raise ValueError('oversized_board_response')
            headers={key:value for key,value in response.getheaders() if key.lower() in
                ('content-type','x-frame-id','x-stream-id','x-frame-age-ms','x-decoded-bgr-sha256','x-station-metadata','x-baseline-record')}
            return response.status,raw,headers
        finally:connection.close();channel.close()

    def close(self):
        if self.client is not None:self.client.close()


def make_server(port,transport):
    server=None
    class Handler(BaseHTTPRequestHandler):
        def setup(self):super().setup();self.request.settimeout(5)
        def log_message(self,*args):pass
        def allowed(self):
            hosts={'127.0.0.1:'+str(server.server_port),'localhost:'+str(server.server_port)}
            return self.headers.get('Host') in hosts and (not self.headers.get('Origin') or
                self.headers['Origin'] in {'http://'+host for host in hosts})
        def reply(self,status,raw,headers=None):
            if not isinstance(raw,bytes):raw=json.dumps(raw).encode()
            try:
                self.send_response(status)
                for key,value in (headers or {'Content-Type':'application/json'}).items():self.send_header(key,value)
                self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store')
                self.end_headers();self.wfile.write(raw)
            except OSError:pass
        def do_GET(self):
            if not self.allowed():return self.reply(403,dict(error='local_origin_required'))
            if urlsplit(self.path).path not in ('/health','/frame.jpg','/vision','/snapshot.png'):
                return self.reply(404,dict(error='not_found'))
            try:self.reply(*transport.request(self.path))
            except (OSError,ValueError,http.client.HTTPException):self.reply(503,dict(error='board_service_unavailable'))
        def do_POST(self):
            if not self.allowed():return self.reply(403,dict(error='local_origin_required'))
            if self.path not in ('/baseline','/regions','/baseline-snapshot.png'):return self.reply(404,dict(error='no_motion_endpoints'))
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=2048 or self.headers.get('Transfer-Encoding'):raise ValueError('invalid_body_size')
                raw=self.rfile.read(size)
                if len(raw)!=size:raise ValueError('incomplete_body')
                self.reply(*transport.request(self.path,raw))
            except ValueError:self.reply(400,dict(error='invalid_baseline_request'))
            except (OSError,http.client.HTTPException):self.reply(503,dict(error='board_service_unavailable'))
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    return server


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--connection-module',type=Path,required=True)
    parser.add_argument('--token-file',type=Path,required=True)
    parser.add_argument('--port',type=int,default=8780);parser.add_argument('--remote-port',type=int,default=8783)
    args=parser.parse_args()
    spec=importlib.util.spec_from_file_location('station_private_connection',args.connection_module.resolve())
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    transport=BoardTransport(module.connect,args.token_file.read_text().strip(),args.remote_port)
    server=make_server(args.port,transport)
    print('Board-local vision bridge ready on 127.0.0.1:'+str(args.port),flush=True)
    try:server.serve_forever()
    finally:server.server_close();transport.close()


if __name__=='__main__':main()
