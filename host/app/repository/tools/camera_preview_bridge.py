"""Camera-only PYNQ JPEG source for local preview; no actuator commands."""
import argparse
import importlib.util
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shlex
import struct
import threading
import time
import uuid

REMOTE = r'''
import cv2,sys,threading,time,struct
cap=cv2.VideoCapture('/dev/video0',cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FPS,10)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
if not cap.isOpened(): raise RuntimeError('camera unavailable')
latest=[None,0.0]
lock=threading.Lock()
done=threading.Event()
def capture():
 while not done.is_set():
  ok,frame=cap.read()
  if not ok: done.set();break
  with lock:latest[:]=[frame,time.monotonic()]
thread=threading.Thread(target=capture,daemon=True);thread.start()
try:
 while sys.stdin.buffer.read(1):
  deadline=time.monotonic()+5
  while latest[0] is None and time.monotonic()<deadline and not done.is_set():time.sleep(.02)
  with lock:frame,stamp=latest
  if frame is None or time.monotonic()-stamp>2:raise RuntimeError('camera stalled')
  ok,jpg=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,85])
  if not ok:raise RuntimeError('jpeg encode failed')
  data=jpg.tobytes();sys.stdout.buffer.write(struct.pack('!I',len(data))+data);sys.stdout.buffer.flush()
finally:
 done.set();thread.join(timeout=2);cap.release()
'''


def exact(stream, size):
    parts = []
    while size:
        chunk = stream.read(size)
        if not chunk:
            raise EOFError('camera stream ended')
        parts.append(chunk)
        size -= len(chunk)
    return b''.join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--connection-module', type=Path, required=True,
                        help='Local private module exposing connect(); never put credentials in Git')
    parser.add_argument('--remote-python', default='/usr/local/share/pynq-venv/bin/python3')
    parser.add_argument('--port', type=int, default=8780)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('camera_private_connection', args.connection_module.resolve())
    connection = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(connection)
    lock = threading.Lock()
    state = dict(jpeg=None, frame_id=0, received=0., connected=False,
                 stream_id=uuid.uuid4().hex, error='等待相机')
    done = threading.Event()

    def capture():
        while not done.is_set():
            client = None
            try:
                client = connection.connect()
                stdin, stdout, stderr = client.exec_command(
                    shlex.quote(args.remote_python) + ' -u -c ' + shlex.quote(REMOTE), timeout=10)
                while not done.is_set():
                    stdin.write('n'); stdin.flush()
                    size = struct.unpack('!I', exact(stdout, 4))[0]
                    if not 0 < size <= 8500000:
                        raise ValueError('invalid camera JPEG size')
                    jpeg = exact(stdout, size)
                    if not jpeg.startswith(b'\xff\xd8'):
                        raise ValueError('invalid camera JPEG')
                    with lock:
                        state.update(jpeg=jpeg, received=time.monotonic(), connected=True, error=None)
                        state['frame_id'] += 1
                    done.wait(.15)
            except Exception as exc:
                with lock:
                    state.update(connected=False, jpeg=None, error=type(exc).__name__)
                done.wait(2)
            finally:
                if client is not None:
                    client.close()

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            self.request.settimeout(10)
            super().setup()

        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.headers.get('Host') not in {'127.0.0.1:' + str(args.port), 'localhost:' + str(args.port)}:
                self.send_error(403); return
            with lock:
                row = dict(state)
            age = (time.monotonic() - row['received']) * 1000
            live = row['connected'] and row['jpeg'] is not None and age < 2000
            if self.path == '/health':
                body = json.dumps(dict(camera_connected=live, frame_id=row['frame_id'],
                    frame_age_ms=age if row['received'] else None, camera_error=row['error'],
                    source='PYNQ /dev/video0', physical_motion=False)).encode()
                status, kind = 200, 'application/json'
            elif self.path == '/frame.jpg' and live:
                body, status, kind = row['jpeg'], 200, 'image/jpeg'
            else:
                self.send_error(503 if self.path == '/frame.jpg' else 404); return
            try:
                self.send_response(status)
                self.send_header('Content-Type', kind)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Frame-Id', str(row['frame_id']))
                self.send_header('X-Stream-Id', row['stream_id'])
                self.send_header('X-Frame-Age-Ms', str(round(age, 2)))
                self.end_headers()
                self.wfile.write(body)
            except (ConnectionError, TimeoutError):
                pass

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    worker = threading.Thread(target=capture, daemon=True); worker.start()
    print('Camera-only source ready on 127.0.0.1:{}'.format(args.port), flush=True)
    try:
        server.serve_forever()
    finally:
        done.set(); server.server_close()


if __name__ == '__main__':
    main()
