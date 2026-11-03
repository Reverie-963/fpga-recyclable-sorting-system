"""One board-local admission service for read-only 24-class previews.

No execute/home/reset endpoint exists. Completion releases an observation lease,
never records mechanical success. Stop/heartbeat loss/restart remain latched.
"""
import argparse
import copy
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import signal
import threading
import time
import uuid

from .board_control_ownership import BoardOwnership
from .board_detector_transport import token_value


class BoardTasks:
    def __init__(self, runtime, database, results, ttl_s=12):
        self.runtime = runtime
        self.ownership = BoardOwnership(database, ttl_s)
        self.results = Path(results)
        self.results.mkdir(parents=True, exist_ok=True)
        self.jobs, self.workers = {}, set()
        self.lock = threading.RLock()
        self.closed = False

    def status(self):
        return dict(contract='board-preview-control-v1', compute_location='pynq_ps',
            mode='prompted_preview_only', execution_enabled=False, physical_motion=False,
            ownership=self.ownership.status(), observation_timing=getattr(self.runtime,'timing',None),
            allowed_actions=['preview','heartbeat','stop'])

    def submit(self, data):
        if not isinstance(data, dict) or set(data) != {'owner','request_id','material','expected'}:
            raise ValueError('invalid_board_preview_request')
        if data['material'] not in ('AUTO','PAPER','PLASTIC','METAL') or not isinstance(data['expected'],dict):
            raise ValueError('invalid_board_preview_parameters')
        with self.lock:
            if self.closed:
                raise ValueError('board_control_service_closed')
            lease = self.ownership.claim(data['owner'],data['request_id'])
            task = uuid.uuid4().hex
            self.jobs[task] = dict(task_id=task,status='running',lease=lease,report=None)
            worker = threading.Thread(target=self.run,args=(task,copy.deepcopy(data)),daemon=True)
            self.workers.add(worker)
            worker.start()
            return dict(task_id=task,lease=lease,execution_enabled=False,physical_motion=False)

    def run(self, task, data):
        lease = self.jobs[task]['lease']
        def check():
            self.ownership.check(lease)
        try:
            report = self.runtime.preview(data['expected'],data['material'],check,task)
            if report.get('physical_motion_attempted') is not False or report.get('execution_enabled') is not False:
                raise ValueError('board_preview_returned_invalid_execution_contract')
            # Only observation/planning happened; never label this motion success.
            check()
            self.ownership.finish(lease,'rejected_no_motion')
        except Exception as exc:
            report = dict(status='blocked',task_id=task,reason=str(exc),
                physical_motion_attempted=False,execution_enabled=False)
            try:
                self.ownership.finish(lease,'rejected_no_motion')
            except ValueError:
                pass  # A latched stop/lost lease cannot be cleared by finishing.
        finally:
            with self.lock:
                self.jobs[task].update(status='finished',report=report)
                (self.results/(task+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
                self.workers.discard(threading.current_thread())

    def result(self, task):
        with self.lock:
            if task not in self.jobs:
                raise ValueError('board_task_not_found')
            job = self.jobs[task]
            return dict(task_id=task,status=job['status'],report=copy.deepcopy(job['report']),execution_enabled=False)

    def heartbeat(self, data):
        self.ownership.check(data['lease'],heartbeat=True)
        return dict(accepted=True,execution_enabled=False)

    def stop(self, data):
        self.ownership.stop(data['reason'])
        return self.status()

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            if self.ownership.status()['status'] == 'active':
                self.ownership.stop('Read-only board service closing; inspect unfinished preview.')
            workers = list(self.workers)
        for worker in workers:
            worker.join(2)
        if not any(worker.is_alive() for worker in workers):
            self.ownership.close()


def make_server(tasks, token, port=18812):
    token = token_value(token)
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.request.settimeout(5)

        def log_message(self,*args):
            pass

        def allowed(self):
            return (self.headers.get('Host') in ('127.0.0.1:'+str(server.server_port),'localhost:'+str(server.server_port)) and
                    not self.headers.get('Origin') and hmac.compare_digest(
                        self.headers.get('Authorization','').encode(),('Bearer '+token).encode()))

        def reply(self,status,value):
            body = json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
            try:
                self.send_response(status)
                self.send_header('Content-Type','application/json; charset=utf-8')
                self.send_header('Content-Length',str(len(body)))
                self.send_header('Cache-Control','no-store')
                self.end_headers()
                self.wfile.write(body)
            except OSError:
                pass

        def do_GET(self):
            if not self.allowed():return self.reply(403,dict(error='authenticated_board_loopback_required'))
            server.last_request = time.monotonic()
            try:
                if self.path == '/health':return self.reply(200,tasks.status())
                if self.path.startswith('/tasks/') and len(self.path.split('/')) == 3:
                    return self.reply(200,tasks.result(self.path.split('/')[-1]))
                return self.reply(404,dict(error='board_route_not_allowed'))
            except ValueError as exc:return self.reply(409,dict(error=str(exc)))

        def do_POST(self):
            if not self.allowed():return self.reply(403,dict(error='authenticated_board_loopback_required'))
            server.last_request = time.monotonic()
            if self.path not in ('/tasks','/heartbeat','/stop','/shutdown'):
                return self.reply(404,dict(error='board_has_no_motion_endpoints'))
            try:
                size = int(self.headers.get('Content-Length','0'))
                if not 0 < size <= 32768 or self.headers.get('Transfer-Encoding'):
                    raise ValueError('invalid_board_control_body')
                raw = self.rfile.read(size)
                if len(raw) != size:raise ValueError('truncated_board_control_body')
                data = json.loads(raw)
                if self.path == '/shutdown':
                    server.stop_requested.set()
                    return self.reply(200,dict(shutting_down=True,execution_enabled=False))
                result = tasks.submit(data) if self.path == '/tasks' else tasks.heartbeat(data) if self.path=='/heartbeat' else tasks.stop(data)
                return self.reply(202 if self.path=='/tasks' else 200,result)
            except (ValueError,KeyError,TypeError) as exc:return self.reply(409,dict(error=str(exc)))

    server = ThreadingHTTPServer(('127.0.0.1',port),Handler)
    server.daemon_threads = True
    server.last_request = time.monotonic()
    server.stop_requested = threading.Event()
    return server


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binding',type=Path,required=True)
    p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--relay-url',required=True)
    p.add_argument('--relay-token-file',type=Path,required=True)
    p.add_argument('--station-token-file',type=Path,required=True)
    p.add_argument('--control-token-file',type=Path,required=True)
    p.add_argument('--state',type=Path,required=True)
    p.add_argument('--ownership-db',type=Path,
                   help='Reuse the authoritative ownership database across candidate upgrades.')
    p.add_argument('--observation-timing-file',type=Path)
    p.add_argument('--port',type=int,default=18812)
    args = p.parse_args()
    import cv2
    cv2.setNumThreads(1)
    from .board_prompted_runtime import BoardPromptedRuntime
    runtime = BoardPromptedRuntime(args.binding,args.policy,args.relay_url,
        args.relay_token_file.read_text().strip(),args.station_token_file.read_text().strip(),args.state/'artifacts',
        json.loads(args.observation_timing_file.read_text()) if args.observation_timing_file else None)
    if args.ownership_db is not None and not args.ownership_db.is_file():
        raise ValueError('authoritative_ownership_database_missing')
    tasks = BoardTasks(runtime,args.ownership_db or args.state/'ownership.db',args.state/'results')
    server = make_server(tasks,args.control_token_file.read_text().strip(),args.port)
    stop = threading.Event()
    signal.signal(signal.SIGTERM,lambda *_:stop.set())
    signal.signal(signal.SIGINT,lambda *_:stop.set())
    server.timeout = .5
    print(json.dumps(dict(status='board_preview_ready',port=server.server_port,execution_enabled=False)),flush=True)
    try:
        while not stop.is_set() and not server.stop_requested.is_set() and time.monotonic()-server.last_request < 120:
            server.handle_request()
    finally:
        tasks.close()
        server.server_close()


if __name__ == '__main__':main()
