"""Recover only workers whose PID, creation time and exact arguments match."""
import json
from pathlib import Path
import socket
import subprocess


class ExistingWorker:
    def __init__(self,process,created):self.process=process;self.created=created;self.pid=process.pid
    def poll(self):
        try:return None if self.process.is_running() and self.process.create_time()==self.created else 0
        except Exception:return 0
    def wait(self,timeout=None):
        import psutil
        try:return self.process.wait(timeout)
        except psutil.TimeoutExpired:raise subprocess.TimeoutExpired(str(self.pid),timeout)
    def kill(self):
        if self.poll() is None:self.process.kill()


class WorkerPool:
    def __init__(self,catalog_path):
        self.catalog=Path(catalog_path).resolve();self.path=self.catalog.parent/'execution-worker-leases.json'
        self.rows=json.loads(self.path.read_text()) if self.path.is_file() else {}
    def remember(self,process,row,url):
        import psutil
        self.rows[row['id']]=dict(pid=process.pid,created=psutil.Process(process.pid).create_time(),url=url)
        temporary=self.path.with_suffix('.tmp');temporary.write_text(json.dumps(self.rows));temporary.replace(self.path)
    def find(self,row):
        import psutil
        saved=self.rows.get(row['id'])
        if not saved:return None
        try:
            process=psutil.Process(saved['pid']);args=process.cmdline()
            if process.create_time()!=saved['created'] or 'src.console_candidate_monitor' not in args:return None
            if Path(args[args.index('--catalog')+1]).resolve()!=self.catalog or args[args.index('--model')+1]!=row['id']:return None
            if str(args[args.index('--port')+1])!=saved['url'].rsplit(':',1)[1]:return None
            return ExistingWorker(process,saved['created']),saved['url']
        except (psutil.Error,ValueError,IndexError):return None


def free_worker_port(ports,exclude=None):
    for port in ports:
        if port==exclude:continue
        with socket.socket() as sock:
            if hasattr(socket,'SO_EXCLUSIVEADDRUSE'):sock.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
            try:sock.bind(('127.0.0.1',port));return port
            except OSError:pass
    raise ValueError('没有空闲的模型工作端口，原模型保留。')
