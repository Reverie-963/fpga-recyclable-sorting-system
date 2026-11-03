"""Compose the current console with a board-only route-preview adapter.

This wrapper leaves ongoing edits to sorting_prompted_console untouched.
Recognition/configuration/background routes and existing motion gates delegate
to that console. Both preview entry points use the board, with no PC fallback.
"""
import argparse
import json
from pathlib import Path

from .sorting_prompted_console import create_console,console_page
from .board_control_gateway_client import BoardPreviewClient
from .monitor_control import ROOT


class BoardPreviewBackend:
    """Route observations to PS; preserve the existing qualified motion path."""
    def __init__(self,original,board,disable_pc_motion=False):
        self.original,self.board,self.disable_pc_motion=original,board,disable_pc_motion
    def __getattr__(self,name):return getattr(self.original,name)
    @property
    def object_height_mm(self):return getattr(self.original,'object_height_mm',None)
    @object_height_mm.setter
    def object_height_mm(self,value):self.original.object_height_mm=value
    @property
    def height_plan(self):return getattr(self.original,'height_plan',None)
    @height_plan.setter
    def height_plan(self,value):self.original.height_plan=value
    def check(self):
        blockers=self.original.check()
        return blockers+(['board_executor_not_connected'] if self.disable_pc_motion else [])
    def home(self):
        if self.disable_pc_motion:raise ValueError('board_executor_not_connected')
        return self.original.home()
    def cycle(self,material,execute):
        if execute and self.disable_pc_motion:raise ValueError('board_executor_not_connected')
        if execute:return self.original.cycle(material,execute)
        payload=dict(material=material,client='motion-window-preview')
        if self.object_height_mm is not None:payload['object_height_mm']=self.object_height_mm
        if self.height_plan is not None:payload['height_plan']=self.height_plan
        report=self.board.plan(payload)
        result=dict(report,board_preview_status=report.get('status'))
        if report.get('status')=='preview_ready':
            result['status']='preview_only'
            if report.get('candidates'):result['choice']=report['candidates'][0]['choice']
        return result


def create_board_console(port,upstream,board_client,operations=None,disable_pc_motion=False):
    if operations is not None and getattr(operations,'backend',None) is not None:
        operations.backend=BoardPreviewBackend(operations.backend,board_client,disable_pc_motion)
    server=create_console(port,upstream,operations=operations)
    original=server.RequestHandlerClass
    class Handler(original):
        def proxy_prompted(self):
            if self.path!='/prompted/plan':return super().proxy_prompted()
            if not self.allowed():return self.reply(403,dict(error='local_origin_required'))
            if self.command!='POST':return self.reply(405,dict(error='preview_requires_post'))
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=524288 or self.headers.get('Transfer-Encoding'):raise ValueError('invalid_board_plan_body')
                raw=self.rfile.read(size)
                if len(raw)!=size:raise ValueError('truncated_board_plan_body')
                return self.reply(200,board_client.plan(json.loads(raw)))
            except (OSError,ValueError,KeyError,TypeError) as exc:
                return self.reply(503,dict(error='板端预演未完成：'+str(exc)))

        def do_GET(self):
            if not self.allowed():return self.reply(403,dict(error='local_origin_required'))
            if self.path=='/board-control':
                try:
                    health=board_client.health()
                    local=operations.status() if operations is not None else {}
                    blockers=list(local.get('blockers',[]))
                    backend=getattr(operations,'backend',None)
                    protected=[]
                    ledger=getattr(backend,'journal_dir',None)
                    if ledger is not None:protected += [Path(ledger)/n for n in ('controller.lock','pending.json','batch-pending.json')]
                    lease=getattr(backend,'device_lease',None)
                    if lease is not None:protected.append(Path(lease))
                    if any(path.exists() for path in protected):blockers.append('legacy_station_pending_or_in_use')
                    ready=not local.get('busy',False) and not blockers and health.get('ownership',{}).get('status')=='idle'
                    return self.reply(200,dict(connected=True,admission_ready=ready,local_blockers=blockers,
                        pc_motion_disabled=disable_pc_motion,**health))
                except (OSError,ValueError):return self.reply(503,dict(connected=False,execution_enabled=False))
            if self.path=='/':
                page=console_page()
                note="""<script>window.addEventListener('load',()=>{const p=document.createElement('p');p.className='caption';p.id='board-control-note';p.textContent='路线预览使用板端处理；正在核对连接…';document.querySelector('.view-tabs').after(p);fetch('/board-control').then(r=>r.json()).then(d=>{p.textContent=d.connected?(d.admission_ready?'板端已连接 · 路线预览在PYNQ处理；机械动作仍沿用现役入口的核验。':'板端已连接 · 现场任务或配置尚未就绪，路线预览暂缓。'):'板端未连接 · 路线预览暂停，请检查迁移服务。'}).catch(()=>{p.textContent='板端未连接 · 路线预览暂停。'});});</script>"""
                return self.reply(200,page.replace('</body>',note+'</body>').encode(),{'Content-Type':'text/html; charset=utf-8'})
            return super().do_GET()
    server.RequestHandlerClass=Handler
    return server


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',type=int,default=8812)
    p.add_argument('--upstream',default='http://127.0.0.1:8811')
    p.add_argument('--board-controller-config',type=Path,required=True)
    p.add_argument('--disable-pc-motion',action='store_true')
    args=p.parse_args()
    from .monitor_prompted_operation import prompted_operations
    board=BoardPreviewClient(args.board_controller_config,ROOT/'local-only')
    server=create_board_console(args.port,args.upstream,board,operations=prompted_operations(ROOT),disable_pc_motion=args.disable_pc_motion)
    print('Board planning console ready: http://127.0.0.1:'+str(args.port),flush=True)
    try:server.serve_forever()
    finally:server.server_close()


if __name__=='__main__':main()
