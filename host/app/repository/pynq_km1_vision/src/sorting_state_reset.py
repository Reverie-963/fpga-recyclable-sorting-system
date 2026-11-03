"""Archive confirmed idle-state recovery. Never connects to an actuator."""
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
from .sorting_loop import Journal,need


def process_alive(pid):
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=kernel.OpenProcess(0x1000,False,pid)
        if not handle:return ctypes.get_last_error()!=87
        try:
            code=wintypes.DWORD()
            return not kernel.GetExitCodeProcess(handle,ctypes.byref(code)) or code.value==259
        finally:kernel.CloseHandle(handle)
    try:os.kill(pid,0)
    except ProcessLookupError:return False
    except PermissionError:return True
    return True


def owner_pid(raw):
    try:
        value=json.loads(raw.decode('utf-8'))
        pid=value.get('pid') if isinstance(value,dict) else value
        return pid if type(pid) is int and pid>0 else None
    except (ValueError,UnicodeError):return None


def remove_unchanged(path,raw):
    for attempt in range(10):
        need(path.read_bytes()==raw,'reset_state_changed_retry_review')
        try:path.unlink();return
        except PermissionError as exc:
            if getattr(exc,'winerror',None) not in (32,33) or attempt==9:raise
            time.sleep(.05)


def reset_state(folder,lease,stops,confirmed=False,observation='',alive=process_alive):
    folder,lease=Path(folder),Path(lease)
    control=folder/'controller.lock'
    protected=[('motion_pending',folder/'pending.json'),('batch_pending',folder/'batch-pending.json'),
               ('device_lease',lease),('controller_lock',control)]
    protected += [('stop_protection',Path(p)) for p in stops]
    found=[(kind,path,path.read_bytes()) for kind,path in protected if path.exists()]
    need(not found or confirmed,'reset_scene_confirmation_required')
    for kind,path,raw in found:
        if kind in ('device_lease','controller_lock'):
            pid=owner_pid(raw)
            need(pid is not None,'reset_lock_owner_unknown')
            need(not alive(pid),'reset_device_still_owned')
    review_id='reset-'+uuid.uuid4().hex
    archive=folder/'resets'/review_id;archive.mkdir(parents=True,exist_ok=False)
    record=dict(kind='operator-idle-state-reset-v1',reset_id=review_id,
        observation=observation[:1000],scene_confirmed=bool(confirmed),
        outcome='state_reset_not_automatic_success',physical_motion=False,
        original_reports_rewritten=False,automatic_rearm=False,files=[])
    for i,(kind,path,raw) in enumerate(found):
        saved=archive/(str(i)+'.snapshot');saved.write_bytes(raw)
        record['files'].append(dict(kind=kind,path=str(path.resolve()),snapshot=saved.name,
            sha256=hashlib.sha256(raw).hexdigest()))
    (archive/'review.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    stale=next(((path,raw) for kind,path,raw in found if kind=='controller_lock'),None)
    if stale:remove_unchanged(*stale)
    with Journal(folder):
        # The new journal lock excludes every executor sharing this station.
        expected={path for kind,path,_ in found if kind!='controller_lock'}
        current={path for kind,path in protected if kind!='controller_lock' and path.exists()}
        need(current==expected,'reset_state_changed_retry_review')
        for kind,path,raw in found:
            if kind!='controller_lock':need(path.read_bytes()==raw,'reset_state_changed_retry_review')
        for kind,path,raw in found:
            if kind!='controller_lock':remove_unchanged(path,raw)
    record['completed']=True
    (archive/'completed.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return dict(status='state_reset',reset_id=review_id,cleared=[kind for kind,_,_ in found],
                original_reports_preserved=True,physical_motion=False,automatic_rearm=False)
