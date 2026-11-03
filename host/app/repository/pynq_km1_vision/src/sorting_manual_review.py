"""Operator acknowledgement of a finished review latch. No actuator access."""
import copy
import hashlib
import json
import os
from pathlib import Path
import uuid

from .sorting_loop import Journal,need


def uncertain_record(value):
    if isinstance(value,list):return any(uncertain_record(item) for item in value)
    if not isinstance(value,dict):return False
    if any(value.get(key) is True for key in ('outcome_unknown','command_outcome_unknown','transport_uncertain')):return True
    if any(value.get(key) is False for key in ('echo_matched','completed_movement_wait')):return True
    words=('partial usb write','partial uart write','reply lost','connection faulted','motion uncertain',
           'outcome_unknown','echo_mismatch','echo_timeout','echo mismatch','echo timeout','receipt_unknown')
    if any(any(word in str(value.get(key,'')).lower() for word in words) for key in ('reason','error','failure','message')):return True
    return any(uncertain_record(item) for item in value.values() if isinstance(item,(dict,list)))


def pending_requires_recovery(folder):
    try:
        for name in ('pending.json','batch-pending.json'):
            path=Path(folder)/name
            if path.exists() and uncertain_record(json.loads(path.read_text(encoding='utf-8'))):return True
    except (OSError,ValueError,TypeError):return True
    return False


def review_status(folder,lease,owner):
    folder=Path(folder);state=owner.status()
    pending=any((folder/name).exists() for name in ('pending.json','batch-pending.json'))
    required=pending or state['status']=='blocked' and state.get('reason')=='needs_review'
    recovery=pending_requires_recovery(folder)
    available=(not recovery and state['status']=='blocked' and state.get('reason')=='needs_review' and
        not Path(lease).exists() and not (folder/'controller.lock').exists())
    return dict(supported=True,required=required,available=bool(available),owner_epoch=state.get('epoch'),
        request_id=state.get('request_id'),recovery_required=recovery,scope='finished_needs_review_only')


def acknowledge(folder,lease,owner,*,epoch,request_id,confirmed=False):
    need(confirmed is True,'manual_review_scene_confirmation_required')
    folder,lease=Path(folder),Path(lease)
    need(isinstance(epoch,str) and isinstance(request_id,str) and bool(request_id),'manual_review_identity_required')
    need(not lease.exists() and not (folder/'controller.lock').exists(),'manual_review_device_still_owned')
    archive=None;moved=[]
    try:
        with owner.transaction() as state:
            owner.current(state)
            need(state.get('epoch')==epoch and state.get('request_id')==request_id,'manual_review_context_changed')
            need(state.get('status')=='blocked' and state.get('reason')=='needs_review','manual_review_not_eligible')
            prior=copy.deepcopy(state)
            with Journal(folder):
                need(not lease.exists(),'manual_review_device_still_owned')
                need(not pending_requires_recovery(folder),'manual_review_requires_recovery')
                review_id='review-'+uuid.uuid4().hex;archive=folder/'reviews'/review_id;archive.mkdir(parents=True,exist_ok=False)
                snapshots=[]
                for name in ('pending.json','batch-pending.json'):
                    path=folder/name
                    if path.exists():
                        raw=path.read_bytes();need(not uncertain_record(json.loads(raw)),'manual_review_requires_recovery');snapshots.append((path,raw))
                record=dict(kind='operator-manual-review-v1',review_id=review_id,phase='operator_confirmed_archive_prepared',
                    observation='Operator confirms arm at initial position, gripper open and empty, hands clear; ready for a new explicit request.',
                    owner_before=prior,scene_confirmed=True,physical_motion=False,automatic_rearm=False,
                    original_reports_rewritten=False,automatic_success=False,files=[dict(name=p.name,sha256=hashlib.sha256(raw).hexdigest()) for p,raw in snapshots])
                body=json.dumps(record,ensure_ascii=False,indent=2).encode()
                with (archive/'review.json').open('wb') as stream:stream.write(body);stream.flush();os.fsync(stream.fileno())
                for path,raw in snapshots:
                    need(path.read_bytes()==raw,'manual_review_pending_changed')
                    destination=archive/path.name;path.replace(destination);moved.append((path,destination,raw))
                need(not lease.exists(),'manual_review_device_still_owned')
                owner.event('operator_review_not_automatic_success',state,review_id=review_id,
                    archive_manifest_sha256=hashlib.sha256(body).hexdigest(),observation=record['observation'])
                state.clear();state.update(epoch=owner.epoch,status='idle')
        return dict(status='operator_reviewed',review_id=review_id,reviewed_request_id=request_id,
            archived=[p.name for p,_,_ in moved],original_reports_preserved=True,physical_motion=False,automatic_rearm=False)
    except BaseException:
        # A failed SQLite/audit commit leaves the owner blocked. Restore only
        # our exact snapshots, never overwrite any newly appearing protection.
        for path,destination,raw in reversed(moved):
            if not path.exists() and destination.exists() and destination.read_bytes()==raw:destination.replace(path)
        raise
