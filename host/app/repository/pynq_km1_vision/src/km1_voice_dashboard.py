"""ASRPRO requests enter the same local material interfaces as the dashboard."""
import hashlib
import json
from pathlib import Path
import re
import threading
import time
from urllib.parse import urlsplit
import urllib.request
import urllib.error
import uuid


ROUTES = {'AUTO': '/operation/all', 'PAPER': '/operation/paper',
          'METAL': '/operation/metal', 'PLASTIC': '/operation/plastic'}
IDENTIFIER = re.compile(r'[a-zA-Z0-9_-]{1,160}')


class VoiceCommandNotSubmitted(ValueError):
    """A local policy rejected a command before any start request was sent."""


class VoiceRecords:
    def __init__(self, folder):
        self.folder = Path(folder)

    def read(self, identifier, suffix='.json'):
        if not isinstance(identifier, str) or not IDENTIFIER.fullmatch(identifier):
            raise ValueError('invalid_voice_record_identifier')
        raw = (self.folder/(identifier+suffix)).read_bytes()
        if len(raw) > 2_000_000:
            raise ValueError('oversized_voice_record')
        return raw, json.loads(raw)

    def selected(self, result):
        """Read only the exact task or the hash-bound final physical batch item."""
        if result.get('batch_id'):
            _, batch = self.read(result['batch_id'], '.batch.json')
            if result.get('status')=='completed_visual_evidence' and batch.get('status') in ('completed','selection_complete'):
                return self.final_batch(result,batch)
            if batch.get('batch_id') != result['batch_id'] or batch.get('status') != result['status']:
                raise ValueError('voice_batch_record_mismatch')
            if batch['status'] not in ('completed', 'selection_complete'):
                return dict(mode='known_physical', status='needs_review' if
                    batch['status']=='needs_review' else 'rejected')
            if batch.get('final_verification') and batch['final_verification'].get('status')!='batch_visual_evidence_pass':
                return dict(mode='known_physical',status='needs_review')
            items = batch.get('items', [])
            physical = [i for i in items if i.get('physical_motion_attempted') is True]
            if not physical:
                return dict(voice_transport_reply='EMPTY')
            item = physical[-1]
            raw, record = self.read(item['task_id'])
            if (hashlib.sha256(raw).hexdigest() != item['report_sha256'] or
                    record.get('batch_id') != batch['batch_id'] or
                    record.get('task_id') != item['task_id']):
                raise ValueError('voice_batch_item_hash_mismatch')
            if record.get('status')=='motion_completed_awaiting_batch_verification':
                _,final=self.read(batch['batch_id'],'.batch-final-verification.json')
                choice=record.get('choice',{});frames=final.get('frames',[])
                if (record.get('mode')!='known_physical' or final!=batch.get('final_verification') or
                    final.get('status')!='batch_visual_evidence_pass' or len(frames)<3 or
                    len({f.get('frame_id') for f in frames})!=len(frames) or
                    not choice.get('track_id') or not all(any(
                        p.get('track_id')==choice['track_id'] and p.get('label')==choice.get('label') and
                        p.get('slot_id')==choice.get('slot_id') for p in f.get('placements',[])) for f in frames)):
                    raise ValueError('voice_batch_final_evidence_mismatch')
                # Read-only voice projection. The saved per-item status remains
                # deferred; success is sourced explicitly from the final batch.
                return dict(record,status='completed_visual_evidence',voice_source_status=record['status'],
                            voice_evidence_basis='matching_batch_final_verification')
            return record
        _, record = self.read(result.get('task_id'))
        if record.get('task_id') != result['task_id'] or record.get('status') != result['status']:
            raise ValueError('voice_task_record_mismatch')
        return record

    def final_batch(self,result,batch):
        """Current aggregate status is normalized; raw batch status is retained.

        Check every physical child, not just the last material. Occupancy-only
        evidence must never be promoted to item/class/sorting success by voice.
        """
        identifier=result['batch_id'];_,aggregate=self.read(identifier)
        if (batch.get('batch_id')!=identifier or aggregate.get('task_id')!=identifier or
            aggregate.get('batch_id')!=identifier or aggregate.get('status')!=result['status'] or
            aggregate.get('batch_status')!=batch.get('status') or aggregate.get('items')!=batch.get('items') or
            aggregate.get('final_verification')!=batch.get('final_verification') or
            aggregate.get('final_verification')!=result.get('final_verification')):
            raise ValueError('voice_batch_aggregate_mismatch')
        final=batch.get('final_verification') or {};_,saved=self.read(identifier,'.batch-final-verification.json')
        physical=[i for i in batch.get('items',[]) if i.get('physical_motion_attempted') is True]
        if not physical:return dict(voice_transport_reply='EMPTY')
        if (saved!=final or final.get('status')!='batch_visual_evidence_pass' or
            final.get('attempted_count')!=len(physical) or batch.get('motion_completed_count')!=len(physical)):
            raise ValueError('voice_batch_final_evidence_mismatch')
        records=[];ids=set()
        for item in physical:
            raw,record=self.read(item['task_id'])
            if (item['task_id'] in ids or hashlib.sha256(raw).hexdigest()!=item.get('report_sha256') or
                record.get('task_id')!=item['task_id'] or record.get('batch_id')!=identifier or
                record.get('mode')!='known_physical' or record.get('status') not in
                    ('motion_completed_awaiting_batch_verification','completed_visual_evidence')):
                raise ValueError('voice_batch_item_hash_mismatch')
            ids.add(item['task_id']);records.append(record)
        frames=final.get('frames',[])
        if final.get('mode')=='bin_occupied':
            bins=final.get('bins',[]);visited={r.get('choice',{}).get('bin_id') for r in records}
            if (final.get('scope')!='visited_bin_occupancy_only' or len(frames)!=1 or
                len(bins)!=len(visited) or None in visited or {b.get('bin_id') for b in bins}!=visited or
                not all(b.get('status')=='bin_occupied_pass' and len(b.get('frames',[]))==1 and
                    b['frames'][0].get('occupied') is True and b['frames'][0].get('frame_id')==frames[0].get('frame_id') for b in bins) or any(final.get(k) is not False for k in
                    ('physical_success_verified','class_verified','new_item_verified','source_removal_verified'))):
                raise ValueError('voice_batch_occupancy_evidence_mismatch')
            return dict(aggregate,voice_source_status=aggregate['status'],
                voice_evidence_basis='visited_bin_occupancy_only',voice_verification_completed=True,
                voice_completion_scope='verification_completed_not_item_or_class_success',
                physical_success_verified=False,class_verified=False,new_item_verified=False,source_removal_verified=False)
        if (len(frames)<3 or len({f.get('frame_id') for f in frames})!=len(frames) or not all(
            r.get('choice',{}).get('track_id') and all(any(p.get('track_id')==r['choice']['track_id'] and
                p.get('label')==r['choice'].get('label') and p.get('slot_id')==r['choice'].get('slot_id')
                for p in f.get('placements',[])) for f in frames) for r in records)):
            raise ValueError('voice_batch_final_evidence_mismatch')
        labels={r['choice']['label'] for r in records}
        return dict(aggregate,choice=dict(label=next(iter(labels)) if len(labels)==1 else None),
            voice_materials=sorted(labels),voice_evidence_basis='matching_batch_final_verification',
            voice_verification_completed=True)


class DashboardVoiceClient:
    def __init__(self, endpoint, records, fallback_stop=lambda: None, record=lambda *a, **k: None):
        parsed = urlsplit(endpoint)
        if (parsed.scheme != 'http' or parsed.hostname not in ('localhost', '127.0.0.1') or
                not parsed.port or parsed.path not in ('', '/') or parsed.query or parsed.fragment or
                parsed.username or parsed.password):
            raise ValueError('voice_requires_local_dashboard')
        self.endpoint = endpoint.rstrip('/')
        self.records, self.fallback_stop, self.record = records, fallback_stop, record
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.owner = 'voice-'+uuid.uuid4().hex
        self.cancelled = threading.Event()
        self.token = None
        self.operation_id = None
        self._status()  # Validate interface, without arming or starting an operation.

    def _request(self, route, body=None):
        request = urllib.request.Request(self.endpoint+route,
            data=json.dumps(body).encode() if body is not None else None,
            headers={'Content-Type': 'application/json'} if body is not None else {})
        with self.opener.open(request, timeout=5) as reply:
            raw = reply.read(2_000_001)
        if len(raw) > 2_000_000:
            raise ValueError('oversized_voice_interface_reply')
        return json.loads(raw)

    def _status(self):
        result = self._request('/operation')
        if result.get('control_version') != 3 or any(
                result.get('operation_routes', {}).get(path) != material for material, path in ROUTES.items()):
            raise ValueError('voice_material_interface_version_mismatch')
        if self.token is not None and result.get('token') != self.token:
            raise ValueError('dashboard_restarted_voice_reconnect_required')
        self.token = result['token']
        return result

    def _command(self, action, route='/operation', **values):
        return self._request(route, dict(token=self.token, owner=self.owner, action=action, **values))

    def stop(self):
        self.cancelled.set()
        # Persist the existing stop even if HTTP is unavailable; never open COM9.
        self.fallback_stop()
        self._command('stop')

    def latest(self):
        status = self._status()
        if status.get('busy'):
            return dict(mode='known_physical', status='motion_in_progress')
        result=status.get('result') or {}
        if self.records is not None and result.get('batch_id') and not result.get('preview') and result.get('mode')=='known_physical':
            return self.records.selected(result)
        data = self._request('/control-status')
        if self.records is not None and data.get('available') and data.get('record_kind')=='physical' and data.get('task_id'):
            _,record=self.records.read(data['task_id'])
            return self.records.selected(record)
        return dict(mode='known_physical' if data.get('available') and
            data.get('record_kind')=='physical' else None,
            status=data.get('raw_status'), choice={'label': data.get('label')})

    def run(self, material, execute, ready):
        if material not in ROUTES or (execute and ready is not True):
            raise ValueError('invalid_voice_execution_request')
        if self.cancelled.is_set():
            return dict(voice_transport_reply='STOPPED')
        status = self._status()
        if status.get('busy'):
            return dict(voice_transport_reply='BUSY')
        request_id = uuid.uuid4().hex
        self.operation_id = request_id
        self.record('interface_request', endpoint=ROUTES[material], operation_id=request_id,
                    material=material, execute=execute)
        try:
            # Send exactly once. A lost POST response must never cause a retry.
            try:
                status = self._command('start' if execute else 'preview', ROUTES[material],
                    request_id=request_id, ready=ready, material=material)
            except VoiceCommandNotSubmitted as exc:
                # This typed outcome is valid only at initial dispatch. An
                # error after acceptance still follows the stop path below.
                self.record('interface_rejected_before_submission',operation_id=request_id,
                    material=material,reason=str(exc),physical_request_sent=False)
                return dict(voice_transport_reply='REJECTED')
            deadline = time.monotonic()+1300
            while True:
                if status.get('owner') != self.owner or status.get('operation_id') != request_id:
                    raise ValueError('voice_operation_identity_mismatch')
                if self.cancelled.is_set():
                    return dict(voice_transport_reply='STOPPED')
                if not status.get('busy'):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError('voice_operation_timeout')
                self._command('heartbeat', operation_id=request_id)
                self.cancelled.wait(.5)
                status = self._status()
            result = status.get('result') or {}
            self.record('interface_result', operation_id=request_id, state=status.get('state'), result=result)
            if status.get('state') in ('rejected', 'stopped'):
                return dict(voice_transport_reply='REJECTED' if status['state']=='rejected' else 'STOPPED')
            if result.get('status') in ('empty_scene', 'no_matching_material'):
                return dict(voice_transport_reply='EMPTY')
            if not execute:
                return dict(voice_transport_reply='PREVIEW' if result.get('status')=='preview_only' else 'REJECTED')
            if result.get('preview'):
                raise ValueError('physical_voice_reply_requires_physical_operation')
            if result.get('status') not in ('completed', 'selection_complete', 'completed_visual_evidence'):
                return dict(mode='known_physical', status='needs_review' if
                    result.get('status')=='needs_review' else 'rejected')
            return self.records.selected(result)
        except urllib.error.HTTPError as exc:
            # A definite rejection did not grant motion ownership. In
            # particular, a simultaneous web request must not be stopped.
            try:
                data=json.loads(exc.read(65536))
            except (ValueError,OSError):
                data={}
            finally:
                exc.close()
            operation=data.get('operation',{})
            if operation.get('owner')==self.owner and operation.get('operation_id')==request_id and execute:
                try:self.stop()
                except Exception:pass
            return dict(voice_transport_reply='BUSY' if operation.get('busy') and
                        operation.get('owner')!=self.owner else 'REJECTED')
        except Exception:
            if execute:
                try:
                    self.stop()
                except Exception:
                    pass  # Persistent stop was attempted; server lease also expires.
            raise
