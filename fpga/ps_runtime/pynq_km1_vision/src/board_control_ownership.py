"""Durable ownership core for the future single board control service.

One board-local SQLite database is authoritative. Host/voice/web clients will
submit to that service, never share local lock files or compare wall clocks.
This module grants no actuator access and is not wired into current executors.
"""
from contextlib import contextmanager
import hashlib
import hmac
import json
import math
from pathlib import Path
import secrets
import sqlite3
import threading
import time
import uuid


def identity(value):
    return isinstance(value, str) and 1 <= len(value) <= 128 and value.isascii() and all(
        c.isalnum() or c in '-_.' for c in value)


class BoardOwnership:
    def __init__(self, database, ttl_s=12, clock=time.monotonic):
        if type(ttl_s) not in (int, float) or not math.isfinite(ttl_s) or not 1 <= ttl_s <= 30:
            raise ValueError('invalid_board_owner_ttl')
        self.path = Path(database)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl, self.clock = ttl_s, clock
        self.epoch = uuid.uuid4().hex
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.path, timeout=3, isolation_level=None, check_same_thread=False)
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS requests (request_id TEXT PRIMARY KEY, owner TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, value TEXT NOT NULL)')
        with self.transaction() as state:
            if state.get('status') == 'active':
                state.update(status='blocked', reason='board_service_restarted_with_owner')
                self.event('restart_blocked', state)
            state.update(epoch=self.epoch)

    @contextmanager
    def transaction(self):
        with self.lock:
            self.db.execute('BEGIN IMMEDIATE')
            try:
                row = self.db.execute('SELECT value FROM state WHERE id=1').fetchone()
                state = json.loads(row[0]) if row else dict(status='idle')
                yield state
                encoded=json.dumps(state,allow_nan=False)
                # A per-poll ownership check need not rewrite/fsync an unchanged
                # state. Real claims, heartbeat renewals and latches stay FULL.
                if row is None or encoded!=row[0]:
                    self.db.execute('INSERT OR REPLACE INTO state VALUES (1,?)', (encoded,))
                self.db.execute('COMMIT')
            except BaseException:
                self.db.execute('ROLLBACK')
                raise

    def event(self, action, state, **extra):
        # Tokens are never stored in clear text in the audit stream.
        value = dict(action=action, epoch=self.epoch, status=state['status'],
                     owner=state.get('owner'), request_id=state.get('request_id'), **extra)
        self.db.execute('INSERT INTO events(value) VALUES (?)', (json.dumps(value, allow_nan=False),))

    def current(self, state):
        if state['epoch'] != self.epoch:
            raise ValueError('board_owner_service_epoch_changed')

    def expire(self, state):
        if state['status'] == 'active' and self.clock() >= state['expires_at']:
            state.update(status='blocked', reason='board_owner_heartbeat_lost')
            self.event('heartbeat_lost', state)

    def status(self):
        with self.transaction() as state:
            self.current(state)
            self.expire(state)
            return dict(epoch=self.epoch, status=state['status'], reason=state.get('reason'),
                        owner=state.get('owner'), request_id=state.get('request_id'),
                        remaining_s=max(0, state.get('expires_at', 0)-self.clock()) if state['status']=='active' else 0,
                        execution_enabled=False)

    def claim(self, owner, request_id):
        if not identity(owner) or not identity(request_id):
            raise ValueError('invalid_board_owner_identity')
        error, lease = None, None
        with self.transaction() as state:
            self.current(state)
            self.expire(state)
            if state['status'] != 'idle':
                error = 'board_control_'+state['status']
            elif self.db.execute('SELECT 1 FROM requests WHERE request_id=?', (request_id,)).fetchone():
                error = 'board_request_already_used'
            else:
                token = secrets.token_hex(32)
                state.clear()
                state.update(epoch=self.epoch, status='active', owner=owner, request_id=request_id,
                             token_sha256=hashlib.sha256(token.encode()).hexdigest(), expires_at=self.clock()+self.ttl)
                self.db.execute('INSERT INTO requests VALUES (?,?)', (request_id, owner))
                self.event('claimed', state)
                lease = dict(epoch=self.epoch, owner=owner, request_id=request_id, token=token)
        if error:
            raise ValueError(error)
        return lease

    def lease_error(self, state, lease):
        if state['status'] != 'active':
            return 'board_control_'+state['status']
        if not isinstance(lease, dict) or not isinstance(lease.get('token'), str):
            return 'invalid_board_owner_lease'
        for key in ('epoch', 'owner', 'request_id'):
            if lease.get(key) != state.get(key):
                return 'board_owner_lease_replaced'
        if not hmac.compare_digest(hashlib.sha256(lease['token'].encode()).hexdigest(), state['token_sha256']):
            return 'board_owner_token_mismatch'

    def check(self, lease, heartbeat=False):
        with self.transaction() as state:
            self.current(state)
            self.expire(state)
            error = self.lease_error(state, lease)
            if error is None and heartbeat:
                state['expires_at'] = self.clock()+self.ttl
        if error:
            raise ValueError(error)

    def finish(self, lease, outcome):
        if outcome not in ('completed', 'rejected_no_motion', 'needs_review', 'outcome_unknown'):
            raise ValueError('invalid_board_control_outcome')
        with self.transaction() as state:
            self.current(state)
            self.expire(state)
            error = self.lease_error(state, lease)
            if error is None:
                state['status'] = 'idle' if outcome in ('completed', 'rejected_no_motion') else 'blocked'
                state['reason'] = outcome
                self.event('finished', state, outcome=outcome)
                if state['status'] == 'idle':
                    state.clear()
                    state.update(epoch=self.epoch, status='idle')
        if error:
            raise ValueError(error)

    def stop(self, reason):
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('board_stop_reason_required')
        with self.transaction() as state:
            self.current(state)
            state.update(status='blocked', reason='operator_stop')
            self.event('stopped', state, observation=reason)

    def review(self, observation, confirmed=False):
        if confirmed is not True or not isinstance(observation, str) or not observation.strip():
            raise ValueError('board_owner_physical_review_required')
        with self.transaction() as state:
            self.current(state)
            self.expire(state)
            if state['status'] != 'blocked':
                raise ValueError('only_blocked_board_owner_can_be_reviewed')
            self.event('operator_review_not_automatic_success', state, observation=observation)
            state.clear()
            state.update(epoch=self.epoch, status='idle')

    def close(self):
        # Closing a service does not release or resolve an active task.
        with self.lock:
            self.db.close()
