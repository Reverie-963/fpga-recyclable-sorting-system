"""One-time group binding; only a matching challenge is retained, no chat logs."""
import argparse
import json
from pathlib import Path
import re
import secrets
import sys
import time
from urllib.parse import urlparse

from qq_merge_notify import get_token, request_json, NotifyError, QQ_API


def bind(output):
    import websocket
    token = get_token()
    gateway = request_json(QQ_API + '/gateway', token='QQBot ' + token).get('url', '')
    parsed = urlparse(gateway)
    if parsed.scheme != 'wss' or not parsed.hostname or not parsed.hostname.endswith('.qq.com'):
        raise NotifyError('QQ returned an unexpected gateway.')
    challenge = 'bind-fpga-' + secrets.token_hex(4)
    print('Connecting to QQ before displaying the binding code...', flush=True)
    ws = websocket.create_connection(gateway, timeout=20)
    try:
        hello = json.loads(ws.recv())
        if hello.get('op') != 10:
            raise NotifyError('QQ gateway did not send Hello.')
        interval = float(hello['d']['heartbeat_interval']) / 1000
        if not 1 <= interval <= 120:
            raise NotifyError('Unexpected QQ heartbeat interval.')
        ws.send(json.dumps({'op': 2, 'd': {'token': 'QQBot ' + token,
                                         'intents': 1 << 25, 'shard': [0, 1]}}))
        deadline = time.monotonic() + 180
        next_heartbeat = time.monotonic()
        sequence = None
        ws.settimeout(1)
        while time.monotonic() < deadline:
            if time.monotonic() >= next_heartbeat:
                ws.send(json.dumps({'op': 1, 'd': sequence}))
                next_heartbeat = time.monotonic() + interval
            try:
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            if not raw:
                raise NotifyError('QQ closed the binding session. Check bot status and retry.')
            message = json.loads(raw)
            if message.get('s') is not None:
                sequence = message['s']
            if message.get('op') in (7, 9):
                raise NotifyError('QQ requires a new binding connection. Retry setup.')
            if message.get('op') == 1:
                ws.send(json.dumps({'op': 1, 'd': sequence}))
            if message.get('t') == 'READY':
                print('Bot is connected. In BlackShore_41784_FPGA, @ your bot and send exactly:', flush=True)
                print(challenge, flush=True)
            if message.get('t') != 'GROUP_AT_MESSAGE_CREATE':
                continue
            data = message.get('d', {})
            content = re.sub(r'<@!?[^>]+>', '', data.get('content', '')).strip()
            if content != challenge:
                continue
            group = data.get('group_openid', '')
            if not re.fullmatch(r'[A-Za-z0-9_-]{8,128}', group):
                raise NotifyError('Binding event has an invalid group OpenID.')
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({'group_openid': group}), encoding='utf-8')
            print('Target group bound. No other group messages were saved.', flush=True)
            return
        raise NotifyError('Binding timed out after 3 minutes. Run setup again.')
    finally:
        ws.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        bind(args.output)
    except Exception as exc:
        # Gateway errors may include server payloads; never print raw exceptions.
        reason = str(exc) if isinstance(exc, NotifyError) else 'Binding failed; check connectivity and bot permissions.'
        print(reason, file=sys.stderr)
        sys.exit(1)
