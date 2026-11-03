"""Send concise PR notifications through the official QQ Bot API."""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

QQ_API = 'https://api.bot.qq.com'
REPOSITORY = 'Reverie-963/FPGA'


class NotifyError(RuntimeError):
    pass


def request_json(url, *, data=None, token=None):
    headers = {'Accept': 'application/json', 'User-Agent': 'FPGA-merge-notifier'}
    if token:
        headers['Authorization'] = token
    if data is not None:
        headers['Content-Type'] = 'application/json'
    req = Request(url, data=None if data is None else json.dumps(data).encode('utf-8'),
                  headers=headers)
    try:
        with urlopen(req, timeout=20) as response:
            result = json.load(response)
    except HTTPError as exc:
        # Do not print remote bodies, request URLs, tokens or secrets.
        detail = ''
        try:
            failure = json.loads(exc.read(4096))
            if isinstance(failure, dict) and type(failure.get('code')) is int:
                detail = f" (business code {failure['code']})"
        except (ValueError, AttributeError, TypeError, OSError):
            pass
        raise NotifyError(f'API HTTP failure: {exc.code}{detail}') from None
    except (URLError, TimeoutError, OSError):
        raise NotifyError('API network failure; delivery may be uncertain. Check before resending.') from None
    except (ValueError, UnicodeError):
        raise NotifyError('API returned invalid JSON.') from None
    if not isinstance(result, dict):
        raise NotifyError('API returned an unexpected response.')
    if result.get('code') not in (None, 0):
        code = result['code']
        safe_code = str(code) if isinstance(code, int) else 'unknown'
        raise NotifyError(f'API business failure: {safe_code}')
    return result


def required_env(name):
    value = os.environ.get(name, '').strip()
    if not value:
        raise NotifyError(f'Missing configuration: {name}')
    return value


def get_token():
    reply = request_json(QQ_API + '/app/getAppAccessToken', data={
        'appId': required_env('QQ_BOT_APP_ID'),
        'clientSecret': required_env('QQ_BOT_APP_SECRET'),
    })
    token = reply.get('access_token')
    if not isinstance(token, str) or not token:
        raise NotifyError('QQ did not return an access token.')
    return token


def send_message(content):
    group = required_env('QQ_BOT_GROUP_OPENID')
    if not re.fullmatch(r'[A-Za-z0-9_-]{8,128}', group):
        raise NotifyError('Invalid group OpenID; use the bound OpenID, not the QQ group number.')
    token = get_token()
    response = request_json(QQ_API + '/v2/groups/' + quote(group, safe='') + '/messages',
                            token='QQBot ' + token, data={'msg_type': 0, 'content': content})
    if not response.get('id'):
        raise NotifyError('QQ did not acknowledge delivery with a message ID.')
    print('QQ accepted the group notification. This does not confirm that members read it.')


def notification_pr(event, event_name):
    if event.get('repository', {}).get('full_name') != REPOSITORY:
        raise NotifyError('Unexpected repository; no notification sent.')
    if event_name in ('pull_request', 'pull_request_target'):
        action = event.get('action')
        if action not in ('opened', 'closed'):
            return None
        pr = event.get('pull_request', {})
        if action == 'opened' and (pr.get('state') != 'open' or pr.get('merged') is True):
            return None
        if action == 'closed' and pr.get('merged') is not True:
            return None
    elif event_name == 'workflow_dispatch':
        number = str(event.get('inputs', {}).get('pr_number', ''))
        if not re.fullmatch(r'[1-9][0-9]*', number):
            raise NotifyError('Manual notification requires a positive PR number.')
        pr = request_json(f'https://api.github.com/repos/{REPOSITORY}/pulls/{number}',
                          token='Bearer ' + required_env('GH_TOKEN'))
        if pr.get('number') != int(number):
            raise NotifyError('Unexpected PR number in GitHub response.')
        if pr.get('merged') is not True and pr.get('state') != 'open':
            return None
    else:
        raise NotifyError('Unsupported event; no notification sent.')
    if pr.get('base', {}).get('ref') != 'main':
        return None
    if pr.get('base', {}).get('repo', {}).get('full_name') != REPOSITORY:
        raise NotifyError('Unexpected PR destination; no notification sent.')
    if type(pr.get('number')) is not int or pr['number'] <= 0:
        raise NotifyError('Invalid PR number.')
    if pr.get('merged') is True and not re.fullmatch(r'[0-9a-f]{40}', pr.get('merge_commit_sha') or ''):
        raise NotifyError('Missing or invalid merge commit.')
    if pr.get('merged') is not True:
        login = pr.get('user', {}).get('login', '')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9-]{0,38}(?:\[bot\])?', login):
            raise NotifyError('Invalid PR author.')
        if not isinstance(pr.get('title'), str):
            raise NotifyError('Missing PR title.')
    return pr


def short_summary(title):
    # Titles are untrusted text: flatten them and strip URLs/mention markup.
    title = ''.join(c for c in title if unicodedata.category(c) != 'Cf')
    title = re.sub(r'(?i)(?:https?://|www\.)\S+', '', title)
    title = re.sub(r'<[^>]*>|\[CQ:[^\]]*\]', '', title)
    title = re.sub(r'[@<>\[\]`]', '', title)
    title = ' '.join(''.join(c if not unicodedata.category(c).startswith('C') else ' '
                            for c in title).split())
    return (title[:59] + '…') if len(title) > 60 else (title or '项目更新')


def reminder(pr):
    if pr.get('merged') is True:
        return f"PR #{pr['number']} 已合并。\n告诉自己的 AI：把最新代码拉到本地。"
    # Canonical link only; never forward a URL supplied in the PR title/body.
    return (f"{pr['user']['login']} 提交了 PR #{pr['number']}：{short_summary(pr['title'])}\n"
            f"https://github.com/{REPOSITORY}/pull/{pr['number']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--event', default=os.environ.get('GITHUB_EVENT_PATH'))
    parser.add_argument('--event-name', default=os.environ.get('GITHUB_EVENT_NAME'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--test-message', action='store_true', help='Send one explicit setup test')
    args = parser.parse_args(argv)
    if args.test_message:
        content = 'FPGA 通知测试：新 PR 和合并消息将在此提醒。'
    else:
        if not args.event:
            raise NotifyError('An event file is required.')
        try:
            event = json.loads(Path(args.event).read_text(encoding='utf-8'))
        except (OSError, ValueError):
            raise NotifyError('Cannot read the event file.') from None
        pr = notification_pr(event, args.event_name)
        if pr is None:
            print('Skipped: this is not a new or merged PR targeting main.')
            return 0
        content = reminder(pr)
    if args.dry_run:
        print(content)
    else:
        send_message(content)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except NotifyError as exc:
        print(f'Notification failed: {exc}', file=sys.stderr)
        sys.exit(1)
