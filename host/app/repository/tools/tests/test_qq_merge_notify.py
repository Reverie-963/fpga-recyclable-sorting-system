import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

spec = importlib.util.spec_from_file_location('notify', Path(__file__).parents[1] / 'qq_merge_notify.py')
n = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n)


def event(action='closed'):
    return {'repository': {'full_name': n.REPOSITORY}, 'action': action,
            'pull_request': {'number': 3, 'merged': action == 'closed', 'merge_commit_sha': 'a' * 40,
                             'state': 'open' if action == 'opened' else 'closed',
                             'user': {'login': 'Porter-Ma'}, 'title': '修复 Windows 工具预检',
                             'base': {'ref': 'main', 'repo': {'full_name': n.REPOSITORY}}}}


class NotificationTests(unittest.TestCase):
    def test_only_new_or_merged_main_is_eligible(self):
        for action in ('closed', 'opened'):
            self.assertEqual(n.notification_pr(event(action), 'pull_request_target')['number'], 3)
        for change in ('closed_only', 'other_branch', 'synchronize', 'edited', 'reopened', 'ready_for_review'):
            e = event()
            if change == 'closed_only': e['pull_request']['merged'] = False
            elif change == 'other_branch': e['pull_request']['base']['ref'] = 'dev'
            else: e['action'] = change
            self.assertIsNone(n.notification_pr(e, 'pull_request_target'))
        e = event('opened'); e['pull_request']['base']['ref'] = 'dev'
        self.assertIsNone(n.notification_pr(e, 'pull_request_target'))

    def test_wrong_repository_is_rejected(self):
        for target in ('repository', 'base'):
            e = event()
            if target == 'repository': e['repository']['full_name'] = 'other/repo'
            else: e['pull_request']['base']['repo']['full_name'] = 'other/repo'
            with self.assertRaises(n.NotifyError): n.notification_pr(e, 'pull_request_target')

    def test_malformed_identity_is_rejected(self):
        for key, value in [('number', True), ('number', -1), ('merge_commit_sha', 'invalid')]:
            e = event(); e['pull_request'][key] = value
            with self.assertRaises(n.NotifyError): n.notification_pr(e, 'pull_request_target')

    def test_untrusted_title_and_body_are_not_forwarded(self):
        e = event(); e['pull_request'].update(title='@everyone https://evil.example', body='secret')
        text = n.reminder(n.notification_pr(e, 'pull_request_target'))
        self.assertNotIn('evil', text); self.assertNotIn('@everyone', text)
        self.assertEqual(text, 'PR #3 已合并。\n告诉自己的 AI：把最新代码拉到本地。')

    def test_new_pr_message_has_author_summary_and_canonical_link(self):
        e = event('opened'); e['pull_request']['html_url'] = 'https://evil.example'
        text = n.reminder(n.notification_pr(e, 'pull_request_target'))
        self.assertEqual(text, 'Porter-Ma 提交了 PR #3：修复 Windows 工具预检\n'
                         'https://github.com/Reverie-963/FPGA/pull/3')

    def test_summary_is_bounded_plain_text(self):
        self.assertEqual(n.short_summary('修复\n  预检\t错误'), '修复 预检 错误')
        title = '修复\u202e <@all> [CQ:at,qq=all] https://evil.example @everyone `问题`'
        summary = n.short_summary(title)
        for forbidden in ('evil', '@', '<', '[', '`', '\u202e', '\n'):
            self.assertNotIn(forbidden, summary)
        self.assertEqual(len(n.short_summary('长' * 100)), 60)
        self.assertEqual(n.short_summary('https://evil.example'), '项目更新')

    def test_new_pr_requires_valid_author_and_title(self):
        for login in ('', '@all', 'user\ntext', 'https://evil.example'):
            e = event('opened'); e['pull_request']['user']['login'] = login
            with self.assertRaises(n.NotifyError): n.notification_pr(e, 'pull_request_target')
        e = event('opened'); e['pull_request']['title'] = None
        with self.assertRaises(n.NotifyError): n.notification_pr(e, 'pull_request_target')

    def test_skipped_event_and_preview_do_not_touch_network_or_secrets(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'event.json'
            with patch.dict(os.environ, {}, clear=True), patch.object(n, 'urlopen') as network, patch('sys.stdout', new_callable=io.StringIO):
                path.write_text(json.dumps(event()))
                self.assertEqual(n.main(['--event', str(path), '--event-name', 'pull_request_target', '--dry-run']), 0)
                e = event(); e['pull_request']['merged'] = False; path.write_text(json.dumps(e))
                self.assertEqual(n.main(['--event', str(path), '--event-name', 'pull_request_target']), 0)
                network.assert_not_called()

    def test_manual_catchup_rechecks_github_merge(self):
        e = event(); e['inputs'] = {'pr_number': '3'}
        with patch.dict(os.environ, {'GH_TOKEN': 'test'}), patch.object(n, 'request_json', return_value=e['pull_request']) as api:
            self.assertEqual(n.notification_pr(e, 'workflow_dispatch')['number'], 3)
            self.assertTrue(api.call_args.args[0].endswith('/pulls/3'))
        e['inputs']['pr_number'] = '../secrets'
        with self.assertRaises(n.NotifyError): n.notification_pr(e, 'workflow_dispatch')

    def test_manual_catchup_handles_open_closed_and_wrong_number(self):
        e = {'repository': {'full_name': n.REPOSITORY}, 'inputs': {'pr_number': '3'}}
        with patch.dict(os.environ, {'GH_TOKEN': 'test'}):
            pr = event('opened')['pull_request']
            with patch.object(n, 'request_json', return_value=pr):
                self.assertEqual(n.notification_pr(e, 'workflow_dispatch'), pr)
            pr['state'] = 'closed'
            with patch.object(n, 'request_json', return_value=pr):
                self.assertIsNone(n.notification_pr(e, 'workflow_dispatch'))
            pr['number'] = 4
            with patch.object(n, 'request_json', return_value=pr):
                with self.assertRaises(n.NotifyError): n.notification_pr(e, 'workflow_dispatch')

    def test_http_200_business_error_is_failure(self):
        with patch.object(n, 'urlopen') as opener:
            opener.return_value.__enter__.return_value = io.StringIO('{"code":100016,"message":"secret"}')
            with self.assertRaisesRegex(n.NotifyError, '100016') as caught:
                n.request_json(n.QQ_API + '/app/getAppAccessToken', data={})
            self.assertNotIn('secret', str(caught.exception))

    def test_http_and_network_errors_are_sanitized(self):
        for error in [HTTPError('https://secret.example', 403, 'secret', {}, None), URLError('secret')]:
            with patch.object(n, 'urlopen', side_effect=error):
                with self.assertRaises(n.NotifyError) as caught: n.request_json(n.QQ_API)
                self.assertNotIn('secret', str(caught.exception))

    def test_http_business_code_is_preserved_without_server_message(self):
        error = HTTPError('https://secret.example', 400, 'secret', {},
                          io.BytesIO(b'{"code":40034105,"message":"secret"}'))
        with patch.object(n, 'urlopen', side_effect=error):
            with self.assertRaisesRegex(n.NotifyError, '40034105') as caught:
                n.request_json(n.QQ_API)
            self.assertNotIn('secret', str(caught.exception))

    def test_success_requires_token_and_message_id(self):
        config = {'QQ_BOT_APP_ID': '123', 'QQ_BOT_APP_SECRET': 'test-secret', 'QQ_BOT_GROUP_OPENID': 'ABCD1234'}
        with patch.dict(os.environ, config), patch.object(n, 'request_json', side_effect=[{'access_token': 'test-token'}, {'id': 'message-id'}]) as api, patch('sys.stdout', new_callable=io.StringIO):
            n.send_message('test')
            self.assertEqual(api.call_args.kwargs['token'], 'QQBot test-token')
            self.assertEqual(api.call_args.kwargs['data'], {'msg_type': 0, 'content': 'test'})
        for replies in [[{}], [{'access_token': 'test-token'}, {}]]:
            with patch.dict(os.environ, config), patch.object(n, 'request_json', side_effect=replies):
                with self.assertRaises(n.NotifyError): n.send_message('test')

    def test_missing_secret_fails_before_network(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(n, 'urlopen') as network:
            with self.assertRaises(n.NotifyError): n.send_message('test')
            network.assert_not_called()


if __name__ == '__main__':
    unittest.main()
