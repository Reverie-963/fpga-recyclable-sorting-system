import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

TOOLS = Path(__file__).parents[1]
sys.path.insert(0, str(TOOLS))
import qq_bind_group as b


class BindingTests(unittest.TestCase):
    def test_only_random_challenge_binds_and_chat_is_not_saved(self):
        ws = MagicMock()
        ws.recv.side_effect = [
            json.dumps({'op': 10, 'd': {'heartbeat_interval': 45000}}),
            json.dumps({'op': 0, 's': 1, 't': 'READY'}),
            json.dumps({'op': 0, 's': 2, 't': 'GROUP_AT_MESSAGE_CREATE',
                        'd': {'content': 'private unrelated chat', 'group_openid': 'WRONG123'}}),
            json.dumps({'op': 0, 's': 3, 't': 'GROUP_AT_MESSAGE_CREATE',
                        'd': {'content': '<@123> bind-fpga-aabbccdd', 'group_openid': 'RIGHT123'}}),
        ]
        fake = SimpleNamespace(create_connection=MagicMock(return_value=ws),
                               WebSocketTimeoutException=TimeoutError)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'group.local.json'
            with patch.dict(sys.modules, {'websocket': fake}), patch.object(b, 'get_token', return_value='secret'), patch.object(b, 'request_json', return_value={'url': 'wss://api.bot.qq.com/websocket'}), patch.object(b.secrets, 'token_hex', return_value='aabbccdd'), patch('sys.stdout', new_callable=io.StringIO) as log:
                b.bind(output)
                self.assertEqual(json.loads(output.read_text()), {'group_openid': 'RIGHT123'})
                self.assertNotIn('private unrelated chat', log.getvalue())
                self.assertNotIn('secret', log.getvalue())
            ws.close.assert_called_once()

    def test_untrusted_gateway_rejected_before_token_is_sent(self):
        fake = SimpleNamespace(create_connection=MagicMock())
        with patch.dict(sys.modules, {'websocket': fake}), patch.object(b, 'get_token', return_value='secret'), patch.object(b, 'request_json', return_value={'url': 'wss://qq.com.evil.example/ws'}):
            with self.assertRaises(b.NotifyError): b.bind(Path('unused'))
            fake.create_connection.assert_not_called()


if __name__ == '__main__':
    unittest.main()
