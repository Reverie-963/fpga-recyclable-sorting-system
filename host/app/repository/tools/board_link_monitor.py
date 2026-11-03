"""Read-only coordinator diagnostics, independent of visual service lifetime.

A failed probe is not an operator stop or proof that the independent main
controller lost its lease. Never tear down camera/vision services or replay a
control request in response to this diagnostic.
"""
import http.client
import json
import time


class BoardLinkMonitor:
    def __init__(self, board, expected_root, clock=time.monotonic):
        self.board = board
        self.expected_root = expected_root
        self.clock = clock
        self.last_success = None
        self.failures = 0

    def poll(self):
        reason = None
        try:
            code, raw, _ = self.board.request('/board-control')
            if code != 200:
                reason = 'board_probe_http_' + str(code)
            else:
                value = json.loads(raw)
                if (not isinstance(value, dict) or value.get('compute_location') != 'pynq_ps' or
                        value.get('board_root') != self.expected_root):
                    reason = 'board_probe_identity_changed'
        except (OSError, EOFError, ValueError, http.client.HTTPException) as exc:
            # PinnedBoardHTTP translates SSH failures to ConnectionError.
            reason = 'board_probe_' + type(exc).__name__
        if reason is None:
            self.last_success = self.clock()
            self.failures = 0
        else:
            self.failures += 1
        return dict(available=reason is None, reason=reason,
                    consecutive_failures=self.failures,
                    seconds_since_success=(None if self.last_success is None else
                                           max(0, self.clock() - self.last_success)),
                    read_only=True, automatic_reconnect=False,
                    control_requests_replayed=False)
