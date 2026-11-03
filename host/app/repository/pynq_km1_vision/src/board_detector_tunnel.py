"""Explicit, finite-lifetime reverse SSH forwarding for the detector relay.

The factory must create a new dedicated SSH client. This class owns and closes
that client; never pass a shared camera/service SSH connection. No reconnect,
subprocess, remote command, camera or actuator endpoint is provided.
"""
import select
import socket
import threading

from .board_detector_transport import make_server


class DetectorTunnel:
    def __init__(self, connect_new, token, upstream='http://127.0.0.1:8778', *,
                 port=0, upstream_request=None, max_channels=2, max_inflight=1, strip_overlay=False, marker_detector=None):
        if type(port) is not int or not (port == 0 or 1024 <= port <= 65535):
            raise ValueError('invalid_detector_tunnel_port')
        if type(max_channels) is not int or not 1 <= max_channels <= 4:
            raise ValueError('invalid_detector_channel_limit')
        self.connect_new, self.token, self.upstream = connect_new, token, upstream
        self.port, self.upstream_request = port, upstream_request
        self.max_inflight = max_inflight
        self.strip_overlay = strip_overlay
        self.marker_detector = marker_detector
        self.client = self.transport = self.server = self.server_thread = None
        self.registered_port = None
        self.state = 'new'
        self.stopped = threading.Event()
        self.guard = threading.Lock()
        self.slots = threading.BoundedSemaphore(max_channels)
        self.channels, self.sockets, self.workers = set(), set(), set()

    def start(self):
        if self.state != 'new':
            raise RuntimeError('detector_tunnel_cannot_restart')
        self.state = 'starting'
        try:
            self.client = self.connect_new()
            self.transport = self.client.get_transport()
            if self.transport is None or not self.transport.is_active():
                raise ConnectionError('detector_ssh_unavailable')
            self.server = make_server(self.port, self.token, self.upstream,
                                      upstream_request=self.upstream_request, max_inflight=self.max_inflight,
                                      strip_overlay=self.strip_overlay,marker_detector=self.marker_detector)
            self.port = self.server.server_port
            # Identical ports preserve HTTP Host validation without broadening
            # the relay allowlist or rewriting the signed request envelope.
            allocated = self.transport.request_port_forward('127.0.0.1', self.port, self._accept)
            if type(allocated) is not int or not 1 <= allocated <= 65535:
                raise ConnectionError('invalid_detector_forward_port')
            self.registered_port = allocated
            if allocated != self.port:
                raise ConnectionError('detector_forward_port_changed')
            self.server_thread = threading.Thread(target=self.server.serve_forever,
                kwargs=dict(poll_interval=.05), daemon=True, name='detector-relay-http')
            self.server_thread.start()
            self.state = 'active'
            self.require_connected()
            return self
        except BaseException:
            self.close()
            raise

    def require_connected(self):
        if (self.state != 'active' or self.stopped.is_set() or self.client is None or
                self.client.get_transport() is not self.transport or not self.transport.is_active()):
            raise ConnectionError('detector_tunnel_disconnected_no_reconnect')

    @property
    def board_url(self):
        self.require_connected()
        return 'http://127.0.0.1:'+str(self.port)

    def _accept(self, channel, origin, server_address):
        try:
            self.require_connected()
            if tuple(server_address) != ('127.0.0.1', self.port):
                raise ConnectionError('detector_forward_address_changed')
        except (ConnectionError, TypeError):
            channel.close()
            return
        if not self.slots.acquire(blocking=False):
            channel.close()
            return
        with self.guard:
            if self.stopped.is_set():
                self.slots.release()
                channel.close()
                return
            self.channels.add(channel)
            worker = threading.Thread(target=self._pump, args=(channel,), daemon=True,
                                      name='detector-relay-channel')
            self.workers.add(worker)
            worker.start()

    def _pump(self, channel):
        local = None
        try:
            self.require_connected()
            local = socket.create_connection(('127.0.0.1', self.port), timeout=3)
            local.settimeout(5)
            channel.settimeout(5)
            with self.guard:
                if self.stopped.is_set():
                    return
                self.sockets.add(local)
            transferred = {channel: 0, local: 0}
            while not self.stopped.is_set():
                self.require_connected()
                ready, _, _ = select.select([channel, local], [], [], .1)
                for source in ready:
                    data = source.recv(65536)
                    if not data:
                        return
                    transferred[source] += len(data)
                    if transferred[source] > 16_000_000:
                        raise ValueError('detector_tunnel_exchange_too_large')
                    (local if source is channel else channel).sendall(data)
        except (OSError, ValueError, ConnectionError):
            pass  # Fail this exchange. No retry, reconnect or alternate route.
        finally:
            if local is not None:
                local.close()
            channel.close()
            with self.guard:
                self.sockets.discard(local)
                self.channels.discard(channel)
                self.workers.discard(threading.current_thread())
            self.slots.release()

    def close(self):
        if self.state == 'closed':
            return
        self.stopped.set()
        self.state = 'closing'
        # Cancel only our registered listener; closing our dedicated SSH client
        # also removes it if the connection is already broken or cancel fails.
        try:
            if self.registered_port is not None and self.transport is not None and self.transport.is_active():
                self.transport.cancel_port_forward('127.0.0.1', self.registered_port)
        except Exception:
            pass
        if self.client is not None:
            self.client.close()
        with self.guard:
            streams, workers = list(self.channels | self.sockets), list(self.workers)
        for stream in streams:
            try:
                stream.shutdown(socket.SHUT_RDWR)
            except (OSError, TypeError):
                pass
            stream.close()
        if self.server is not None:
            if self.server_thread is not None and self.server_thread.is_alive():
                self.server.shutdown()
                self.server_thread.join(2)
            self.server.server_close()
        for worker in workers:
            if worker is not threading.current_thread():
                worker.join(2)
        self.state = 'closed'

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.close()
