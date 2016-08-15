import zmq
import zmq.asyncio
import socket
import os


class Publisher:
    """
    A callback that publishes documents to a 0MQ forwarder device.

    Parameters
    ----------
    RE : ``bluesky.RunEngine``
        RunEngine to which the Publish will subscribe
    host : string
        name of host running forwarder_device
        (See bluesky/examples/forwarder_device.py.)
    port : int
        frontend port of Forwarder Device
        (See bluesky/examples/forwarder_device.py.)

    Example
    -------
    >>> Publisher(RE, '127.0.0.1', 5567)
    """
    def __init__(self, RE, host, port):
        self._host = host
        self._port = int(port)
        self._RE = RE
        url = "tcp://%s:%d" % (self.host, self.port)
        self._fmt_string = '{hostname} {pid} {RE_id} {{name}} {{doc}}'.format(
            hostname=socket.gethostname(),
            pid=os.getpid(),
            RE_id=id(RE))
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.connect(url)
        self._subscription_token = RE.subscribe('all', self)

    @property
    def host(self):
        return self._host

    @property
    def port(self):
        return self._port

    def __call__(self, name, doc):
        message = self._fmt_string.format(name=name, doc=doc)
        self._socket.send_string(message)

    def close(self):
        self._RE.unsubscribe(self._subscription_token)
        self._context.destroy()  # close Socket(s); terminate Context
