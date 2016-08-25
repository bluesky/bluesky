import multiprocessing
import ast
import zmq
import zmq.asyncio
import asyncio
import time
from ..utils import expiring_function
from ..run_engine import Dispatcher, DocumentNames


class RemoteDispatcher(Dispatcher):
    """
    Dispatch documents received over a socket.

    Parameters
    ----------
    host : string
        name of host running forwarder_device
        (See bluesky/examples/forwarder_device.py.)
    port : int, optional
        default 5560
    filter_hostname : string, optional
        only process documents from a host with this name
    filter_pid : int, optional
        only process documents from a process with this pid
    filter_run_engine_id : int, optional
        only process documents from a RunEngine with this Python id
        (memory address)
    loop : zmq.asyncio.ZMQEventLoop, optional

    Example
    -------
    Create a LivePlot and feed it documents published to a 0MQ forwarder
    device at running on localhost at port 5560.

    >>> from bluesky.qt_kicker import install_qt_kicker
    >>> install_qt_kicker()
    >>> dispatcher = Dispatcher('localhost', 5568)
    >>> dispatcher.subscribe(LivePlot('y', 'x'))
    >>> dispatcher.start()
    """
    def __init__(self, host, port, *, filter_hostname=None, filter_pid=None,
                 filter_run_engine_id=None, loop=None):
        if loop is None:
            loop = zmq.asyncio.ZMQEventLoop()
        self._loop = loop
        asyncio.set_event_loop(self._loop)
        self._host = host
        self._port = int(port)
        self._context = zmq.asyncio.Context()
        self._socket = self._context.socket(zmq.SUB)
        url = "tcp://%s:%d" % (self.host, self.port)
        self._socket.connect(url)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._task = None

        def is_our_message(hostname, pid, RE_id):
            # Close over filters and decide if this message applies to this
            # RemoteDispatcher.
            return ((filter_hostname is None
                     or filter_hostname == hostname)
                    and (filter_pid is None
                         or filter_pid == pid)
                    and (filter_run_engine_id is None
                         or filter_run_engine_id == RE_id))
        self._is_our_message = is_our_message

        super().__init__()

    @property
    def host(self):
        return self._host

    @property
    def port(self):
        return self._port

    @property
    def loop(self):
        return self._loop

    @asyncio.coroutine
    def _poll(self):
        while True:
            message = yield from self._socket.recv()
            hostname, pid, RE_id, name, doc = message.decode().split(' ', 4)
            if self._is_our_message(hostname, pid, RE_id):
                doc = ast.literal_eval(doc)
                self._loop.call_soon(self.process, DocumentNames[name], doc)

    def start(self):
        self._task = self._loop.create_task(self._poll())
        self._loop.run_forever()

    def stop(self):
        if self._task is not None:
            self._task.cancel()
        self._task = None
