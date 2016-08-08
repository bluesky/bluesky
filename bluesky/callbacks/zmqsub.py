import multiprocessing
import ast
import zmq
import zmq.asyncio
import asyncio
import time
from ..utils import expiring_function
from ..run_engine import Dispatcher, DocumentNames


_loop = zmq.asyncio.ZMQEventLoop()
asyncio.set_event_loop(_loop)


class RemoteDispatcher(Dispatcher):
    def __init__(self, host, port, *, filter_hostname=None, filter_pid=None,
                 filter_run_engine_id=None, event_timeout=None):
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
        event_timeout : float
            expiring time for skipping an event

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
        self._context = zmq.asyncio.Context()
        self._socket = self._context.socket(zmq.SUB)
        url = "tcp://%s:%d" % (host, port)
        self._socket.connect(url)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.event_timeout = event_timeout
        self._host = host
        self._port = port

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

    @asyncio.coroutine
    def _poll(self):
        message = yield from self._socket.recv()
        hostname, pid, RE_id, name, doc = message.decode().split(' ', 4)
        doc = ast.literal_eval(doc)
        self._loop.create_task(self._poll())
        start_time = time.time()  # for timeout, if used below
        yield from asyncio.sleep(0)  # Give scheduler second chance to switch.
        if self._is_our_message(hostname, pid, RE_id):
            if self.event_timeout is None or name != 'event':
                self._loop.call_soon(self.process, DocumentNames[name],
                                     doc)
                pass
            else:
                # This dummy function will execute self.process(name, doc)
                # as long as it is called within `timeout` seconds
                # `start_time`.
                dummy = expiring_function(self.process, self._loop,
                                          DocumentNames[name], doc)
                self._loop.call_soon(dummy, start_time, self.event_timeout)

    def start(self):
        self._loop.create_task(self._poll())
        self._loop.run_forever()
