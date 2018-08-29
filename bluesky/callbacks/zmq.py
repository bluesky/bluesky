import asyncio
import copy
import multiprocessing
import os
import pickle
import socket
import time
from ..run_engine import Dispatcher, DocumentNames
from ..utils import apply_to_dict_recursively, sanitize_np


class Publisher:
    """
    A callback that publishes documents to a 0MQ proxy.

    Parameters
    ----------
    address : string or tuple
        Address of a running 0MQ proxy, given either as a string like
        ``'127.0.0.1:5567'`` or as a tuple like ``('127.0.0.1', 5567)``
    RE : ``bluesky.RunEngine``, optional
        RunEngine to which the Publisher will be automatically subscribed
        (and, more importantly, unsubscribed when it is closed).
    zmq : object, optional
        By default, the 'zmq' module is imported and used. Anything else
        mocking its interface is accepted.
    serializer: function, optional
        optional function to serialize data. Default is pickle.dumps

    Example
    -------

    Publish from a RunEngine to a Proxy running on localhost on port 5567.

    >>> RE = RunEngine({})
    >>> publisher = Publisher('localhost:5567', RE=RE)
    """
    def __init__(self, address, *, RE=None, zmq=None, serializer=pickle.dumps):
        if zmq is None:
            import zmq
        if isinstance(address, str):
            address = address.split(':', maxsplit=1)
        self.address = (address[0], int(address[1]))
        self.RE = RE
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
        url = "tcp://%s:%d" % self.address
        self._prefix = b'%s %d %d ' % (self.hostname.encode(),
                                       self.pid, id(RE))
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.connect(url)
        if RE:
            self._subscription_token = RE.subscribe(self)
        self._serializer = serializer

    def __call__(self, name, doc):
        doc = copy.deepcopy(doc)
        apply_to_dict_recursively(doc, sanitize_np)
        message = bytes(self._prefix)  # making a copy
        message += b' '.join([name.encode(), self._serializer(doc)])
        self._socket.send(message)

    def close(self):
        if self.RE:
            self.RE.unsubscribe(self._subscription_token)
        self._context.destroy()  # close Socket(s); terminate Context


class Proxy:
    """
    Start a 0MQ proxy on the local host.

    Parameters
    ----------
    in_port : int, optional
        Port that RunEngines should broadcast to. If None, a random port is
        used.
    out_port : int, optional
        Port that subscribers should subscribe to. If None, a random port is
        used.
    zmq : object, optional
        By default, the 'zmq' module is imported and used. Anything else
        mocking its interface is accepted.

    Attributes
    ----------
    in_port : int
        Port that RunEngines should broadcast to.
    out_port : int
        Port that subscribers should subscribe to.
    closed : boolean
        True if the Proxy has already been started and subsequently
        interrupted and is therefore unusable.

    Examples
    --------

    Run on specific ports.

    >>> proxy = Proxy(5567, 5568)
    >>> proxy
    Proxy(in_port=5567, out_port=5568)
    >>> proxy.start()  # runs until interrupted

    Run on random ports, and access those ports before starting.

    >>> proxy = Proxy()
    >>> proxy
    Proxy(in_port=56504, out_port=56505)
    >>> proxy.in_port
    56504
    >>> proxy.out_port
    56505
    >>> proxy.start()  # runs until interrupted
    """
    def __init__(self, in_port=None, out_port=None, *, zmq=None):
        if zmq is None:
            import zmq
        self.zmq = zmq
        self.closed = False
        try:
            context = zmq.Context(1)
            # Socket facing clients
            frontend = context.socket(zmq.SUB)
            if in_port is None:
                in_port = frontend.bind_to_random_port("tcp://*")
            else:
                frontend.bind("tcp://*:%d" % in_port)

            frontend.setsockopt_string(zmq.SUBSCRIBE, "")

            # Socket facing services
            backend = context.socket(zmq.PUB)
            if out_port is None:
                out_port = backend.bind_to_random_port("tcp://*")
            else:
                backend.bind("tcp://*:%d" % out_port)
        except:
            # Clean up whichever components we have defined so far.
            try:
                frontend.close()
            except NameError:
                ...
            try:
                backend.close()
            except NameError:
                ...
            context.term()
            raise
        else:
            self.in_port = in_port
            self.out_port = out_port
            self._frontend = frontend
            self._backend = backend
            self._context = context
        
    def start(self):
        if self.closed:
            raise RuntimeError("This Proxy has already been started and "
                               "interrupted. Create a fresh instance with "
                               "{}".format(repr(self)))
        try:
            self.zmq.device(self.zmq.FORWARDER, self._frontend, self._backend)
        finally:
            self.closed = True
            self._frontend.close()
            self._backend.close()
            self._context.term()

    def __repr__(self):
        return ("{}(in_port={in_port}, out_port={out_port})"
                "".format(type(self).__name__, **vars(self)))


class RemoteDispatcher(Dispatcher):
    """
    Dispatch documents received over the network from a 0MQ proxy.

    Parameters
    ----------
    address : tuple
        Address of a running 0MQ proxy, given either as a string like
        ``'127.0.0.1:5567'`` or as a tuple like ``('127.0.0.1', 5567)``
    hostname : string, optional
        A filter: only process documents from a host with this name.
    pid : int, optional
        A filter: only process documents from a process with this pid.
    run_engine_id : int, optional
        A filter: only process documents from a RunEngine with this Python id
        (memory address).
    loop : zmq.asyncio.ZMQEventLoop, optional
    zmq : object, optional
        By default, the 'zmq' module is imported and used. Anything else
        mocking its interface is accepted.
    zmq_asyncio : object, optional
        By default, the 'zmq.asyncio' module is imported and used. Anything
        else mocking its interface is accepted.
    deserializer: function, optional
        optional function to deserialize data. Default is pickle.loads

    Example
    -------

    Print all documents generated by remote RunEngines.

    >>> d = RemoteDispatcher(('localhost', 5568))
    >>> d.subscribe(print)
    >>> d.start()  # runs until interrupted
    """
    def __init__(self, address, *, hostname=None, pid=None, run_engine_id=None,
                 loop=None, zmq=None, zmq_asyncio=None,
                 deserializer=pickle.loads):
        if zmq is None:
            import zmq
        if zmq_asyncio is None:
            import zmq.asyncio as zmq_asyncio
        if isinstance(address, str):
            address = address.split(':', maxsplit=1)
        self._deserializer = deserializer
        self.address = (address[0], int(address[1]))
        self.hostname = hostname
        self.pid = pid
        self.run_engine_id = run_engine_id

        if loop is None:
            loop = zmq_asyncio.ZMQEventLoop()
        self.loop = loop
        asyncio.set_event_loop(self.loop)
        self._context = zmq_asyncio.Context()
        self._socket = self._context.socket(zmq.SUB)
        url = "tcp://%s:%d" % self.address
        self._socket.connect(url)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._task = None

        def is_our_message(_hostname, _pid, _RE_id):
            # Close over filters and decide if this message applies to this
            # RemoteDispatcher.
            return ((hostname is None or hostname == _hostname)
                    and (pid is None or pid == _pid)
                    and (run_engine_id is None or run_engine_id ==
                         run_engine_id))
        self._is_our_message = is_our_message

        super().__init__()

    @asyncio.coroutine
    def _poll(self):
        while True:
            message = yield from self._socket.recv()
            hostname, pid, RE_id, name, doc = message.split(b' ', 4)
            hostname = hostname.decode()
            pid = int(pid)
            RE_id = int(RE_id)
            name = name.decode()
            if self._is_our_message(hostname, pid, RE_id):
                doc = self._deserializer(doc)
                self.loop.call_soon(self.process, DocumentNames[name], doc)

    def start(self):
        try:
            self._task = self.loop.create_task(self._poll())
            self.loop.run_forever()
        except:
            self.stop()
            raise

    def stop(self):
        if self._task is not None:
            self._task.cancel()
            self.loop.stop()
        self._task = None
