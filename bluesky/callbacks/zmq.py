"""
The key classes needed to use 0MQ for multiprocess document communication.

`Publisher` : subscribe this to the RE to emit the documents.  Expects a server to
have a SUBSCRIBE port open to PUB to.

`RemoteDispatcher` : subscribe callbacks to this class in a remote process.  Expects
a server to have a PUB port open to SUBSCRIBE to.

`Proxy` : server that binds ports for Pubslisher to push to and the Dispatcher
to listen to.  Typically this is started with the cli tool ``bluesky-zmq-proxy``

"""

import asyncio
import copy
import pickle
import warnings

from ..run_engine import Dispatcher, DocumentNames


class Bluesky0MQDecodeError(Exception):
    """Custom exception class for things that go wrong reading message from wire."""

    ...


class Publisher:
    """
    A callback that publishes documents to a 0MQ proxy.

    Parameters
    ----------
    address : string or tuple
        Address of a running 0MQ proxy, given either as a string like
        ``'127.0.0.1:5567'`` or as a tuple like ``('127.0.0.1', 5567)``
    prefix : bytes, optional
        User-defined bytestring used to distinguish between multiple
        Publishers. May not contain b' '.
    RE : ``bluesky.RunEngine``, optional
        DEPRECATED.
        RunEngine to which the Publisher will be automatically subscribed
        (and, more importantly, unsubscribed when it is closed).
    zmq : object, optional
        By default, the 'zmq' module is imported and used. Anything else
        mocking its interface is accepted.
    serializer: function, optional
        optional function to serialize data. Default is pickle.dumps

    Examples
    --------

    Publish from a RunEngine to a Proxy running on localhost on port 5567.

    >>> publisher = Publisher('localhost:5567')
    >>> RE = RunEngine({})
    >>> RE.subscribe(publisher)
    """
    def __init__(self, address, *, prefix=b'',
                 RE=None, zmq=None, serializer=pickle.dumps):
        if RE is not None:
            warnings.warn("The RE argument to Publisher is deprecated and "
                          "will be removed in a future release of bluesky. "
                          "Update your code to subscribe this Publisher "
                          "instance to (and, if needed, unsubscribe from) to "
                          "the RunEngine manually.")
        if isinstance(prefix, str):
            raise ValueError("prefix must be bytes, not string")
        if b' ' in prefix:
            raise ValueError("prefix {!r} may not contain b' '".format(prefix))
        if zmq is None:
            import zmq
        if isinstance(address, str):
            address = address.split(':', maxsplit=1)
        self.address = (address[0], int(address[1]))
        self.RE = RE
        url = "tcp://%s:%d" % self.address
        self._prefix = bytes(prefix)
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.connect(url)
        if RE:
            self._subscription_token = RE.subscribe(self)
        self._serializer = serializer

    def __call__(self, name, doc):
        doc = copy.deepcopy(doc)
        message = b' '.join([self._prefix,
                             name.encode(),
                             self._serializer(doc)])
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
        except BaseException:
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
    prefix : bytes, optional
        User-defined bytestring used to distinguish between multiple
        Publishers. If set, messages without this prefix will be ignored.
        If unset, no mesages will be ignored.
    loop : zmq.asyncio.ZMQEventLoop, optional
    zmq : object, optional
        By default, the 'zmq' module is imported and used. Anything else
        mocking its interface is accepted.
    zmq_asyncio : object, optional
        By default, the 'zmq.asyncio' module is imported and used. Anything
        else mocking its interface is accepted.
    deserializer: function, optional
        optional function to deserialize data. Default is pickle.loads

    Examples
    --------

    Print all documents generated by remote RunEngines.

    >>> d = RemoteDispatcher(('localhost', 5568))
    >>> d.subscribe(print)
    >>> d.start()  # runs until interrupted
    """
    def __init__(self, address, *, prefix=b'',
                 loop=None, zmq=None, zmq_asyncio=None,
                 deserializer=pickle.loads, strict=False):
        if isinstance(prefix, str):
            raise ValueError("prefix must be bytes, not string")
        if b' ' in prefix:
            raise ValueError("prefix {!r} may not contain b' '".format(prefix))
        self._prefix = prefix
        if zmq is None:
            import zmq
        if zmq_asyncio is None:
            import zmq.asyncio as zmq_asyncio
        if isinstance(address, str):
            address = address.split(':', maxsplit=1)
        self._deserializer = deserializer
        self.address = (address[0], int(address[1]))

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
        self.closed = False
        self._strict = strict
        super().__init__()

    async def _poll(self):
        our_prefix = self._prefix  # local var to save an attribute lookup
        while True:
            message = await self._socket.recv()
            try:
                prefix, name, doc = message.split(b' ', 2)
            except ValueError as e:
                if self._strict:
                    raise Bluesky0MQDecodeError from e
                else:
                    print(f"The message {message} could not be split into "
                          "three parts by b' '.  Dropping message on floor "
                          "and continuing"
                          f"\n\n{e}")
                    continue

            try:
                name = name.decode()
            except UnicodeDecodeError as e:
                if self._strict:
                    raise Bluesky0MQDecodeError from e
                else:
                    print(f"The name {name} can not be decoded as utf-8. "
                          "Dropping message on the floor and continuing. "
                          f"\n\n{e}")
                    continue
            if (not our_prefix) or prefix == our_prefix:
                try:
                    doc = self._deserializer(doc)
                except Exception as e:
                    if self._strict:
                        raise Bluesky0MQDecodeError from e
                    else:
                        if len(doc) > 1024:
                            msg_doc = doc[:1024] + b'--SNIPPED--'
                        else:
                            msg_doc = doc
                        print(f"Failed to deserialize the {name} document "
                              f"{msg_doc} using {self._deserializer}. "
                              "Dropping on floor and continuing"
                              f"\n\n{e}")
                        continue
                self.loop.call_soon(self.process, DocumentNames[name], doc)

    def start(self):
        if self.closed:
            raise RuntimeError("This RemoteDispatcher has already been "
                               "started and interrupted. Create a fresh "
                               f"instance with {self!r}")
        try:
            self._task = self.loop.create_task(self._poll())
            self.loop.run_until_complete(self._task)
            task_exception = self._task.exception()
            if task_exception is not None:
                raise task_exception
        except BaseException:
            self.stop()
            raise

    def stop(self):
        if self._task is not None:
            self._task.cancel()
        self.closed = True
