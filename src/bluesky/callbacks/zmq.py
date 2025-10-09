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
import logging
import pickle
import warnings
from pathlib import Path
from typing import NamedTuple, Union

from ..run_engine import Dispatcher, DocumentNames

logger = logging.getLogger(__name__)


class ServerCurve(NamedTuple):
    # path to the secret key for the server
    secret_path: Path
    # path to folder of client's public keys.  If None, allow all clients
    client_public_keys: Union[Path, None]
    # set of ip addresses to allow
    allow: Union[set[str], None]


class ClientCurve(NamedTuple):
    # path to the secret key for the server
    secret_path: Path
    # path to the servers public key
    server_public_key: Path


def _normalize_address(inp: Union[str, tuple, int, None]):
    if isinstance(inp, str):
        if "://" in inp:
            protocol, _, rest_str = inp.partition("://")
        else:
            protocol = "tcp"
            rest_str = inp
    elif isinstance(inp, tuple):
        if inp[0] in ["tcp", "ipc"]:
            protocol, *rest = inp
        else:
            protocol = "tcp"
            rest = list(inp)
        if protocol == "tcp":
            if len(rest) == 2:
                rest_str = ":".join(str(r) for r in rest)
            else:
                (rest_str,) = rest
        else:
            (rest_str,) = rest
    elif isinstance(inp, int):
        protocol = "tcp"
        rest_str = f"0.0.0.0:{inp}"
    elif inp is None:
        protocol = "tcp"
        rest_str = "*"

    else:
        raise TypeError(f"Input expected to be int, str, or tuple, not {type(inp)}")

    return f"{protocol}://{rest_str}"


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
    curve_config: ClientCurve, optional
        CURVE security configuration for client authentication.

    Examples
    --------

    Publish from a RunEngine to a Proxy running on localhost on port 5567.

    >>> publisher = Publisher('localhost:5567')
    >>> RE = RunEngine({})
    >>> RE.subscribe(publisher)
    """

    def __init__(self, address, *, prefix=b"", RE=None, zmq=None, serializer=pickle.dumps, curve_config=None):
        if RE is not None:
            warnings.warn(  # noqa: B028
                "The RE argument to Publisher is deprecated and "
                "will be removed in a future release of bluesky. "
                "Update your code to subscribe this Publisher "
                "instance to (and, if needed, unsubscribe from) to "
                "the RunEngine manually."
            )
        if isinstance(prefix, str):
            raise ValueError("prefix must be bytes, not string")
        if b" " in prefix:
            raise ValueError(f"prefix {prefix!r} may not contain b' '")
        if zmq is None:
            import zmq
            import zmq.auth

        self.address = _normalize_address(address)
        self.RE = RE

        self._prefix = bytes(prefix)
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)

        if curve_config is not None:
            # Load the client cert pair
            client_public, client_secret = zmq.auth.load_certificate(curve_config.secret_path)
            self._socket.setsockopt(zmq.CURVE_PUBLICKEY, client_public)
            if client_secret is None:
                raise ValueError("The client secret key could not be found.")
            self._socket.setsockopt(zmq.CURVE_SECRETKEY, client_secret)

            # Load the server public key and register with the socket
            server_key, _ = zmq.auth.load_certificate(curve_config.server_public_key)
            self._socket.setsockopt(zmq.CURVE_SERVERKEY, server_key)

        self._socket.connect(self.address)
        if RE:
            self._subscription_token = RE.subscribe(self)
        self._serializer = serializer

    def __call__(self, name, doc):
        doc = copy.deepcopy(doc)
        message = b" ".join([self._prefix, name.encode(), self._serializer(doc)])
        self._socket.send(message)

    def close(self):
        if self.RE:
            self.RE.unsubscribe(self._subscription_token)
        self._socket.close()
        self._context.destroy()  # close Socket(s); terminate Context


class Proxy:
    """
    Start a 0MQ proxy on the local host.

    The addresses can be specified flexibly.  It is best to use
    a domain_socket (available on unix):

     - ``'icp:///tmp/domain_socket'``
     - ``('ipc', '/tmp/domain_socket')``

    tcp sockets are also supported:

     - ``'tcp://localhost:6557'``
     - ``6657``  (implicitly binds to ``'tcp://localhost:6657'``
     - ``('tcp', 'localhost', 6657)``
     - ``('localhost', 6657)``

    Parameters
    ----------
    in_address : str or tuple or int, optional
        Address that RunEngines should broadcast to.

        If None, a random tcp port on all interfaces is used.

    out_address : str or tuple or int, optional
        Address that subscribers should subscribe to.

        If None, a random tcp port on all interfaces is used.

    zmq : object, optional
        By default, the 'zmq' module is imported and used. Anything else
        mocking its interface is accepted.

    Attributes
    ----------
    in_address: int or str or tuple
        Port that RunEngines should broadcast to.
    out_address : int or str or tuple
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

    @staticmethod
    def configure_server_socket(
        ctx,
        sock_type,
        address: Union[str, tuple, int, None],
        curve: Union[ServerCurve, ClientCurve, None],
        zmq,
        auth_class,
        bind: bool = True,
    ):
        socket = ctx.socket(sock_type)
        norm_address = _normalize_address(address)
        logger.debug(f"Creating socket of type {sock_type} for address {norm_address}, bind={bind}")
        random_port = False
        if norm_address.startswith("tcp"):
            if ":" not in norm_address[6:]:
                random_port = True

        if curve is not None:
            if bind:
                # Server mode - expect ServerCurve
                if not isinstance(curve, ServerCurve):
                    raise TypeError("When bind=True, curve must be a ServerCurve instance")
                logger.debug(f"Configuring CURVE server security with secret_path={curve.secret_path}")
                # build authenticator
                auth = auth_class(ctx)
                auth.start()
                logger.debug("Started ZMQ authenticator")
                if curve.allow is not None:
                    auth.allow(*curve.allow)
                    logger.debug(f"Configured IP address allowlist: {curve.allow}")

                # Tell the authenticator how to handle CURVE requests
                if curve.client_public_keys is None:
                    # accept any client that knows the public key
                    auth.configure_curve(domain="*", location=zmq.auth.CURVE_ALLOW_ANY)
                    logger.debug("Configured CURVE to allow any client with valid public key")
                else:
                    auth.configure_curve(domain="*", location=curve.client_public_keys)
                    logger.debug(f"Configured CURVE client public keys from: {curve.client_public_keys}")

                # get public and private keys from the certificate
                server_public, server_secret = zmq.auth.load_certificate(curve.secret_path)
                # attach them to the socket
                socket.setsockopt(zmq.CURVE_PUBLICKEY, server_public)
                socket.setsockopt(zmq.CURVE_SECRETKEY, server_secret)
                socket.setsockopt(zmq.CURVE_SERVER, True)
                logger.debug("Applied CURVE keys and enabled CURVE server mode")
            else:
                # Client mode - expect ClientCurve
                if not isinstance(curve, ClientCurve):
                    raise TypeError("When bind=False, curve must be a ClientCurve instance")
                logger.debug(f"Configuring CURVE client security with secret_path={curve.secret_path}")

                # Load the client cert pair
                client_public, client_secret = zmq.auth.load_certificate(curve.secret_path)
                socket.setsockopt(zmq.CURVE_PUBLICKEY, client_public)
                if client_secret is None:
                    raise ValueError("The client secret key could not be found.")
                socket.setsockopt(zmq.CURVE_SECRETKEY, client_secret)

                # Load the server public key and register with the socket
                server_key, _ = zmq.auth.load_certificate(curve.server_public_key)
                socket.setsockopt(zmq.CURVE_SERVERKEY, server_key)
                logger.debug("Applied CURVE client keys and server public key")

        if bind:
            if random_port:
                port = socket.bind_to_random_port(norm_address)
                logger.debug(f"Bound to random port: {port}")
            else:
                port = socket.bind(norm_address)
                logger.debug(f"Bound to address: {norm_address}")
        else:
            socket.connect(norm_address)
            port = norm_address
            logger.debug(f"Connected to address: {norm_address}")

        return socket, port

    def __init__(
        self,
        in_address=None,
        out_address=None,
        *,
        zmq=None,
        in_curve=None,
        out_curve=None,
        in_bind=True,
        out_bind=True,
    ):
        if zmq is None:
            import zmq
        self.zmq = zmq
        self.closed = False

        try:
            context = zmq.Context(1)
            from zmq.auth.thread import ThreadAuthenticator

            frontend, in_port = self.configure_server_socket(
                context, zmq.SUB, in_address, in_curve, zmq, ThreadAuthenticator, bind=in_bind
            )
            frontend.setsockopt_string(zmq.SUBSCRIBE, "")

            backend, out_port = self.configure_server_socket(
                context, zmq.PUB, out_address, out_curve, zmq, ThreadAuthenticator, bind=out_bind
            )

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
            try:
                context.destroy()
            except NameError:
                ...
            raise
        else:
            self.in_port = (
                in_port.addr if hasattr(in_port, "addr") else _normalize_address(in_port) if in_bind else in_port
            )
            self.out_port = (
                out_port.addr
                if hasattr(out_port, "addr")
                else _normalize_address(out_port)
                if out_bind
                else out_port
            )
            self._frontend = frontend
            self._backend = backend
            self._context = context

    def start(self):
        if self.closed:
            raise RuntimeError(
                f"This Proxy has already been started and interrupted. Create a fresh instance with {repr(self)}"
            )
        try:
            self.zmq.device(self.zmq.FORWARDER, self._frontend, self._backend)
        finally:
            self.closed = True
            self._frontend.close()
            self._backend.close()
            self._context.destroy()

    def __repr__(self):
        return "{}(in_port={in_port}, out_port={out_port})".format(type(self).__name__, **vars(self))


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

    def __init__(
        self,
        address,
        *,
        prefix=b"",
        loop=None,
        zmq=None,
        zmq_asyncio=None,
        deserializer=pickle.loads,
        strict=False,
        curve_config=None,
    ):
        if isinstance(prefix, str):
            raise ValueError("prefix must be bytes, not string")
        if b" " in prefix:
            raise ValueError(f"prefix {prefix!r} may not contain b' '")
        self._prefix = prefix
        if zmq is None:
            import zmq
            import zmq.auth
        if zmq_asyncio is None:
            import zmq.asyncio as zmq_asyncio

        self._deserializer = deserializer
        self.address = _normalize_address(address)

        if loop is None:
            loop = asyncio.new_event_loop()
        self.loop = loop
        self._context = None
        self._socket = None

        def __finish_setup():
            asyncio.set_event_loop(self.loop)

            self._context = zmq_asyncio.Context()
            self._socket = sock = self._context.socket(zmq.SUB)

            if curve_config is not None:
                # Load the client cert pair
                client_public, client_secret = zmq.auth.load_certificate(curve_config.secret_path)
                sock.setsockopt(zmq.CURVE_PUBLICKEY, client_public)
                if client_secret is None:
                    raise ValueError("The client secret key could not be found.")
                sock.setsockopt(zmq.CURVE_SECRETKEY, client_secret)

                # Load the server public key and register with the socket
                server_key, _ = zmq.auth.load_certificate(curve_config.server_public_key)
                sock.setsockopt(zmq.CURVE_SERVERKEY, server_key)

            self._socket.connect(self.address)
            self._socket.setsockopt_string(zmq.SUBSCRIBE, "")

        self.__factory = __finish_setup
        self._task = None
        self.closed = False
        self._strict = strict
        super().__init__()

    async def _poll(self):
        our_prefix = self._prefix  # local var to save an attribute lookup
        while True:
            message = await self._socket.recv()
            try:
                prefix, name, doc = message.split(b" ", 2)
            except ValueError as e:
                if self._strict:
                    raise Bluesky0MQDecodeError from e
                else:
                    print(
                        f"The message {message} could not be split into "
                        "three parts by b' '.  Dropping message on floor "
                        "and continuing"
                        f"\n\n{e}"
                    )
                    continue

            try:
                name = name.decode()
            except UnicodeDecodeError as e:
                if self._strict:
                    raise Bluesky0MQDecodeError from e
                else:
                    print(
                        f"The name {name} can not be decoded as utf-8. "
                        "Dropping message on the floor and continuing. "
                        f"\n\n{e}"
                    )
                    continue
            if (not our_prefix) or prefix == our_prefix:
                try:
                    doc = self._deserializer(doc)
                except Exception as e:
                    if self._strict:
                        raise Bluesky0MQDecodeError from e
                    else:
                        if len(doc) > 1024:
                            msg_doc = doc[:1024] + b"--SNIPPED--"
                        else:
                            msg_doc = doc
                        print(
                            f"Failed to deserialize the {name} document "
                            f"{msg_doc} using {self._deserializer}. "
                            "Dropping on floor and continuing"
                            f"\n\n{e}"
                        )
                        continue
                self.loop.call_soon(self.process, DocumentNames[name], doc)

    def start(self):
        if self.closed:
            raise RuntimeError(
                "This RemoteDispatcher has already been "
                "started and interrupted. Create a fresh "
                f"instance with {self!r}"
            )
        try:
            self.__factory()
            self._task = self.loop.create_task(self._poll())
            self.loop.run_until_complete(self._task)
            task_exception = self._task.exception()
            if task_exception is not None:
                raise task_exception
        finally:
            self.stop()

    def stop(self):
        if self._task is not None:
            self._task.cancel()
        if self._socket is not None:
            self._socket.close()
        if self._context is not None:
            self._context.destroy()
        self.loop.close()
        self.closed = True
