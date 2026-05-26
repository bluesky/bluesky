import asyncio
import itertools
import logging
import os
import signal
import sys
import threading
import time
from subprocess import run
from unittest.mock import MagicMock, patch

import multiprocess
import pytest
import zmq
from pytest_mock import MockerFixture

from bluesky.callbacks.zmq import ClientCurve, Proxy, Publisher, RemoteDispatcher, ServerCurve, _normalize_address
from bluesky.plans import count
from bluesky.run_engine import RunEngine
from bluesky.tests import uses_os_kill_sigint

from .conftest import ReadableSignal

# ZMQ subscription propagation is slower on Windows CI runners
_ZMQ_CONNECTION_TIMEOUT = 5.0 if sys.platform == "win32" else 0.5


@pytest.fixture
def mock_zmq_context(mocker: MockerFixture):
    """Fixture that mocks the ZMQ context and socket classes"""

    class MockSocket:
        # Keep track of any socket options, so we can verify them in tests.
        opts = {}

        def close(self):
            pass

        def bind(self, address):
            class MockConnectionResult:
                def __init__(self, address):
                    self.addr = address

            return MockConnectionResult(address)

        def bind_to_random_port(self, address):
            return 12345

        def connect(self, address):
            class MockConnectionResult:
                def __init__(self, address):
                    self.addr = address

            return MockConnectionResult(address)

        def setsockopt_string(self, opt, value):
            self.opts[opt] = value

        def setsockopt(self, opt, value):
            self.opts[opt] = value

        def send(self, *args, **kwargs):
            pass

    class MockContext:
        def destroy(self):
            pass

        def socket(self, *args, **kwargs):
            return MockSocket()

    return mocker.patch("zmq.Context", return_value=MockContext())


def test_proxy_script():
    p = run(["bluesky-0MQ-proxy", "-h"])
    assert p.returncode == 0


@pytest.fixture
def proxy():
    def start_proxy(ready_event):
        p = Proxy(5567, 5568, in_bind=True, out_bind=True)
        ready_event.set()  # Ports are bound after __init__ returns
        p.start()  # Blocks on zmq.device()

    ready_event = multiprocess.Event()
    proc = multiprocess.Process(target=start_proxy, args=(ready_event,), daemon=True)
    proc.start()
    ready_event.wait(timeout=5)
    assert ready_event.is_set()
    yield
    proc.terminate()
    proc.join(timeout=5)


@pytest.fixture
def publisher():
    p = Publisher("127.0.0.1:5567")
    # TODO: Replace sleep with handshake event wait
    time.sleep(_ZMQ_CONNECTION_TIMEOUT)
    return p


@pytest.fixture
def dispatcher():
    """RemoteDispatcher fixture for tracking messages"""

    docs_received = []
    stop_event = threading.Event()

    def store_document(name, doc):
        docs_received.append((name, doc))

    def stop_doc_watcher(name, doc):
        if name == "stop":
            stop_event.set()

    # On Windows, zmq.asyncio requires SelectorEventLoop (not ProactorEventLoop)
    loop = asyncio.SelectorEventLoop() if sys.platform == "win32" else None
    d = RemoteDispatcher("127.0.0.1:5568", loop=loop)
    d.subscribe(store_document)
    d.subscribe(stop_doc_watcher)
    threading.Thread(target=d.start, daemon=True).start()
    # TODO: Replace sleep with handshake event wait
    time.sleep(_ZMQ_CONNECTION_TIMEOUT)
    return stop_event, docs_received


def test_zmq_round_trip(proxy, publisher, dispatcher):
    """
    Generate two documents. The Publisher will send them to the proxy
    device over 5567, and the proxy will send them to the
    RemoteDispatcher over 5568. The RemoteDispatcher will push them into
    the queue, where we can verify that they round-tripped.
    """
    RE = RunEngine({})
    RE.subscribe(publisher)

    remote_stop_event, remote_docs = dispatcher
    local_docs = []
    det = ReadableSignal("det")

    def local_cb(name, doc):
        local_docs.append((name, doc))

    RE(count([det]), local_cb)
    remote_stop_event.wait(timeout=5)
    assert remote_stop_event.is_set()

    assert remote_docs == local_docs


@uses_os_kill_sigint
def test_zmq_proxy_blocks_sigint_exits():
    zmq_device_event = threading.Event()

    def delayed_sigint():
        zmq_device_event.wait(timeout=5)
        assert zmq_device_event.is_set()
        os.kill(os.getpid(), signal.SIGINT)

    def device_mock(*args, **kwargs):
        zmq_device_event.set()
        # Block until interrupted by SIGINT, just like the real zmq.device
        threading.Event().wait()

    proxy = Proxy(5567, 5568)
    assert not proxy.closed
    threading.Thread(target=delayed_sigint, daemon=True).start()
    try:
        with patch("bluesky.callbacks.zmq.zmq.device", side_effect=device_mock):
            proxy.start()
    except KeyboardInterrupt:
        ...
    assert proxy.closed
    with pytest.raises(RuntimeError):
        proxy.start()


@patch("zmq.Context")
def test_publisher_custom_serializer(mock_ctx):
    """Verify Publisher uses the custom serializer"""
    mock_socket = mock_ctx.return_value.socket.return_value

    custom_serializer = MagicMock(return_value=b"custom_bytes")
    p = Publisher("127.0.0.1:5567", serializer=custom_serializer)
    p("start", {"uid": "abc123"})

    custom_serializer.assert_called_once_with({"uid": "abc123"})
    mock_socket.send.assert_called_once_with(b" start custom_bytes")
    p.close()


def test_dispatcher_custom_deserializer():
    """Verify RemoteDispatcher uses the custom deserializer."""
    custom_deserializer = MagicMock(return_value={"uid": "abc123"})
    docs_received = []
    received_event = threading.Event()

    d = RemoteDispatcher("127.0.0.1:5568", deserializer=custom_deserializer)

    def cb(name, doc):
        docs_received.append((name, doc))
        received_event.set()

    d.subscribe(cb)

    async def fake_recv():
        await asyncio.sleep(0)  # Yield to let pending callbacks fire
        if not received_event.is_set():
            return b" start custom_bytes"
        await asyncio.Event().wait()

    with patch("bluesky.callbacks.zmq.zmq_asyncio.Context") as mock_ctx:
        mock_ctx.return_value.socket.return_value.recv = fake_recv

        t = threading.Thread(target=d.start, daemon=True)
        t.start()

        received_event.wait(timeout=5)
        assert received_event.is_set()

        d.stop()
        t.join(timeout=5)

    custom_deserializer.assert_called_once_with(b"custom_bytes")
    assert docs_received == [("start", {"uid": "abc123"})]


def test_zmq_prefix(proxy):
    """
    Two publishers send with different prefixes. The dispatcher subscribes
    to only one prefix and should only receive documents from that publisher.
    """
    RE = RunEngine({})

    # Two publishers with different prefixes
    pub_match = Publisher("127.0.0.1:5567", prefix=b"sb")
    pub_other = Publisher("127.0.0.1:5567", prefix=b"not_sb")
    RE.subscribe(pub_match)
    RE.subscribe(pub_other)

    # Dispatcher only subscribes to prefix b"sb"
    docs_received = []
    stop_event = threading.Event()

    def store_document(name, doc):
        docs_received.append((name, doc))

    def stop_doc_watcher(name, doc):
        if name == "stop":
            stop_event.set()

    d = RemoteDispatcher(
        "127.0.0.1:5568",
        prefix=b"sb",
        # On Windows, zmq.asyncio requires SelectorEventLoop (not ProactorEventLoop)
        loop=asyncio.SelectorEventLoop() if sys.platform == "win32" else None,
    )
    d.subscribe(store_document)
    d.subscribe(stop_doc_watcher)
    threading.Thread(target=d.start, daemon=True).start()
    time.sleep(_ZMQ_CONNECTION_TIMEOUT)  # TODO: Replace sleep with handshake event wait

    local_docs = []

    def local_cb(name, doc):
        local_docs.append((name, doc))

    det = ReadableSignal("det")
    RE(count([det]), local_cb)
    stop_event.wait(timeout=5)
    assert stop_event.is_set()

    # Only docs from the matching prefix publisher should arrive
    assert docs_received == local_docs

    pub_match.close()
    pub_other.close()


@pytest.mark.parametrize(
    ("cls", "prefix", "err_msg"),
    [
        (c, *t)
        for c, t in itertools.product(
            [Publisher, RemoteDispatcher],
            [("not_bytes", "prefix must be bytes, not string"), (b"has space", "may not contain b' '")],
        )
    ],
)
def test_zmq_invalid_prefix(cls, prefix, err_msg):
    with pytest.raises(ValueError, match=err_msg):
        cls("127.0.0.1:5567", prefix=prefix)


@pytest.mark.parametrize(
    "host",
    ["localhost:5555", ("localhost", 5555)],
)
def test_zmq_RD_ports_spec(host: str | tuple[str, int]):
    # test that two ways of specifying address are equivalent
    d = RemoteDispatcher(host)
    assert d.address == "tcp://localhost:5555"
    assert d._socket is None
    assert d._context is None
    assert not d.closed
    del d


@pytest.mark.parametrize(
    "address",
    [
        ("localhost", "tcp://localhost"),
        ("localhost:9", "tcp://localhost:9"),
        ("remote.host", "tcp://remote.host"),
        ("remote.host:9", "tcp://remote.host:9"),
        ("tcp://remote.host", "tcp://remote.host"),
        ("tcp://localhost", "tcp://localhost"),
        ("tcp://localhost:9", "tcp://localhost:9"),
        ("tcp://remote.host:9", "tcp://remote.host:9"),
        ("ipc:///tmp/path", "ipc:///tmp/path"),
        (("localhost",), "tcp://localhost"),
        (("localhost", 9), "tcp://localhost:9"),
        (("ipc", "/tmp/path"), "ipc:///tmp/path"),
        (("tcp", "localhost"), "tcp://localhost"),
        (("tcp", "localhost", 9), "tcp://localhost:9"),
        (("tcp", "localhost", "9"), "tcp://localhost:9"),
        ((None), "tcp://*"),
    ],
)
def test_address_normaliaztion(address: tuple[tuple[str | None, ...], str]):
    inp, outp = address
    assert _normalize_address(inp) == outp


def test_normlize_address_invalid_input():
    with pytest.raises(TypeError, match="Input expected to be int, str, or tuple, not float"):
        _normalize_address(123.0)


@pytest.mark.parametrize("in_or_out", ["in", "out"])
def test_specify_port_and_address_raises(in_or_out):
    err_msg = f"Cannot specify both '{in_or_out}_port' and '{in_or_out}_address'"
    with pytest.raises(ValueError, match=err_msg):
        match in_or_out:
            case "in":
                Proxy(in_address="tcp://localhost:5555", in_port="tcp://localhost:5555")
            case "out":
                Proxy(out_address="tcp://localhost:5555", out_port="tcp://localhost:5555")


@pytest.mark.parametrize(
    ("context_err", "frontend_err", "backend_err"),
    [
        (True, False, False),
        (False, True, False),
        (False, False, True),
    ],
)
def test_proxy_fails_to_configure_sockets(
    mocker: MockerFixture, context_err: bool, frontend_err: bool, backend_err: bool
):
    """Checks to make sure we are properly cleaning up resources on failure"""

    nclosed: int = 0
    ndestroyed: int = 0

    class MockPortOrContext:
        def close(self):
            nonlocal nclosed
            nclosed += 1

        def destroy(self):
            nonlocal ndestroyed
            ndestroyed += 1

        def setsockopt_string(self, *args, **kwargs):
            pass

    mock_context = mocker.patch("zmq.Context")

    mock_context.side_effect = [RuntimeError("Context error") if context_err else MockPortOrContext()]

    mock_configure_server_socket = mocker.patch("bluesky.callbacks.zmq.Proxy.configure_server_socket")
    mock_configure_server_socket.side_effect = [
        RuntimeError("Frontend socket error") if frontend_err else (MockPortOrContext(), ""),
        RuntimeError("Backend socket error") if backend_err else (MockPortOrContext(), ""),
    ]

    with pytest.raises(RuntimeError, match="Context error|Frontend socket error|Backend socket error"):
        Proxy()

    assert ndestroyed == (1 if not context_err else 0)
    if context_err or frontend_err:
        expected_nclosed = 0
    elif backend_err:
        expected_nclosed = 1
    else:
        expected_nclosed = 2
    assert nclosed == expected_nclosed


@pytest.mark.parametrize("in_or_out", ["in", "out"])
@pytest.mark.filterwarnings("always::DeprecationWarning")
def test_dep_warning_if_using_port_args(mock_zmq_context, in_or_out: str):
    with pytest.deprecated_call(
        match=f"The '{in_or_out}_port' parameter is deprecated and will be removed in a future release"
    ):
        match in_or_out:
            case "in":
                Proxy(in_port=5555)
            case "out":
                Proxy(out_port=5555)


@pytest.mark.parametrize("cls", [ServerCurve, ClientCurve])
def test_cannot_bind_with_incorrect_curve(mock_zmq_context, tmp_path, cls):

    args = [tmp_path, tmp_path]
    if cls == ServerCurve:
        args.append(None)  # ServerCurve takes an extra argument

    other_cls = ClientCurve if cls == ServerCurve else ServerCurve

    with pytest.raises(TypeError, match=f"When bind={cls == ClientCurve}, curve must be a {other_cls.__name__}"):
        ctx = zmq.Context()
        Proxy.configure_server_socket(ctx, zmq.PUB, None, curve=cls(*args), bind=cls == ClientCurve)


@pytest.mark.parametrize("cls", [ServerCurve, ClientCurve])
def test_secret_not_found_raises_value_error(mock_zmq_context, mocker, tmp_path, cls):
    mocker.patch("bluesky.callbacks.zmq.ThreadAuthenticator")
    mocker.patch("zmq.auth.load_certificate", return_value=(None, None))
    curve_args = [tmp_path, tmp_path]
    if cls == ServerCurve:
        curve_args.append(None)  # ServerCurve takes an extra argument
    with pytest.raises(ValueError, match=f"The {cls.__name__.lower()[:-5]} secret key could not be found"):
        ctx = zmq.Context()
        Proxy.configure_server_socket(ctx, zmq.PUB, None, curve=cls(*curve_args), bind=cls == ServerCurve)


def test_configure_server_socket_client_curve(mock_zmq_context, mocker, tmp_path, caplog):
    ctx = zmq.Context()
    mock_load_certs = mocker.patch("zmq.auth.load_certificate")
    mock_load_certs.side_effect = [("client_public", "client_secret"), ("server_public", None)]

    secret_path = tmp_path / "secret.key"
    server_public_key = tmp_path / "server_public" / "pub.key"
    curve = ClientCurve(secret_path, server_public_key)
    with caplog.at_level(logging.DEBUG, logger="bluesky.callbacks.zmq"):
        socket, address = Proxy.configure_server_socket(ctx, zmq.PUB, None, curve=curve, bind=False)
    assert isinstance(socket, type(zmq.Context().socket(zmq.PUB)))
    assert isinstance(address, str)

    assert socket.opts.get(zmq.CURVE_SERVERKEY) == "server_public"
    assert socket.opts.get(zmq.CURVE_PUBLICKEY) == "client_public"
    assert socket.opts.get(zmq.CURVE_SECRETKEY) == "client_secret"
    assert f"Creating socket of type {zmq.PUB} for address {address}, bind=False" in caplog.text
    assert f"Configuring CURVE client security with secret_path={str(secret_path)}" in caplog.text
    assert "Applied CURVE client keys and server public key" in caplog.text
    assert f"Connected to address: {address}" in caplog.text


@pytest.mark.parametrize(
    ("allow", "client_pubkeys", "random_port"),
    [
        (None, None, False),
        (["ip1", "ip2"], ["client_pub1", "client_pub2"], True),
    ],
)
def test_configure_server_socket_server_curve(
    mock_zmq_context, mocker, tmp_path, caplog, allow, client_pubkeys, random_port
):
    ctx = zmq.Context()
    mocker.patch("zmq.auth.load_certificate", return_value=("server_public", "server_secret"))

    mock_auth = mocker.patch("bluesky.callbacks.zmq.ThreadAuthenticator")

    secret_path = tmp_path / "secret.key"
    curve = ServerCurve(secret_path, client_pubkeys, allow)

    base_address = "tcp://127.0.0.1" + ("" if random_port else ":54321")
    with caplog.at_level(logging.DEBUG, logger="bluesky.callbacks.zmq"):
        socket, address = Proxy.configure_server_socket(
            ctx,
            zmq.PUB,
            base_address,
            curve=curve,
            bind=True,
        )

    mock_auth.return_value.start.assert_called_once()
    if allow is not None:
        mock_auth.return_value.allow.assert_called_once_with(*allow)
    else:
        mock_auth.return_value.allow.assert_not_called()

    if client_pubkeys is not None:
        mock_auth.return_value.configure_curve.assert_called_once_with(domain="*", location=client_pubkeys)
    else:
        mock_auth.return_value.configure_curve.assert_called_once_with(
            domain="*", location=zmq.auth.CURVE_ALLOW_ANY
        )

    assert isinstance(socket, type(zmq.Context().socket(zmq.PUB)))
    # address is always a string (normalized address), never int/None
    assert isinstance(address, str)

    assert socket.opts.get(zmq.CURVE_PUBLICKEY) == "server_public"
    assert socket.opts.get(zmq.CURVE_SECRETKEY) == "server_secret"
    assert socket.opts.get(zmq.CURVE_SERVER)
    # The address in the log is always the normalized address, not the port number
    expected_addr = "tcp://127.0.0.1" + (":12345" if random_port else ":54321")
    assert f"Creating socket of type {zmq.PUB} for address {base_address}, bind=True" in caplog.text
    assert "Started ZMQ authenticator" in caplog.text
    if allow is not None:
        assert f"Configured IP address allowlist: {allow}" in caplog.text
    if client_pubkeys is not None:
        assert f"Configured CURVE client public keys from: {client_pubkeys}" in caplog.text
    else:
        assert "Configured CURVE to allow any client with valid public key" in caplog.text

    assert "Applied CURVE keys and enabled CURVE server mode" in caplog.text
    if random_port:
        assert "Bound to random port: 12345" in caplog.text
    else:
        assert f"Bound to address: {expected_addr}" in caplog.text


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_remote_dispatcher_stop_from_other_thread():
    """Regression test for #2012: stop() called from another thread must not raise RuntimeError.

    stop() is synchronous, so once it returns the dispatcher is fully torn down.
    """
    dispatcher = RemoteDispatcher("127.0.0.1:60611")  # nothing listening
    thread = threading.Thread(target=dispatcher.start, daemon=True)
    thread.start()
    # Allow the loop and poll task to start.
    time.sleep(0.5)
    try:
        dispatcher.stop()
        assert dispatcher.closed
        assert dispatcher._task is None
        assert dispatcher._socket is None
        assert dispatcher._context is None
    finally:
        thread.join(timeout=5)
    assert not thread.is_alive()
