import gc
import itertools
import logging
import os
import signal
import threading
import time
from subprocess import run

import multiprocess
import numpy as np
import pytest
import zmq
from event_model import sanitize_doc
from pytest_mock import MockerFixture

from bluesky import Msg
from bluesky.callbacks.zmq import ClientCurve, Proxy, Publisher, RemoteDispatcher, ServerCurve, _normalize_address
from bluesky.plans import count
from bluesky.run_engine import RunEngine
from bluesky.tests import uses_os_kill_sigint


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


def test_zmq(RE, hw):
    # COMPONENT 1
    # Run a 0MQ proxy on a separate process.
    def start_proxy():
        Proxy(5567, 5568).start()

    proxy_proc = multiprocess.Process(target=start_proxy, daemon=True)
    proxy_proc.start()
    time.sleep(5)  # Give this plenty of time to start up.

    # COMPONENT 2
    # Run a Publisher and a RunEngine in this main process.

    p = Publisher("127.0.0.1:5567")  # noqa
    RE.subscribe(p)

    # COMPONENT 3
    # Run a RemoteDispatcher on another separate process. Pass the documents
    # it receives over a Queue to this process, so we can count them for our
    # test.

    def make_and_start_dispatcher(queue):
        def put_in_queue(name, doc):
            print("putting ", name, "in queue")
            queue.put((name, doc))

        d = RemoteDispatcher("127.0.0.1:5568")
        d.subscribe(put_in_queue)
        print("REMOTE IS READY TO START")
        d.loop.call_later(9, d.stop)
        d.start()

    queue = multiprocess.Queue()
    dispatcher_proc = multiprocess.Process(target=make_and_start_dispatcher, daemon=True, args=(queue,))
    dispatcher_proc.start()
    time.sleep(5)  # As above, give this plenty of time to start.

    # Generate two documents. The Publisher will send them to the proxy
    # device over 5567, and the proxy will send them to the
    # RemoteDispatcher over 5568. The RemoteDispatcher will push them into
    # the queue, where we can verify that they round-tripped.

    local_accumulator = []

    def local_cb(name, doc):
        local_accumulator.append((name, doc))

    # Check that numpy stuff is sanitized by putting some in the start doc.
    md = {"stuff": {"nested": np.array([1, 2, 3])}, "scalar_stuff": np.float64(3), "array_stuff": np.ones((3, 3))}

    # RE([Msg('open_run', **md), Msg('close_run')], local_cb)
    RE(count([hw.det]), local_cb, **md)
    time.sleep(1)

    # Get the two documents from the queue (or timeout --- test will fail)
    remote_accumulator = []
    for i in range(len(local_accumulator)):  # noqa: B007
        remote_accumulator.append(queue.get(timeout=2))
    p.close()
    proxy_proc.terminate()
    dispatcher_proc.terminate()
    proxy_proc.join()
    dispatcher_proc.join()
    ra = sanitize_doc(remote_accumulator)
    la = sanitize_doc(local_accumulator)
    assert ra == la

    gc.collect()
    gc.collect()


@uses_os_kill_sigint
def test_zmq_proxy_blocks_sigint_exits():
    # The test `test_zmq` runs Proxy and RemoteDispatcher in a separate
    # process, which coverage misses.

    def delayed_sigint(delay):
        time.sleep(delay)
        os.kill(os.getpid(), signal.SIGINT)

    proxy = Proxy(5567, 5568)
    assert not proxy.closed
    threading.Thread(target=delayed_sigint, args=(1,)).start()
    try:
        proxy.start()
        # delayed_sigint stops the proxy
    except KeyboardInterrupt:
        ...
    assert proxy.closed
    with pytest.raises(RuntimeError):
        proxy.start()

    proxy = Proxy()  # random port
    threading.Thread(target=delayed_sigint, args=(1,)).start()
    try:
        proxy.start()
        # delayed_sigint stops the proxy
    except KeyboardInterrupt:
        ...
    assert proxy.closed
    repr(proxy)
    gc.collect()
    gc.collect()


def test_zmq_no_RE(RE: RunEngine):
    # COMPONENT 1
    # Run a 0MQ proxy on a separate process.
    def start_proxy():
        Proxy(5567, 5568).start()

    proxy_proc = multiprocess.Process(target=start_proxy, daemon=True)
    proxy_proc.start()
    time.sleep(5)  # Give this plenty of time to start up.

    # COMPONENT 2
    # Run a Publisher and a RunEngine in this main process.

    p = Publisher("127.0.0.1:5567")  # noqa

    # COMPONENT 3
    # Run a RemoteDispatcher on another separate process. Pass the documents
    # it receives over a Queue to this process, so we can count them for our
    # test.

    def make_and_start_dispatcher(queue):
        def put_in_queue(name, doc):
            print("putting ", name, "in queue")
            queue.put((name, doc))

        d = RemoteDispatcher("127.0.0.1:5568")
        d.subscribe(put_in_queue)
        print("REMOTE IS READY TO START")
        d.loop.call_later(9, d.stop)
        d.start()

    queue = multiprocess.Queue()
    dispatcher_proc = multiprocess.Process(target=make_and_start_dispatcher, daemon=True, args=(queue,))
    dispatcher_proc.start()
    time.sleep(5)  # As above, give this plenty of time to start.

    # Generate two documents. The Publisher will send them to the proxy
    # device over 5567, and the proxy will send them to the
    # RemoteDispatcher over 5568. The RemoteDispatcher will push them into
    # the queue, where we can verify that they round-tripped.

    local_accumulator = []

    def local_cb(name, doc):
        local_accumulator.append((name, doc))

    RE([Msg("open_run"), Msg("close_run")], local_cb)

    # This time the Publisher isn't attached to an RE. Send the documents
    # manually. (The idea is, these might have come from a Broker instead...)
    for name, doc in local_accumulator:
        p(name, doc)
    time.sleep(1)

    # Get the two documents from the queue (or timeout --- test will fail)
    remote_accumulator = []
    for i in range(2):  # noqa: B007
        remote_accumulator.append(queue.get(timeout=2))
    p.close()
    proxy_proc.terminate()
    dispatcher_proc.terminate()
    proxy_proc.join()
    dispatcher_proc.join()
    ra = [sanitize_doc(doc) for doc in remote_accumulator]
    la = [sanitize_doc(doc) for doc in local_accumulator]
    assert ra == la


def test_zmq_no_RE_newserializer(RE: RunEngine):
    cloudpickle = pytest.importorskip("cloudpickle")

    # COMPONENT 1
    # Run a 0MQ proxy on a separate process.
    def start_proxy():
        Proxy(5567, 5568).start()

    proxy_proc = multiprocess.Process(target=start_proxy, daemon=True)
    proxy_proc.start()
    time.sleep(5)  # Give this plenty of time to start up.

    # COMPONENT 2
    # Run a Publisher and a RunEngine in this main process.
    p = Publisher("127.0.0.1:5567", serializer=cloudpickle.dumps)  # noqa

    # COMPONENT 3
    # Run a RemoteDispatcher on another separate process. Pass the documents
    # it receives over a Queue to this process, so we can count them for our
    # test.
    def make_and_start_dispatcher(queue):
        def put_in_queue(name, doc):
            print("putting ", name, "in queue")
            queue.put((name, doc))

        d = RemoteDispatcher("127.0.0.1:5568", deserializer=cloudpickle.loads)
        d.subscribe(put_in_queue)
        print("REMOTE IS READY TO START")
        d.loop.call_later(9, d.stop)
        d.start()

    queue = multiprocess.Queue()
    dispatcher_proc = multiprocess.Process(target=make_and_start_dispatcher, daemon=True, args=(queue,))
    dispatcher_proc.start()
    time.sleep(5)  # As above, give this plenty of time to start.

    # Generate two documents. The Publisher will send them to the proxy
    # device over 5567, and the proxy will send them to the
    # RemoteDispatcher over 5568. The RemoteDispatcher will push them into
    # the queue, where we can verify that they round-tripped.

    local_accumulator = []

    def local_cb(name, doc):
        local_accumulator.append((name, doc))

    RE([Msg("open_run"), Msg("close_run")], local_cb)

    # This time the Publisher isn't attached to an RE. Send the documents
    # manually. (The idea is, these might have come from a Broker instead...)
    for name, doc in local_accumulator:
        p(name, doc)
    time.sleep(1)

    # Get the two documents from the queue (or timeout --- test will fail)
    remote_accumulator = []
    for i in range(2):  # noqa: B007
        remote_accumulator.append(queue.get(timeout=2))
    p.close()
    proxy_proc.terminate()
    dispatcher_proc.terminate()
    proxy_proc.join()
    dispatcher_proc.join()
    ra = [sanitize_doc(doc) for doc in remote_accumulator]
    la = [sanitize_doc(doc) for doc in local_accumulator]
    assert ra == la


def test_zmq_prefix(RE: RunEngine, hw):
    # COMPONENT 1
    # Run a 0MQ proxy on a separate process.
    def start_proxy():
        Proxy(5567, 5568).start()

    proxy_proc = multiprocess.Process(target=start_proxy, daemon=True)
    proxy_proc.start()
    time.sleep(5)  # Give this plenty of time to start up.

    # COMPONENT 2
    # Run a Publisher and a RunEngine in this main process.
    p = Publisher("127.0.0.1:5567", prefix=b"sb")  # noqa
    p2 = Publisher("127.0.0.1:5567", prefix=b"not_sb")  # noqa
    RE.subscribe(p)
    RE.subscribe(p2)

    # COMPONENT 3
    # Run a RemoteDispatcher on another separate process. Pass the documents
    # it receives over a Queue to this process, so we can count them for our
    # test.

    def make_and_start_dispatcher(queue):
        def put_in_queue(name, doc):
            print("putting ", name, "in queue")
            queue.put((name, doc))

        d = RemoteDispatcher("127.0.0.1:5568", prefix=b"sb")
        d.subscribe(put_in_queue)
        print("REMOTE IS READY TO START")
        d.loop.call_later(9, d.stop)
        d.start()

    queue = multiprocess.Queue()
    dispatcher_proc = multiprocess.Process(target=make_and_start_dispatcher, daemon=True, args=(queue,))
    dispatcher_proc.start()
    time.sleep(5)  # As above, give this plenty of time to start.

    # Generate two documents. The Publisher will send them to the proxy
    # device over 5567, and the proxy will send them to the
    # RemoteDispatcher over 5568. The RemoteDispatcher will push them into
    # the queue, where we can verify that they round-tripped.

    local_accumulator = []

    def local_cb(name, doc):
        local_accumulator.append((name, doc))

    # Check that numpy stuff is sanitized by putting some in the start doc.
    md = {"stuff": {"nested": np.array([1, 2, 3])}, "scalar_stuff": np.float64(3), "array_stuff": np.ones((3, 3))}

    # RE([Msg('open_run', **md), Msg('close_run')], local_cb)
    RE(count([hw.det]), local_cb, **md)
    time.sleep(1)

    # Get the two documents from the queue (or timeout --- test will fail)
    remote_accumulator = []
    for i in range(len(local_accumulator)):  # noqa: B007
        remote_accumulator.append(queue.get(timeout=2))
    p.close()
    p2.close()
    proxy_proc.terminate()
    dispatcher_proc.terminate()
    proxy_proc.join()
    dispatcher_proc.join()
    ra = [sanitize_doc(doc) for doc in remote_accumulator]
    la = [sanitize_doc(doc) for doc in local_accumulator]
    assert ra == la


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
    d.stop()
    assert d._socket is None
    assert d._context is None
    assert d.closed
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
