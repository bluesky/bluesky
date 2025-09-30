import gc
import os
import signal
import threading
import time
from subprocess import run

import multiprocess
import numpy as np
import pytest
from event_model import sanitize_doc

from bluesky import Msg
from bluesky.callbacks.zmq import Proxy, Publisher, RemoteDispatcher
from bluesky.plans import count
from bluesky.tests import uses_os_kill_sigint


def test_proxy_script():
    p = run(["bluesky-0MQ-proxy", "-h"])
    assert p.returncode == 0


def _start_proxy_proc():
    def start_proxy(start_event):
        start_event.set()
        Proxy(5567, 5568).start()

    proxy_start_event = multiprocess.Event()
    proxy_proc = multiprocess.Process(target=start_proxy, args=(proxy_start_event,), daemon=True)
    proxy_proc.start()
    proxy_start_event.wait(timeout=5)
    assert proxy_start_event.is_set()
    # The event only monitors that the process is started.
    # Extra delay is needed to ensure that the proxy is started.
    time.sleep(0.2)

    return proxy_proc


def _start_dispatcher_proc(prefix=None):
    def make_and_start_dispatcher(queue, start_event):
        def put_in_queue(name, doc):
            print("putting ", name, "in queue")
            start_event.set()
            queue.put((name, doc))

        kwargs = {"prefix": prefix} if prefix else {}
        d = RemoteDispatcher("127.0.0.1:5568", **kwargs)
        d.subscribe(put_in_queue)
        print("REMOTE IS READY TO START")
        d.loop.call_later(9, d.stop)
        start_event.set()
        d.start()

    dispatcher_start_event = multiprocess.Event()
    queue = multiprocess.Queue()
    dispatcher_proc = multiprocess.Process(
        target=make_and_start_dispatcher, daemon=True, args=(queue, dispatcher_start_event)
    )
    dispatcher_proc.start()
    dispatcher_start_event.wait(timeout=5)
    assert dispatcher_start_event.is_set()
    # The event is set before dispatcher is started. It indicates that the process
    # is running. Extra delay is needed to ensure that the dispatcher is started.
    time.sleep(0.2)

    return dispatcher_proc, queue


def test_zmq_basic(RE, hw):
    # COMPONENT 1
    # Run a 0MQ proxy on a separate process.
    proxy_proc = _start_proxy_proc()

    # COMPONENT 2
    # Run a Publisher and a RunEngine in this main process.
    p = Publisher("127.0.0.1:5567")  # noqa
    RE.subscribe(p)

    # COMPONENT 3
    # Run a RemoteDispatcher on another separate process. Pass the documents
    # it receives over a Queue to this process, so we can count them for our
    # test.
    dispatcher_proc, queue = _start_dispatcher_proc()

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


@pytest.mark.parametrize("host", ["localhost:5555", ("localhost", 5555)])
def test_zmq_RD_ports_spec(host):
    # test that two ways of specifying address are equivalent
    d = RemoteDispatcher(host)
    assert d.address == ("localhost", 5555)
    assert d._socket is None
    assert d._context is None
    assert not d.closed
    d.stop()
    assert d._socket is None
    assert d._context is None
    assert d.closed
    del d


def test_zmq_no_RE_basic(RE):
    # COMPONENT 1
    # Run a 0MQ proxy on a separate process.
    proxy_proc = _start_proxy_proc()

    # COMPONENT 2
    # Run a Publisher and a RunEngine in this main process.
    p = Publisher("127.0.0.1:5567")  # noqa

    # COMPONENT 3
    # Run a RemoteDispatcher on another separate process. Pass the documents
    # it receives over a Queue to this process, so we can count them for our
    # test.
    dispatcher_proc, queue = _start_dispatcher_proc()

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
    ra = sanitize_doc(remote_accumulator)
    la = sanitize_doc(local_accumulator)
    assert ra == la


def test_zmq_no_RE_newserializer(RE):
    cloudpickle = pytest.importorskip("cloudpickle")

    # COMPONENT 1
    # Run a 0MQ proxy on a separate process.
    proxy_proc = _start_proxy_proc()

    # COMPONENT 2
    # Run a Publisher and a RunEngine in this main process.
    p = Publisher("127.0.0.1:5567", serializer=cloudpickle.dumps)  # noqa

    # COMPONENT 3
    # Run a RemoteDispatcher on another separate process. Pass the documents
    # it receives over a Queue to this process, so we can count them for our
    # test.
    dispatcher_proc, queue = _start_dispatcher_proc()

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
    ra = sanitize_doc(remote_accumulator)
    la = sanitize_doc(local_accumulator)
    assert ra == la


def test_zmq_prefix(RE, hw):
    # COMPONENT 1
    # Run a 0MQ proxy on a separate process.
    proxy_proc = _start_proxy_proc()

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
    dispatcher_proc, queue = _start_dispatcher_proc(prefix=b"sb")

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
    ra = sanitize_doc(remote_accumulator)
    la = sanitize_doc(local_accumulator)
    assert ra == la
