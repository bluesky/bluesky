import multiprocessing
import os
import signal
from subprocess import run
import threading
import time
import numpy as np
import pytest

from bluesky import Msg
from bluesky.callbacks.zmq import Proxy, Publisher, RemoteDispatcher
from bluesky.plans import count
from event_model import sanitize_doc


def test_proxy_script():
    p = run(['bluesky-0MQ-proxy', '-h'])
    assert p.returncode == 0


def test_zmq(RE, hw):
    # COMPONENT 1
    # Run a 0MQ proxy on a separate process.
    def start_proxy():
        Proxy(5567, 5568).start()

    proxy_proc = multiprocessing.Process(target=start_proxy, daemon=True)
    proxy_proc.start()
    time.sleep(5)  # Give this plenty of time to start up.

    # COMPONENT 2
    # Run a Publisher and a RunEngine in this main process.

    p = Publisher('127.0.0.1:5567')  # noqa
    RE.subscribe(p)

    # COMPONENT 3
    # Run a RemoteDispatcher on another separate process. Pass the documents
    # it receives over a Queue to this process, so we can count them for our
    # test.

    def make_and_start_dispatcher(queue):
        def put_in_queue(name, doc):
            print('putting ', name, 'in queue')
            queue.put((name, doc))

        d = RemoteDispatcher('127.0.0.1:5568')
        d.subscribe(put_in_queue)
        print("REMOTE IS READY TO START")
        d.loop.call_later(9, d.stop)
        d.start()

    queue = multiprocessing.Queue()
    dispatcher_proc = multiprocessing.Process(target=make_and_start_dispatcher,
                                              daemon=True, args=(queue,))
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
    md = {'stuff': {'nested': np.array([1, 2, 3])},
          'scalar_stuff': np.float64(3),
          'array_stuff': np.ones((3, 3))}

    # RE([Msg('open_run', **md), Msg('close_run')], local_cb)
    RE(count([hw.det]), local_cb, **md)
    time.sleep(1)

    # Get the two documents from the queue (or timeout --- test will fail)
    remote_accumulator = []
    for i in range(len(local_accumulator)):
        remote_accumulator.append(queue.get(timeout=2))
    p.close()
    proxy_proc.terminate()
    dispatcher_proc.terminate()
    proxy_proc.join()
    dispatcher_proc.join()
    ra = sanitize_doc(remote_accumulator)
    la = sanitize_doc(local_accumulator)
    assert ra == la


def test_zmq_components():
    # The test `test_zmq` runs Proxy and RemoteDispatcher in a separate
    # process, which coverage misses.

    def delayed_sigint(delay):
        time.sleep(delay)
        os.kill(os.getpid(), signal.SIGINT)

    proxy = Proxy(5567, 5568)
    assert not proxy.closed
    threading.Thread(target=delayed_sigint, args=(5,)).start()
    try:
        proxy.start()
        # delayed_sigint stops the proxy
    except KeyboardInterrupt:
        ...
    assert proxy.closed
    with pytest.raises(RuntimeError):
        proxy.start()

    proxy = Proxy()  # random port
    threading.Thread(target=delayed_sigint, args=(5,)).start()
    try:
        proxy.start()
        # delayed_sigint stops the proxy
    except KeyboardInterrupt:
        ...

    repr(proxy)

    # test that two ways of specifying address are equivalent
    d = RemoteDispatcher('localhost:5555')
    assert d.address == ('localhost', 5555)
    d = RemoteDispatcher(('localhost', 5555))
    assert d.address == ('localhost', 5555)

    repr(d)


def test_zmq_no_RE(RE):
    # COMPONENT 1
    # Run a 0MQ proxy on a separate process.
    def start_proxy():
        Proxy(5567, 5568).start()

    proxy_proc = multiprocessing.Process(target=start_proxy, daemon=True)
    proxy_proc.start()
    time.sleep(5)  # Give this plenty of time to start up.

    # COMPONENT 2
    # Run a Publisher and a RunEngine in this main process.

    p = Publisher('127.0.0.1:5567')  # noqa

    # COMPONENT 3
    # Run a RemoteDispatcher on another separate process. Pass the documents
    # it receives over a Queue to this process, so we can count them for our
    # test.

    def make_and_start_dispatcher(queue):
        def put_in_queue(name, doc):
            print('putting ', name, 'in queue')
            queue.put((name, doc))

        d = RemoteDispatcher('127.0.0.1:5568')
        d.subscribe(put_in_queue)
        print("REMOTE IS READY TO START")
        d.loop.call_later(9, d.stop)
        d.start()

    queue = multiprocessing.Queue()
    dispatcher_proc = multiprocessing.Process(target=make_and_start_dispatcher,
                                              daemon=True, args=(queue,))
    dispatcher_proc.start()
    time.sleep(5)  # As above, give this plenty of time to start.

    # Generate two documents. The Publisher will send them to the proxy
    # device over 5567, and the proxy will send them to the
    # RemoteDispatcher over 5568. The RemoteDispatcher will push them into
    # the queue, where we can verify that they round-tripped.

    local_accumulator = []

    def local_cb(name, doc):
        local_accumulator.append((name, doc))

    RE([Msg('open_run'), Msg('close_run')], local_cb)

    # This time the Publisher isn't attached to an RE. Send the documents
    # manually. (The idea is, these might have come from a Broker instead...)
    for name, doc in local_accumulator:
        p(name, doc)
    time.sleep(1)

    # Get the two documents from the queue (or timeout --- test will fail)
    remote_accumulator = []
    for i in range(2):
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
    cloudpickle = pytest.importorskip('cloudpickle')

    # COMPONENT 1
    # Run a 0MQ proxy on a separate process.
    def start_proxy():
        Proxy(5567, 5568).start()

    proxy_proc = multiprocessing.Process(target=start_proxy, daemon=True)
    proxy_proc.start()
    time.sleep(5)  # Give this plenty of time to start up.

    # COMPONENT 2
    # Run a Publisher and a RunEngine in this main process.
    p = Publisher('127.0.0.1:5567', serializer=cloudpickle.dumps)  # noqa

    # COMPONENT 3
    # Run a RemoteDispatcher on another separate process. Pass the documents
    # it receives over a Queue to this process, so we can count them for our
    # test.
    def make_and_start_dispatcher(queue):
        def put_in_queue(name, doc):
            print('putting ', name, 'in queue')
            queue.put((name, doc))

        d = RemoteDispatcher('127.0.0.1:5568', deserializer=cloudpickle.loads)
        d.subscribe(put_in_queue)
        print("REMOTE IS READY TO START")
        d.loop.call_later(9, d.stop)
        d.start()

    queue = multiprocessing.Queue()
    dispatcher_proc = multiprocessing.Process(target=make_and_start_dispatcher,
                                              daemon=True, args=(queue,))
    dispatcher_proc.start()
    time.sleep(5)  # As above, give this plenty of time to start.

    # Generate two documents. The Publisher will send them to the proxy
    # device over 5567, and the proxy will send them to the
    # RemoteDispatcher over 5568. The RemoteDispatcher will push them into
    # the queue, where we can verify that they round-tripped.

    local_accumulator = []

    def local_cb(name, doc):
        local_accumulator.append((name, doc))

    RE([Msg('open_run'), Msg('close_run')], local_cb)

    # This time the Publisher isn't attached to an RE. Send the documents
    # manually. (The idea is, these might have come from a Broker instead...)
    for name, doc in local_accumulator:
        p(name, doc)
    time.sleep(1)

    # Get the two documents from the queue (or timeout --- test will fail)
    remote_accumulator = []
    for i in range(2):
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
    def start_proxy():
        Proxy(5567, 5568).start()

    proxy_proc = multiprocessing.Process(target=start_proxy, daemon=True)
    proxy_proc.start()
    time.sleep(5)  # Give this plenty of time to start up.

    # COMPONENT 2
    # Run a Publisher and a RunEngine in this main process.
    p = Publisher('127.0.0.1:5567', prefix=b'sb')  # noqa
    p2 = Publisher('127.0.0.1:5567', prefix=b'not_sb')  # noqa
    RE.subscribe(p)
    RE.subscribe(p2)

    # COMPONENT 3
    # Run a RemoteDispatcher on another separate process. Pass the documents
    # it receives over a Queue to this process, so we can count them for our
    # test.

    def make_and_start_dispatcher(queue):
        def put_in_queue(name, doc):
            print('putting ', name, 'in queue')
            queue.put((name, doc))

        d = RemoteDispatcher('127.0.0.1:5568', prefix=b'sb')
        d.subscribe(put_in_queue)
        print("REMOTE IS READY TO START")
        d.loop.call_later(9, d.stop)
        d.start()

    queue = multiprocessing.Queue()
    dispatcher_proc = multiprocessing.Process(target=make_and_start_dispatcher,
                                              daemon=True, args=(queue,))
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
    md = {'stuff': {'nested': np.array([1, 2, 3])},
          'scalar_stuff': np.float64(3),
          'array_stuff': np.ones((3, 3))}

    # RE([Msg('open_run', **md), Msg('close_run')], local_cb)
    RE(count([hw.det]), local_cb, **md)
    time.sleep(1)

    # Get the two documents from the queue (or timeout --- test will fail)
    remote_accumulator = []
    for i in range(len(local_accumulator)):
        remote_accumulator.append(queue.get(timeout=2))
    p.close()
    proxy_proc.terminate()
    dispatcher_proc.terminate()
    proxy_proc.join()
    dispatcher_proc.join()
    ra = sanitize_doc(remote_accumulator)
    la = sanitize_doc(local_accumulator)
    assert ra == la
