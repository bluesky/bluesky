import multiprocessing
import pprint
import time

import numpy as np
import pytest

from confluent_kafka import Consumer, KafkaException

from bluesky.callbacks.kafka import Publisher, RemoteDispatcher
from bluesky.plans import count


# TODO: consider using kazoo to talk to zookeeper
def kafka_available():
    try:
        consumer_params = {
            'bootstrap.servers': '127.0.0.1',
            'group.id': 'kafka-unit-test',
            'auto.offset.reset': 'latest'
        }
        consumer = Consumer(consumer_params)
        consumer.list_topics(timeout=5)
        return True
    except KafkaException:
        return False


skipif_kafka_not_available = pytest.mark.skipif(
    not kafka_available(), reason='Failed to connect to Kafka on 127.0.0.1'
)


@skipif_kafka_not_available
def test_kafka(RE, hw):
    # COMPONENT 1
    # A Kafka server must be running

    # COMPONENT 2
    # Run a Publisher and a RunEngine in this process
    kafka_publisher = Publisher(address='127.0.0.1:9092')
    RE.subscribe(kafka_publisher)

    # COMPONENT 3
    # Run a RemoteDispatcher on another separate process. Pass the documents
    # it receives over a Queue to this process, so we can count them for our
    # test.

    def make_and_start_dispatcher(queue):
        def put_in_queue(name, doc):
            print('putting ', name, 'in queue')
            queue.put((name, doc))

        kafka_dispatcher = RemoteDispatcher('127.0.0.1:9092', group_id='kafka-unit-test')
        kafka_dispatcher.subscribe(put_in_queue)
        kafka_dispatcher.start()

    queue = multiprocessing.Queue()
    dispatcher_proc = multiprocessing.Process(target=make_and_start_dispatcher,
                                              daemon=True, args=(queue,))
    dispatcher_proc.start()
    time.sleep(10)  # As above, give this plenty of time to start.

    local_accumulator = []
    def local_cb(name, doc):
        print('local_cb: {}'.format(name))
        local_accumulator.append((name, doc))

    # Check that numpy stuff is sanitized by putting some in the start doc.
    # md = {'stuff': {'nested': np.array([1, 2, 3])},
    #      'scalar_stuff': np.float64(3),
    #      'array_stuff': np.ones((3, 3))}
    md = {}

    RE.subscribe(local_cb)
    RE(count([hw.det]), md=md)

    # Get the documents from the queue (or timeout --- test will fail)
    remote_accumulator = []
    for i in range(len(local_accumulator)):
        remote_accumulator.append(queue.get(timeout=2))

    dispatcher_proc.terminate()
    dispatcher_proc.join()

    print('local_accumulator:')
    pprint.pprint(local_accumulator)
    print('remote_accumulator:')
    pprint.pprint(remote_accumulator)

    # numpy arrays cause trouble sometimes
    assert len(remote_accumulator) == len(local_accumulator)
    assert sorted(remote_accumulator) == sorted(local_accumulator)


from multiprocessing.managers import BaseManager
from queue import Queue


class QueueManager(BaseManager): pass


@skipif_kafka_not_available
def test_kafka_queue_mgr(RE, hw):
    # COMPONENT 1
    # A Kafka server must be running

    # COMPONENT 2
    # Run a Publisher and a RunEngine in this process
    kafka_publisher = Publisher(address='127.0.0.1:9092')
    RE.subscribe(kafka_publisher)

    # COMPONENT 3
    # Run a RemoteDispatcher on another separate process. Pass the documents
    # it receives over a Queue to this process, so we can count them for our
    # test.
    def start_queue_mgr():
        queue = Queue()
        QueueManager.register('get_queue', callable=lambda: queue)
        m = QueueManager(address=('', 50000), authkey=b'justatest')
        s = m.get_server()
        s.serve_forever()

    queue_mgr_proc = multiprocessing.Process(target=start_queue_mgr, daemon=True, args=())
    queue_mgr_proc.start()
    time.sleep(10)

    def start_dispatcher():
        def put_in_queue(name, doc):
            print('putting ', name, 'in queue')
            QueueManager.register('get_queue')
            m = QueueManager(address=('', 50000), authkey=b'justatest')
            m.connect()
            queue = m.get_queue()
            queue.put((name, doc))

        kafka_dispatcher = RemoteDispatcher('127.0.0.1:9092', group_id='kafka-unit-test')
        kafka_dispatcher.subscribe(put_in_queue)
        kafka_dispatcher.start()

    dispatcher_proc = multiprocessing.Process(target=start_dispatcher, daemon=True, args=())
    dispatcher_proc.start()
    time.sleep(10)  # As above, give this plenty of time to start.

    local_accumulator = []
    def local_cb(name, doc):
        print('local_cb: {}'.format(name))
        local_accumulator.append((name, doc))

    # Check that numpy stuff is sanitized by putting some in the start doc.
    # md = {'stuff': {'nested': np.array([1, 2, 3])},
    #      'scalar_stuff': np.float64(3),
    #      'array_stuff': np.ones((3, 3))}
    md = {}

    RE.subscribe(local_cb)
    RE(count([hw.det]), md=md)

    QueueManager.register('get_queue')
    m = QueueManager(address=('', 50000), authkey=b'justatest')
    m.connect()
    queue = m.get_queue()

    # Get the documents from the queue (or timeout --- test will fail)
    remote_accumulator = []
    for i in range(len(local_accumulator)):
        remote_accumulator.append(queue.get(timeout=2))

    dispatcher_proc.terminate()
    dispatcher_proc.join()

    queue_mgr_proc.terminate()
    queue_mgr_proc.join()

    print('local_accumulator:')
    pprint.pprint(local_accumulator)
    print('remote_accumulator:')
    pprint.pprint(remote_accumulator)

    # numpy arrays cause trouble sometimes
    assert len(remote_accumulator) == len(local_accumulator)
    assert remote_accumulator == local_accumulator
