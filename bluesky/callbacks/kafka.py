import json
import asyncio

from kafka import KafkaProducer, KafkaConsumer
from bluesky.run_engine import Dispatcher

from bluesky.callbacks.core import CallbackBase


class Producer(CallbackBase):
    """
    A callback that publishes documents to kafka.

    Parameters
    ----------
    boostrap_servers: list
        list of servers the brokers run on
    topic: str
        topic to push to

    Example
    -------
    >>> publisher = KafkaCallback('localhost:9092')
    """
    def __init__(self, bootstrap_servers, topic):
        self._publisher = KafkaProducer(bootstrap_servers=bootstrap_servers)
        self._topic = topic

    def start(self, doc):
        bs = json.dumps(doc).encode('utf-8')
        self._publisher.send(self._topic, bs)

class Consumer(Dispatcher):
    '''
    '''
    def __init__(self, bootstrap_servers, topic):
        self._topic = topic
        self._consumer = KafkaConsumer(topic, booststrap_servers=bootstrap_servers)

    def

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
                doc = pickle.loads(doc)
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




