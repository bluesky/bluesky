import json

from bluesky.callbacks.core import CallbackBase


class Publisher(CallbackBase):
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
        self._publisher = KafkaProducer(bootstrap_servers)
        self._topic = topic

    def start(self, name, doc):
        self._publiser.send(self._topic, json.dumps(doc))






