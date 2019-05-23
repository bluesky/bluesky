import copy
import pickle

from confluent_kafka import Producer


def delivery_report(err, msg):
    """ Called once for each message produced to indicate delivery result.
        Triggered by poll() or flush(). """
    if err is not None:
        print('Message delivery failed: {}'.format(err))
    else:
        print('Message delivered to {} [{}]'.format(msg.topic(), msg.partition()))


class Publisher:
    """
    A callback that publishes documents to a Kafka server.

    Parameters
    ----------
    address : string
        Address of a running Kafka server as a string like
        ``'127.0.0.1:9092'``
    topic : string
        Kafka topic string
    serializer: function, optional
        optional function to serialize data. Default is pickle.dumps

    Example
    -------

    Publish from a RunEngine to a Kafka server on localhost on port 9092 with topic 'some-topic'.

    >>> publisher = Publisher('localhost:9092', 'some-topic')
    >>> RE = RunEngine({})
    >>> RE.subscribe(publisher)
    """
    def __init__(self, address, *, topic, serializer=pickle.dumps):
        self.address = address
        self.topic = topic
        self.producer = Producer({'bootstrap.servers': self.address})
        self._serializer = serializer

    def __call__(self, name, doc):
        doc = copy.deepcopy(doc)
        self.producer.produce(self.topic, self._serializer(doc), callback=delivery_report)

    def close(self):
        pass
