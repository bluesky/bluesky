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
    serializer: function, optional
        optional function to serialize data. Default is pickle.dumps

    Example
    -------

    Publish from a RunEngine to a Kafka server on localhost on port 9092.

    >>> publisher = Publisher('localhost:9092')
    >>> RE = RunEngine({})
    >>> RE.subscribe(publisher)
    """
    def __init__(self, address, *, serializer=pickle.dumps):
        self.address = address
        self.producer = Producer({'bootstrap.servers': self.address})
        self._serializer = serializer

    def __call__(self, name, doc):
        print(f'name: {name}')
        print(f'doc:\n{doc}')
        doc = copy.deepcopy(doc)
        self.producer.poll(0)
        self.producer.produce(name, self._serializer(doc), callback=delivery_report)
        print('flush')
        self.producer.flush()

    def close(self):
        pass
