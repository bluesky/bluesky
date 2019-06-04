import copy
import pickle

from confluent_kafka import Consumer, Producer

from ..run_engine import Dispatcher, DocumentNames


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
        optional function to serialize data. Default is msgpack

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


class RemoteDispatcher(Dispatcher):
    """
    Dispatch documents received over the network from a Kafka server.

    Parameters
    ----------
    address : str or tuple
        Address of a Kafka server, given either as a string like
        ``'127.0.0.1:9022'`` or as a tuple like ``('127.0.0.1', 9022)``
    deserializer: function, optional
        optional function to deserialize data. Default is msgpack

    Example
    -------

    Print all documents generated by remote RunEngines.

    >>> d = RemoteDispatcher(('localhost', 9022))
    >>> d.subscribe(print)
    >>> d.start()  # runs until interrupted
    """
    def __init__(self, address, *, deserializer=pickle.loads):
        if isinstance(address, str):
            self.address, self.port = address.split(':', maxsplit=1)
        elif isinstance(address, tuple):
            self.address, self.port = address[0], address[1]
        else:
            raise TypeError('"address" argument must be str or tuple')
        self._deserializer = deserializer
        self.address = (address[0], int(address[1]))

        self.consumer = Consumer({
            'bootstrap.servers': f'{self.address}:{self.port}',
            'group.id': None,
            'auto.offset.reset': 'earliest'
        })

        #self.loop = None  # ???
        #self._task = None
        self.closed = False

        super().__init__()

    def _poll(self):
        while True:
            msg = self.consumer.poll()
            name = msg.topic()

            if msg.value() is None:
                print(f'{name} doc is None')
            else:
                doc = self._deserializer(msg.value())
                self.process(DocumentNames[name], doc)

    def start(self):
        if self.closed:
            raise RuntimeError("This RemoteDispatcher has already been "
                               "started and interrupted. Create a fresh "
                               "instance with {}".format(repr(self)))
        try:
            self.consumer.subscribe(topics=('start', 'descriptor', 'event', 'stop'))
            #self._task = self.loop.create_task(self._poll())
            #self.loop.run_forever()
        except:
            self.stop()
            raise

    def stop(self):
        #if self._task is not None:
            #self._task.cancel()
            #self.loop.stop()
        self.consumer.close()
        #self._task = None
        self.closed = True
