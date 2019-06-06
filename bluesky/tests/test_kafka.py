from bluesky.callbacks.kafka import Publisher
from bluesky.plans import count


def test_kafka(RE, hw):
    kafka_publisher = Publisher(address='localhost:9092')
    RE.subscribe(kafka_publisher)

    def local_cb(name, doc):
        print(name)

    RE(count([hw.det]), local_cb)
