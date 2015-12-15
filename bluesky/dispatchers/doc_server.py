from jsonsocket import Client, Server
from bluesky.run_engine import Dispatcher


class DocumentClient:

    def __init__(self, host, port):
        self.client = Client()
        self.client.connect(host, port)
        self.dispatcher = Dispatcher()
        self.subscribe = self.dispatcher.subscribe

    def __call__(self):
        while True:
            response = self.client.recv()
            (name, doc), = response.items()
            self.dispatcher.process(name, doc)

    def __del__(self):
        self.client.close()


class DocumentServer:

    def __init__(self, host, port):
        self.server = Server()

    def __call__(self, name, doc):
        self.server.send({name: doc})

    def __del__(self):
        self.server.close()
