import os
import zmq
import yaml


name = 'zmq_document_forwarder'
filenames = [
    os.path.join('/etc', name + '.yml'),
    os.path.join(os.path.expanduser('~'), '.config', name, 'connection.yml'),
    ]

config = {}
for filename in filenames:
    if os.path.isfile(filename):
        print('found config file at', filename)
        with open(filename) as f:
            config.update(yaml.load(f))


def main(frontend_port, backend_port):

    try:
        context = zmq.Context(1)
        # Socket facing clients
        frontend = context.socket(zmq.SUB)
        frontend.bind("tcp://*:%d" % frontend_port)
        
        frontend.setsockopt_string(zmq.SUBSCRIBE, "")
        
        # Socket facing services
        backend = context.socket(zmq.PUB)
        backend.bind("tcp://*:%d" % backend_port)

        zmq.device(zmq.FORWARDER, frontend, backend)
    finally:
        frontend.close()
        backend.close()
        context.term()


if __name__ == "__main__":
    main(int(config['frontend_port']), int(config['backend_port']))
