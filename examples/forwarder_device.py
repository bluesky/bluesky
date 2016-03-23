import zmq

def main(frontend_port=5559, backend_port=5560):

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
    main()
