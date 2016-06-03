import json
import zmq


def execute_on_remote(plan, host='localhost', port=5554):

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PAIR)
    sock.connect("tcp://{host}:{port}".format(host=host, port=port))

    try:
        for msg in plan:
            response = sock.recv()
            print('RECEIVED:', response.decode())
            json_msg = json.dumps(msg)
            print('SENDING:', json_msg)
            sock.send_string(json_msg)
    except:
        pass
    finally:
        sock.close()
        ctx.term()
