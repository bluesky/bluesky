import json
import zmq
from multiprocessing import Process, Queue


def start_run_engine(RE, host='127.0.0.1', port=5554):
    msg_queue = Queue()
    response_queue = Queue()
    server_process = Process(target=start_server,
                             args=(msg_queue, response_queue, host, port))
    server_process.start()
    exception = None

    def plan():
        while True:
            raw_msg = msg_queue.get()
            msg = raw_msg['msg']
            response = yield msg
            response_queue.put({'response': repr(response)})

    while True:
        if RE.state == 'idle':
            response = {'state': RE.state}
            if exception is not None:
                response['exception'] = repr(exception)
                exception = None
            response_queue.put(json.dumps(response))
            try:
                RE(plan())
            except Exception as exc:
                print('EXCEPTION RAISED:', exc)
                print('The RunEngine will be re-started....')
                exception = exc
                continue
        # The RunEngine is paused or idle.
        response_queue.put(json.dumps({'state': RE.state}))
        raw_msg = msg_queue.get()
        if 'change_state' in raw_msg:
            getattr(RE, raw_msg['change_state'])()
        
    server_process.join()


def start_server(msg_queue, response_queue, host='127.0.0.1', port=5554):

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PAIR)
    sock.bind("tcp://{host}:{port}".format(host=host, port=port))

    try:
        print('Server is ready to receive messages.')
        while True:
            response = response_queue.get()
            print('SENDING:', response)
            sock.send_string(json.dumps(response))

            payload = sock.recv()  # bytes
            msg = json.loads(payload.decode())
            print('RECEIVED:', msg)
            msg_queue.put(msg)
    except:
        pass
    finally:
        sock.close()
        ctx.term()
