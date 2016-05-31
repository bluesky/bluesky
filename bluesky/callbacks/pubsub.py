import zmq
import random
import socket
import os


def make_publisher(host, port=5559):
    """
    Create a callback function that publishes documents to a socket.

    Parameters
    ----------
    host : string
        name of host running forwarder_device
        (See bluesky/examples/forwarder_device.py.)
    port : int, optional
        default 5559

    Example
    -------
    Publish all documents to port 5559.

    >>> RE.subscribe('all', make_publisher('localhost'))
    """
    pid = os.getpid()
    local_hostname = socket.gethostname()
    url = "tcp://%s:%d" % (host, port)
    fmt_string = '{hostname} {pid} {{name}} {{doc}}'.format(
            hostname=local_hostname, pid=pid)

    context = zmq.Context()
    s = context.socket(zmq.PUB)
    s.connect(url)
    def cb(name, doc):
        message = fmt_string.format(name=name, doc=doc)
        s.send_string(message)
    return cb


def subscribe(host, port=5560, filter_hostname=None, filter_pid=None):
    """
    Yield documents received over a socket.

    Parameters
    ----------
    host : string
        name of host running forwarder_device
        (See bluesky/examples/forwarder_device.py.)
    port : int, optional
        default 5560
    filter_hostname : string, optional
        only process documents from a host with this name
    filter_pid : int, optional
        only process documents from a process with this pid

    Example
    -------
    Create a LivePlot and feed it documents published from 'hostname:7878'

    >>> lp = LivePlot('y', 'x')
    >>> for name, doc in subscribe('localhost'):
    ...     lp(name, doc)
    """
    context = zmq.Context()
    s = context.socket(zmq.SUB)
    url = "tcp://%s:%d" % (host, port)
    s.connect(url)
    s.setsockopt_string(zmq.SUBSCRIBE, "")
    while True:
        message = s.recv()
        hostname, pid, name, doc = message.decode().split(' ', 3)
        if ((filter_hostname is None or filter_hostname == hostname) and 
            (filter_pid is None or filter_pid == pid)):
            yield name, doc
