#!/usr/bin/env python

# tests require pytest-cov and pytest-xdist
import os
import signal
import sys
import pytest

try:
    from pcaspy import Driver, SimpleServer
    from multiprocessing import Process

    def to_subproc():

        prefix = 'BSTEST:'
        pvdb = {
            'VAL': {
                'prec': 3,
            },
        }

        class myDriver(Driver):
            def __init__(self):
                super(myDriver, self).__init__()

        if __name__ == '__main__':
            server = SimpleServer()
            server.createPV(prefix, pvdb)
            driver = myDriver()

            # process CA transactions
            while True:
                try:
                    server.process(0.1)
                except KeyboardInterrupt:
                    break

    p = Process(target=to_subproc)
except ImportError:
    p = None


def run():
    if p is not None:
        p.start()
    try:
        # adding rxs to show extra info on skips and xfails
        args = ['--cov bluesky -srx -vv']
        args.extend(sys.argv)
        pytest.main(args)
    finally:
        if p is not None:
            os.kill(p.pid, signal.SIGINT)
            p.join()

if __name__ == '__main__':
    run()
