#!/usr/bin/env python
# This file is closely based on tests.py from matplotlib
#
# This allows running the matplotlib tests from the command line: e.g.
#
#   $ python run_tests.py -v -d
#
# The arguments are identical to the arguments accepted by nosetests.
#
# See https://nose.readthedocs.org/ for a detailed description of
# these options.
import os
import signal
import nose
from bluesky.testing.noseclasses import KnownFailure

plugins = [KnownFailure]
env = {"NOSE_WITH_COVERAGE": 1,
       'NOSE_COVER_PACKAGE': ['bluesky'],
       'NOSE_COVER_HTML': 1}

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
        nose.main(addplugins=[x() for x in plugins], env=env)
    finally:
        if p is not None:
            os.kill(p.pid, signal.SIGINT)
            p.join()

if __name__ == '__main__':
    run()
