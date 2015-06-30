#!/usr/bin/env python
# This file is closely based on tests.py from matplotlib
#
# This allows running the matplotlib tests from the command line: e.g.
#
#   $ python tests.py -v -d
#
# The arguments are identical to the arguments accepted by nosetests.
#
# See https://nose.readthedocs.org/ for a detailed description of
# these options.
import os
import signal
import nose
from bluesky.testing.noseclasses import KnownFailure
from multiprocessing import Process
from pcaspy import Driver, SimpleServer

plugins = [KnownFailure]
env = {"NOSE_WITH_COVERAGE": 1,
       'NOSE_COVER_PACKAGE': ['bluesky'],
       'NOSE_COVER_HTML': 1}


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


def run():
    p.start()
    print(p)
    try:
        nose.main(addplugins=[x() for x in plugins], env=env)
    finally:
        os.kill(p.pid, signal.SIGINT)
        p.join()

if __name__ == '__main__':
    run()
