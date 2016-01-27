import signal
import sys
import asyncio
import os
import time as ttime
from bluesky.examples import loop
from bluesky import Msg
from bluesky.tests.utils import setup_test_run_engine
RE = setup_test_run_engine()

def run():
    ev = asyncio.Event()
    def done():
        print("Done")
        ev.set()

    pid = os.getpid()

    def sim_kill():
        os.kill(pid, signal.SIGINT)
        ttime.sleep(0.1)
        os.kill(pid, signal.SIGINT)

    scan = [Msg('checkpoint'), Msg('wait_for', [ev.wait(), ]), ]
    RE.verbose = True
    assert RE.state == 'idle'
    start = ttime.time()
    loop.call_later(1, sim_kill)
    loop.call_later(2, done)

    RE(scan)
    assert RE.state == 'idle'
    stop = ttime.time()

if __name__ == "__main__":
    try:
        run()
    except AssertionError:
        sys.exit(1)
    sys.exit(0)