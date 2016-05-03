import pytest
import ophyd
import sys
from bluesky.suspenders import (SuspendBoolHigh,
                                SuspendBoolLow,
                                SuspendFloor,
                                SuspendCeil,
                                SuspendInBand,
                                SuspendOutBand)
from bluesky import RunEngine, Msg
import time as ttime


@pytest.mark.parametrize(
    'klass,sc_args,start_val,fail_val,resume_val,wait_time',
    [(SuspendBoolHigh, (), 0, 1, 0, .5),
     (SuspendBoolLow, (), 1, 0, 1, .5),
     (SuspendFloor, (.5,), 1, 0, 1, .5),
     (SuspendCeil, (.5,), 0, 1, 0, .5),
     (SuspendInBand, (.5, 1.5), 1, 0, 1, .5),
     (SuspendOutBand, (.5, 1.5), 0, 1, 0, .5)])
def test_suspender(klass, sc_args, start_val, fail_val,
                   resume_val, wait_time):
    RE = RunEngine({})
    loop = RE._loop
    if sys.platform == 'darwin':
        pytest.xfail('OSX event loop is different; resolve this later')
    sig = ophyd.Signal()
    my_suspender = klass(sig,
                         *sc_args, sleep=wait_time)
    my_suspender.install(RE)

    def putter(val):
        sig.put(val)

    # make sure we start at good value!
    putter(start_val)
    # dumb scan
    scan = [Msg('checkpoint'), Msg('sleep', None, .2)]
    RE(scan)
    # paranoid
    assert RE.state == 'idle'

    start = ttime.time()
    # queue up fail and resume conditions
    loop.call_later(.1, putter, fail_val)
    loop.call_later(1, putter, resume_val)
    # start the scan
    RE(scan)
    stop = ttime.time()
    # assert we waited at least 2 seconds + the settle time
    delta = stop - start
    print(delta)
    assert delta > 1 + wait_time + .2
