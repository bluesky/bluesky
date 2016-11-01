from bluesky.tests import requires_ophyd
from bluesky.tests.utils import MsgCollector
from bluesky.plans import (ramp_plan, trigger_and_read)
from bluesky import Msg
from bluesky.utils import RampFail
import numpy as np
import time
import pytest


@requires_ophyd
def test_ramp(RE, db):
    from ophyd.positioner import SoftPositioner
    from ophyd import StatusBase
    from bluesky.examples import SynGauss

    tt = SoftPositioner(name='mot')
    tt.set(0)
    dd = SynGauss('det', tt, 'mot', 0, 3)

    st = StatusBase()

    def kickoff():
        yield Msg('null')
        for j, v in enumerate(np.linspace(-5, 5, 10)):
            RE.loop.call_later(.1 * j, lambda v=v: tt.set(v))
        RE.loop.call_later(1.2, st._finished)
        return st

    def inner_plan():
        yield from trigger_and_read([dd])

    g = ramp_plan(kickoff(), tt, inner_plan, period=0.08)
    RE.subscribe('all', db.mds.insert)
    RE.msg_hook = MsgCollector()
    rs_uid, = RE(g)
    hdr = db[-1]
    assert hdr.start.uid == rs_uid
    assert len(hdr.descriptors) == 2

    assert set([d['name'] for d in hdr.descriptors]) == \
        set(['primary', 'mot_monitor'])

    primary_events = list(db.get_events(hdr, stream_name='primary'))
    assert len(primary_events) > 11

    monitor_events = list(db.get_events(hdr, stream_name='mot_monitor'))
    assert len(monitor_events) == 10


@requires_ophyd
def test_timeout(RE):
    from ophyd.positioner import SoftPositioner
    from bluesky.examples import SynGauss
    from ophyd import StatusBase

    mot = SoftPositioner(name='mot')
    mot.set(0)
    det = SynGauss('det', mot, 'mot', 0, 3)

    def kickoff():
        yield Msg('null')
        for j in range(5):
            RE.loop.call_later(.05 * j, lambda j=j: mot.set(j))

        return StatusBase()

    def inner_plan():
        yield from trigger_and_read([det])

    g = ramp_plan(kickoff(), mot, inner_plan, period=.01, timeout=.1)

    start = time.time()
    with pytest.raises(RampFail):
        RE(g)
    stop = time.time()
    elapsed = stop - start

    assert .1 < elapsed < .2
