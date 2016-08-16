from bluesky.tests import requires_ophyd
from bluesky.tests.utils import MsgCollector
from bluesky.plans import (abs_set, ramp_plan, trigger_and_read)
from bluesky import Msg
import numpy as np


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
        for j, v in enumerate(np.linspace(-5, 5, 25)):
            RE.loop.call_later(.05 * j, lambda v=v: tt.set(v))
        RE.loop.call_later(.05 * 30, st._finished)
        return st

    def inner_plan():
        yield from trigger_and_read([dd])

    g = ramp_plan(kickoff(), tt, inner_plan, period=0.04)
    RE._subscribe_lossless('all', db.mds.insert)
    RE.msg_hook = MsgCollector()
    rs_uid, = RE(g)
    hdr = db[-1]
    assert hdr.start.uid == rs_uid
    assert len(hdr.descriptors) == 2

    assert set([d['name'] for d in hdr.descriptors]) == \
        set(['primary', 'mot-monitor'])

    primary_events = list(db.get_events(hdr, stream_name='primary'))
    assert len(primary_events) > 27

    monitor_events = list(db.get_events(hdr, stream_name='mot-monitor'))
    assert len(monitor_events) == 25
