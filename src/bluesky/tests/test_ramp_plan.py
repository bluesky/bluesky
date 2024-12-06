import time

import numpy as np
import pytest

from bluesky import Msg
from bluesky.plan_stubs import declare_stream, trigger_and_read
from bluesky.plans import ramp_plan
from bluesky.tests import requires_ophyd
from bluesky.tests.utils import DocCollector
from bluesky.utils import RampFail


@requires_ophyd
def test_ramp(RE):
    from ophyd import StatusBase
    from ophyd.positioner import SoftPositioner
    from ophyd.sim import SynGauss

    import bluesky.utils as bsu

    RE.msg_hook = bsu.ts_msg_hook

    tt = SoftPositioner(name="mot")
    tt.set(0)
    dd = SynGauss("det", tt, "mot", 0, 3)

    st = StatusBase()

    def kickoff():
        yield Msg("null")
        for j, v in enumerate(np.linspace(-5, 5, 10)):
            RE.loop.call_later(0.1 * j, lambda v=v: tt.set(v))
        RE.loop.call_later(1.2, st._finished)
        print("YOLO")
        return st

    first = True

    def inner_plan():
        nonlocal first
        if first:
            yield from declare_stream(dd, name="primary")
            first = False

        yield from trigger_and_read([dd])

    g = ramp_plan(kickoff(), tt, inner_plan, period=0.08)
    db = DocCollector()
    RE.subscribe(db.insert)
    rs = RE(g)
    if RE.call_returns_result:
        uid = rs.run_start_uids[0]
    else:
        uid = rs[0]
    assert db.start[0]["uid"] == uid
    assert len(db.descriptor[uid]) == 2
    descs = {d["name"]: d for d in db.descriptor[uid]}

    assert set(descs) == set(["primary", "mot_monitor"])  # noqa: C405

    primary_events = db.event[descs["primary"]["uid"]]
    assert len(primary_events) > 11

    monitor_events = db.event[descs["mot_monitor"]["uid"]]
    # the 10 from the updates, 1 from 'run at subscription time'
    assert len(monitor_events) == 11


@requires_ophyd
def test_timeout(RE):
    from ophyd import StatusBase
    from ophyd.positioner import SoftPositioner
    from ophyd.sim import SynGauss

    mot = SoftPositioner(name="mot")
    mot.set(0)
    det = SynGauss("det", mot, "mot", 0, 3)

    def kickoff():
        # This process should take a total of 5 seconds, but it will be
        # interrupted after 0.1 seconds via the timeout keyword passed to
        # ramp_plan below.
        yield Msg("null")
        for j in range(100):
            RE.loop.call_later(0.05 * j, lambda j=j: mot.set(j))

        return StatusBase()

    first = True

    def inner_plan():
        nonlocal first
        if first:
            yield from declare_stream(det, name="primary")
            first = False

        yield from trigger_and_read([det])

    g = ramp_plan(kickoff(), mot, inner_plan, period=0.01, timeout=0.1)

    start = time.time()
    with pytest.raises(RampFail):
        RE(g)
    stop = time.time()
    elapsed = stop - start

    # This test has a tendency to randomly fail, so we're giving it plenty of
    # time to run. 4, though much greater than 0.1, is still less than 5.
    assert 0.1 < elapsed < 4
