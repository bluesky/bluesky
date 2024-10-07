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

    # setup devices
    motor = SoftPositioner(name="mot")
    motor.set(0)
    simulated_detector = SynGauss("det", motor, "mot", 0, 3)

    status = StatusBase()

    # define plan functions nested setup
    def kickoff():
        yield Msg("null")
        for j, v in enumerate(np.linspace(-5, 5, 10)):
            RE.loop.call_later(0.1 * j, lambda v=v: motor.set(v))
        RE.loop.call_later(1.2, status._finished)
        print("YOLO")
        return status

    first = True

    def inner_plan():
        nonlocal first
        if first:
            yield from declare_stream(simulated_detector, name="primary")
            first = False

        yield from trigger_and_read([simulated_detector])

    ramp_plan_generator = ramp_plan(kickoff(), motor, inner_plan, period=0.08)
    document_collector = DocCollector()

    # work subscription logic
    RE.subscribe(document_collector.insert)
    rs = RE(ramp_plan_generator)
    uid = rs.run_start_uids[0] if RE.call_returns_result else rs[0]

    assert document_collector.start[0]["uid"] == uid
    descriptors = document_collector.descriptor[uid]
    assert len(descriptors) == 2
    descs = {d["name"]: d for d in descriptors}

    assert set(descs) == set(["primary", "mot_monitor"])  # noqa: C405

    primary_events = document_collector.event[descs["primary"]["uid"]]
    assert len(primary_events) > 11

    monitor_events = document_collector.event[descs["mot_monitor"]["uid"]]
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
