# Example: nested runs

from databroker import Broker
from event_model import RunRouter
from ophyd.sim import hw

import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback

hw = hw()

RE = RunEngine({})

db = Broker.named("temp")
RE.subscribe(db.insert)


def factory(name, doc):
    # Documents from each run is routed to an independent
    #   instance of BestEffortCallback
    bec = BestEffortCallback()
    return [bec], []


rr = RunRouter([factory])
RE.subscribe(rr)


@bpp.set_run_key_decorator("run_2")
@bpp.run_decorator(md={})
def sim_plan_inner(npts):
    yield from bps.declare_stream(hw.motor1, hw.motor2, hw.det2, name="primary")
    for j in range(npts):
        yield from bps.mov(hw.motor1, j * 0.1 + 1, hw.motor2, j * 0.2 - 2)
        yield from bps.trigger_and_read([hw.motor1, hw.motor2, hw.det2])


@bpp.set_run_key_decorator("run_1")
@bpp.run_decorator(md={})
def sim_plan_outer(npts):
    yield from bps.declare_stream(hw.motor, hw.det, name="primary")
    for j in range(int(npts / 2)):
        yield from bps.mov(hw.motor, j * 0.2)
        yield from bps.trigger_and_read([hw.motor, hw.det])

    yield from sim_plan_inner(npts + 1)

    for j in range(int(npts / 2), npts):
        yield from bps.mov(hw.motor, j * 0.2)
        yield from bps.trigger_and_read([hw.motor, hw.det])
