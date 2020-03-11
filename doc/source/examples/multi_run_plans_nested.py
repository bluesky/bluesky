# Example: demo of the run 'sim_plan_inner' started from the run 'sim_plan_outer'

from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback
import bluesky.preprocessors as bpp
import bluesky.plan_stubs as bps
from databroker import Broker
from event_model import RunRouter

from ophyd.sim import hw
det1, det2, motor1, motor2 = hw().det1, hw().det2, hw().motor1, hw().motor2

RE = RunEngine({})

db = Broker.named("temp")
RE.subscribe(db.insert)

def factory(name, doc):
    # Each run is subscribed to independent instance of BEC
    bec = BestEffortCallback()
    bec(name, doc)
    return [bec], []

rr = RunRouter([factory])
RE.subscribe(rr)

@bpp.set_run_key_decorator("run_2")
@bpp.run_decorator(md={})
def sim_plan_inner(npts):
    for j in range(npts):
        yield from bps.mov(motor1, j * 0.1 + 1, motor2, j * 0.2 - 2)
        yield from bps.trigger_and_read([motor1, motor2, det2])

@bpp.set_run_key_decorator("run_1")
@bpp.run_decorator(md={})
def sim_plan_outer(npts):
    for j in range(int(npts/2)):
        yield from bps.mov(motor1, j)
        yield from bps.trigger_and_read([motor1, det1])

    yield from sim_plan_inner(npts + 1)

    for j in range(int(npts/2), npts):
        yield from bps.mov(motor1, j)
        yield from bps.trigger_and_read([motor1, det1])

# RE(sim_plan_outer(10))
