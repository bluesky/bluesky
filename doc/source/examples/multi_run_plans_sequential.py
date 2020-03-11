
from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback
from databroker import Broker
from bluesky.plans import scan, rel_scan

from ophyd.sim import hw
det1, det2, motor = hw().det1, hw().det2, hw().motor

RE = RunEngine({})

db = Broker.named("temp")
RE.subscribe(db.insert)

bec = BestEffortCallback()
RE.subscribe(bec)


def plan_sequential_runs(npts):
    # Two plans are called consecutively
    yield from scan([det1], motor, -1, 1, npts)
    yield from rel_scan([det1, det2], motor, -0.1, 0.1, npts)


# RE(plan_sequential_runs(10))
