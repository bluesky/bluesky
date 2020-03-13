# Example: consecutive execution of single-run plans

from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback
from databroker import Broker
from bluesky.plans import scan, rel_scan

from ophyd.sim import hw
hw = hw()

RE = RunEngine({})

db = Broker.named("temp")
RE.subscribe(db.insert)

bec = BestEffortCallback()
RE.subscribe(bec)

def plan_sequential_runs(npts):
    # Single-run plans may be called consecutively. No special handling is required
    #   as long as the previous scan is closed before the next one is opened
    yield from scan([hw.det1], hw.motor1, -1, 1, npts)
    yield from rel_scan([hw.det1, hw.det2], hw.motor1, -1, 1, npts)
