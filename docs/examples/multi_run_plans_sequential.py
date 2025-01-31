# Example: consecutive execution of single-run plans

from databroker import Broker
from ophyd.sim import hw

from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback
from bluesky.plans import rel_scan, scan

hw = hw()

RE = RunEngine({})

db = Broker.named("temp")
RE.subscribe(db.insert)

bec = BestEffortCallback()
RE.subscribe(bec)


def plan_sequential_runs(npts):
    # Single-run plans may be called consecutively. No special handling is required
    #   as long as the previous scan is closed before the next one is opened
    yield from scan([hw.det], hw.motor, -1, 1, npts)
    yield from rel_scan([hw.det, hw.noisy_det], hw.motor, -1, 1, npts)
