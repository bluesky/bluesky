# Example: subscribing runs to individual sets of callbacks

from bluesky import RunEngine
from bluesky.callbacks import LiveTable, LivePlot
import bluesky.preprocessors as bpp
import bluesky.plan_stubs as bps
from databroker import Broker
from event_model import RunRouter

from ophyd.sim import hw
hw = hw()

RE = RunEngine({})

db = Broker.named("temp")
RE.subscribe(db.insert)

def factory(name, doc):
    # Runs may be subscribed to different sets of callbacks. Metadata from start
    #   document may be used to identify, which run is currently being started.
    #   In this example, the run key is explicitely added to the start document
    #   and used to identify runs, but other data can be similarly used.
    cb_list = []
    if doc["run_key"] == "run_1":
        cb_list.append(LiveTable([hw.motor1, hw.det1]))
        cb_list.append(LivePlot('det1', x='motor1'))
    elif doc["run_key"] == "run_2":
        cb_list.append(LiveTable([hw.motor1, hw.motor2, hw.det2]))
    return cb_list, []

rr = RunRouter([factory])
RE.subscribe(rr)

@bpp.set_run_key_decorator("run_2")
@bpp.run_decorator(md={"run_key": "run_2"})
def sim_plan_inner(npts):
    for j in range(npts):
        yield from bps.mov(hw.motor1, j * 0.1 + 1, hw.motor2, j * 0.2 - 2)
        yield from bps.trigger_and_read([hw.motor1, hw.motor2, hw.det2])

@bpp.set_run_key_decorator("run_1")
@bpp.run_decorator(md={"run_key": "run_1"})
def sim_plan_outer(npts):
    for j in range(int(npts/2)):
        yield from bps.mov(hw.motor1, j)
        yield from bps.trigger_and_read([hw.motor1, hw.det1])

    yield from sim_plan_inner(npts + 1)

    for j in range(int(npts/2), npts):
        yield from bps.mov(hw.motor1, j)
        yield from bps.trigger_and_read([hw.motor1, hw.det1])
