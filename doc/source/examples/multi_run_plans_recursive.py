# Example: demo of a plan using dynamically generated run key

from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback
import bluesky.preprocessors as bpp
import bluesky.plan_stubs as bps
from databroker import Broker
from event_model import RunRouter

from ophyd.sim import hw
det1, motor1 = hw().det1, hw().motor1

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

# Current and maximum number plan calls
n_calls, n_calls_max = 0, 3

def sim_plan_recursive(npts):
    global n_calls, n_calls_max

    n_calls += 1  # Increment counter
    if n_calls <= n_calls_max:
        # Generate unique run key
        run_key = f"run_key_{n_calls}"

        @bpp.set_run_key_decorator(run_key)
        @bpp.run_decorator(md={})
        def plan(npts):

            for j in range(int(npts/2)):
                yield from bps.mov(motor1, j)
                yield from bps.trigger_and_read([motor1, det1])

            # Note, that the parameter value (number of scan points)
            #   is increased at each iteration
            yield from sim_plan_recursive(npts + 2)

            for j in range(int(npts/2), npts):
                yield from bps.mov(motor1, j)
                yield from bps.trigger_and_read([motor1, det1])

        yield from plan(npts)
        
# RE(sim_plan_recursive(4))
