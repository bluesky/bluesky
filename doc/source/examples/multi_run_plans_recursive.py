# Example: recursive runs

from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback
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
    # Each run is subscribed to independent instance of BEC
    bec = BestEffortCallback()
    return [bec], []

rr = RunRouter([factory])
RE.subscribe(rr)

# Call counter and the maximum number calls
n_calls, n_calls_max = 0, 3

def sim_plan_recursive(npts):
    global n_calls, n_calls_max

    n_calls += 1  # Increment counter
    if n_calls <= n_calls_max:
        # Generate unique key for each run. The key generation algorithm
        #   must only guarantee that execution of the runs that are assigned
        #   the same key will never overlap in time.
        run_key = f"run_key_{n_calls}"

        @bpp.set_run_key_decorator(run_key)
        @bpp.run_decorator(md={})
        def plan(npts):

            for j in range(int(npts/2)):
                yield from bps.mov(hw.motor1, j * 0.2)
                yield from bps.trigger_and_read([hw.motor1, hw.det1])

            # Different parameter values may be passed to the recursively called plans
            yield from sim_plan_recursive(npts + 2)

            for j in range(int(npts/2), npts):
                yield from bps.mov(hw.motor1, j * 0.2)
                yield from bps.trigger_and_read([hw.motor1, hw.det1])

        yield from plan(npts)
