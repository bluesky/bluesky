# %matplotlib qt5
import shutil
from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from databroker import Broker
from databroker.v0 import temp_config
from ophyd.sim import det4, motor1, motor2

from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback
from bluesky.plan_stubs import mov
from bluesky.plans import (
    inner_product_scan,
    outer_product_scan,
    relative_inner_product_scan,
    relative_outer_product_scan,
    scan_nd,
)

plt.ion()

# db setup
config = temp_config()
tempdir = config["metadatastore"]["config"]["directory"]


def cleanup():
    shutil.rmtree(tempdir)


db = Broker.named("temp")

RE = RunEngine({})
# subscribe BEC
bec = BestEffortCallback()
RE.subscribe(bec)
RE.subscribe(db.insert)

# move motor to a reproducible location
RE(mov(motor1, 0))  # type: ignore
RE(mov(motor2, 0))  # type: ignore
RE(relative_outer_product_scan([det4], motor1, -1, 0, 10, motor2, -2, 0, 20, True))
RE(outer_product_scan([det4], motor1, -1, 0, 10, motor2, -2, 0, 20, True))

# move motor to a reproducible location
RE(mov(motor1, 0))  # type: ignore
RE(mov(motor2, 0))  # type: ignore
RE(relative_inner_product_scan([det4], 10, motor1, -1, 0, motor2, -2, 0))
RE(inner_product_scan([det4], 10, motor1, -1, 0, motor2, -2, 0))

# do it manually
from cycler import cycler  # noqa: E402

mot1_cycl = cycler(motor1, np.linspace(-1, 0, 10))
mot2_cycl = cycler(motor2, np.linspace(-2, 0, 10))


# inner product
inner_hints = {"fields": [det4.name], "dimensions": [([motor1.name, motor2.name], "primary")]}
RE(scan_nd([det4], mot1_cycl + mot2_cycl), hints=inner_hints)

# outer product
outer_hints: Mapping[str, Any] = {
    "fields": [det4.name],
    "dimensions": [([motor2.name], "primary"), ([motor1.name], "primary")],
}
md = {
    "shape": (20, 20),
    # 'extents': ([-1, 0], [-1, 0]),
    "hints": outer_hints,
}

RE(scan_nd([det4], mot1_cycl * mot2_cycl), **md)

# make 40 points just to test
mot2_cycl = cycler(motor2, np.linspace(-1, 0, 20))
# now try rectilinear gridding
outer_hints = {
    "fields": [det4.name],
    "dimensions": [([motor2.name], "primary"), ([motor1.name], "primary")],
    "gridding": "rectilinear",
}
md = {
    "shape": (10, 20),
    "extents": ([-1, 0], [-1, 0]),
    "hints": outer_hints,
}

RE(scan_nd([det4], mot1_cycl * mot2_cycl), **md)
