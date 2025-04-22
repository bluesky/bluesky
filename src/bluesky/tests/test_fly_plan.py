import functools
from collections import defaultdict

import pytest
from ophyd import Device
from ophyd.sim import NewTrivialFlyer, StatusBase

from bluesky.plans import fly
from bluesky.tests import requires_ophyd


def call_counter(func):
    @functools.wraps(func)
    def inner(self, *args, **kwargs):
        ret = func(self, *args, **kwargs)
        self.call_counts[func.__name__] += 1
        return ret

    return inner


class AlwaysSucceedsStatus(StatusBase):
    @property
    def success(self):
        return True


class SlowFlyer(NewTrivialFlyer, Device):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.call_counts = defaultdict(int)
        self.complete_timeout = 0.5

    @call_counter
    def complete(self):
        return AlwaysSucceedsStatus(timeout=self.complete_timeout)

    def get_index(self):
        return self.call_counts["collect_asset_docs"] + 1

    @call_counter
    def collect_asset_docs(self, index=None):
        if index is None:
            index = self.get_index()
        if self.call_counts["collect_asset_docs"] == 0:
            yield (
                "resource",
                {
                    "path_semantics": "posix",
                    "resource_kwargs": {"frame_per_point": 1},
                    "resource_path": "det.h5",
                    "root": "/tmp/tmpcvxbqctr/",
                    "spec": "AD_HDF5",
                    "uid": "9123df61-a09f-49ae-9d23-41d4d6c6d788",
                },
            )

        yield (
            "datum",
            {
                "datum_id": "9123df61-a09f-49ae-9d23-41d4d6c6d788/0",
                "datum_kwargs": {"point_number": index},
                "resource": "9123df61-a09f-49ae-9d23-41d4d6c6d788",
            },
        )


@requires_ophyd
def test_fly_plan_no_flush_period(RE):
    slow_flyer = SlowFlyer(name="slow_flyer")

    RE(fly([slow_flyer], stream_name="stream_name"))
    print(slow_flyer.call_counts)

    assert slow_flyer.call_counts["complete"] == 1
    assert slow_flyer.call_counts["collect_asset_docs"] == 1


@requires_ophyd
def test_fly_plan_with_flush_period_set(RE):
    slow_flyer = SlowFlyer(name="slow_flyer")

    RE(fly([slow_flyer], collect_flush_period=0.1, stream_name="stream_name"))

    assert slow_flyer.call_counts["complete"] == 1
    assert slow_flyer.call_counts["collect_asset_docs"] == pytest.approx(slow_flyer.complete_timeout / 0.1, rel=2)
