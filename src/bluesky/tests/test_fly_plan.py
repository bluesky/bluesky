import functools
from collections import defaultdict
from time import time

import pytest
from event_model.documents.event import PartialEvent
from ophyd import Component as Cpt
from ophyd import Device
from ophyd.sim import StatusBase, NewTrivialFlyer

from bluesky import Msg
from bluesky.plan_stubs import (
    close_run,
    collect_while_completing,
    complete,
    complete_all,
    declare_stream,
    kickoff,
    kickoff_all,
    open_run,
    wait,
)
from bluesky.plans import count, fly
from bluesky.protocols import Collectable, Preparable
from bluesky.run_engine import IllegalMessageSequence
from bluesky.tests import requires_ophyd
from bluesky.tests.utils import DocCollector


def call_counter(func):
    @functools.wraps(func)
    def inner(self, *args, **kwargs):
        self.call_counts[func.__name__] += 1
        return func(self, *args, **kwargs)

    return inner

class AlwaysSucceedsStatus(StatusBase):

    @property
    def success(self):
        return True

class SlowFlyer(NewTrivialFlyer, Device):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.call_counts = defaultdict(int)

    @call_counter
    def complete(self):

        return AlwaysSucceedsStatus(timeout=0.5)

    def get_index(self):
        return self.call_counts["collect_asset_docs"] + 1

    @call_counter
    def collect_asset_docs(self, index=None):
        if index is None:
            index = self.get_index()
        if self.call_counts["collect_asset_docs"] == 0:
            yield 'resource', {
                'path_semantics': 'posix',
                'resource_kwargs': {'frame_per_point': 1},
                'resource_path': 'det.h5',
                'root': '/tmp/tmpcvxbqctr/',
                'spec': 'AD_HDF5',
                'uid': '9123df61-a09f-49ae-9d23-41d4d6c6d788'
            }
        
        yield 'datum', {
            'datum_id': '9123df61-a09f-49ae-9d23-41d4d6c6d788/0',
            'datum_kwargs': {'point_number': index},
            'resource': '9123df61-a09f-49ae-9d23-41d4d6c6d788'}


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
    assert slow_flyer.call_counts["collect_asset_docs"] > 1
