import functools
from collections import defaultdict

import pytest

from bluesky import Msg
from bluesky.plans import fly, count
from bluesky.run_engine import IllegalMessageSequence
from bluesky.tests import requires_ophyd
from bluesky.tests.utils import DocCollector

from ophyd import Component as Cpt, Device
from ophyd.sim import NullStatus, TrivialFlyer


@requires_ophyd
def test_flyer_with_collect_asset_documents(RE):

    from ophyd.sim import det, new_trivial_flyer, trivial_flyer
    from bluesky.preprocessors import fly_during_wrapper

    assert hasattr(new_trivial_flyer, "collect_asset_docs")
    assert not hasattr(trivial_flyer, "collect_asset_docs")
    RE(fly_during_wrapper(count([det], num=5), [new_trivial_flyer, trivial_flyer]))


@requires_ophyd
def test_collect_uncollected_and_log_any_errors(RE):
    # test that if stopping one motor raises an error, we can carry on
    collected = {}

    from ophyd.sim import TrivialFlyer

    class DummyFlyerWithFlag(TrivialFlyer):
        def collect(self):
            collected[self.name] = True
            super().collect()

    class BrokenDummyFlyerWithFlag(DummyFlyerWithFlag):
        def collect(self):
            super().collect()
            raise Exception

    flyer1 = DummyFlyerWithFlag()
    flyer1.name = "flyer1"
    flyer2 = BrokenDummyFlyerWithFlag()
    flyer2.name = "flyer2"

    collected.clear()
    RE([Msg("open_run"), Msg("kickoff", flyer1), Msg("kickoff", flyer2)])
    assert "flyer1" in collected
    assert "flyer2" in collected

    collected.clear()
    RE([Msg("open_run"), Msg("kickoff", flyer2), Msg("kickoff", flyer1)])
    assert "flyer1" in collected
    assert "flyer2" in collected


@requires_ophyd
def test_flying_outside_a_run_is_illegal(RE, hw):
    flyer = hw.trivial_flyer

    # This is normal, legal usage.
    RE(
        [
            Msg("open_run"),
            Msg("kickoff", flyer, group="foo"),
            Msg("wait", group="foo"),
            Msg("complete", flyer, group="bar"),
            Msg("wait", group="bar"),
            Msg("collect", flyer),
            Msg("close_run"),
        ]
    )

    # This is normal, legal usage (partial collection).
    RE(
        [
            Msg("open_run"),
            Msg("kickoff", flyer, group="foo"),
            Msg("wait", group="foo"),
            Msg("collect", flyer),
            Msg("collect", flyer),
            Msg("collect", flyer),
            Msg("complete", flyer, group="bar"),
            Msg("wait", group="bar"),
            Msg("collect", flyer),
            Msg("collect", flyer),
            Msg("collect", flyer),
            Msg("close_run"),
        ]
    )

    # It is not legal to kickoff outside of a run.
    with pytest.raises(IllegalMessageSequence):
        RE([Msg("kickoff", flyer)])


@requires_ophyd
def test_flyer_descriptor(RE, hw):
    class Flyer(TrivialFlyer):
        def __init__(self, name):
            self.name = name
            self.detector = FlyerDetector(name="flyer-detector")

        def read_configuration(self):
            return self.detector.read_configuration()

        def describe_configuration(self):
            return self.detector.describe_configuration()

        def describe_collect(self):
            return {
                "primary": {
                    "data_key_1": {
                        "dims": [],
                        "dtype": "string",
                        "shape": [],
                        "source": "",
                    },
                    "data_key_2": {
                        "dims": [],
                        "dtype": "number",
                        "shape": [],
                        "source": "",
                    },
                },
                "secondary": {
                    "data_key_3": {
                        "dims": [],
                        "dtype": "string",
                        "shape": [],
                        "source": "",
                    },
                    "data_key_4": {
                        "dims": [],
                        "dtype": "number",
                        "shape": [],
                        "source": "",
                    },
                },
            }

        def collect(self):
            yield {
                "data": {"data_key_1": "1", "data_key_2": 2},
                "timestamps": {"data_key_1": 0, "data_key_2": 0},
                "time": 0,
            }

            yield {
                "data": {"data_key_3": "3", "data_key_4": 4},
                "timestamps": {"data_key_3": 0, "data_key_4": 0},
                "time": 0,
            }

    flyers = [Flyer(name="flyer"), TrivialFlyer()]
    descriptors = dict()

    RE(
        fly(flyers),
        {"descriptor": lambda name, doc: descriptors.update({doc["name"]: doc})},
    )

    primary_descriptor = descriptors["primary"]
    assert primary_descriptor["configuration"]["flyer"] == {
        "data": {
            "config_key_1": "1",
            "config_key_2": 2,
            "config_key_3": "3",
            "config_key_4": 4,
        },
        "timestamps": {
            "config_key_1": 1,
            "config_key_2": 2,
            "config_key_3": 3,
            "config_key_4": 4,
        },
        "data_keys": {
            "config_key_1": {"dtype": "string", "shape": [], "source": "PV:Config:1"},
            "config_key_2": {"dtype": "number", "shape": [], "source": "PV:Config:2"},
            "config_key_3": {"dtype": "string", "shape": [], "source": "PV:Config:3"},
            "config_key_4": {"dtype": "number", "shape": [], "source": "PV:Config:4"},
        },
    }

    assert "flyer" in primary_descriptor["object_keys"]

    secondary_descriptor = descriptors["secondary"]
    assert len(secondary_descriptor["configuration"]["flyer"]["data"]) == 4
    assert secondary_descriptor["configuration"] == primary_descriptor["configuration"]

    assert "flyer" in secondary_descriptor["object_keys"]

    trivial_flyer_descriptor = descriptors["stream_name"]
    print(f"trivial flyer descriptor: {trivial_flyer_descriptor}")
    assert len(trivial_flyer_descriptor["configuration"]) == 1
    assert "trivial_flyer" in trivial_flyer_descriptor["object_keys"]


@requires_ophyd
def test_device_flyer_descriptor(RE, hw):
    # TrivialFlyer is not a Device
    flyers = [FlyerDevice(name="flyer-detector"), TrivialFlyer()]
    descriptors = dict()

    RE(
        fly(flyers),
        {"descriptor": lambda name, doc: descriptors.update({doc["name"]: doc})},
    )

    primary_descriptor = descriptors["primary"]
    print(f"primary descriptor: {primary_descriptor}")
    assert len(primary_descriptor["configuration"]) == 1
    assert primary_descriptor["configuration"]["flyer-detector"] == {
        "data": {
            "config_key_1": "1",
            "config_key_2": 2,
            "config_key_3": "3",
            "config_key_4": 4,
        },
        "timestamps": {
            "config_key_1": 1,
            "config_key_2": 2,
            "config_key_3": 3,
            "config_key_4": 4,
        },
        "data_keys": {
            "config_key_1": {"dtype": "string", "shape": [], "source": "PV:Config:1"},
            "config_key_2": {"dtype": "number", "shape": [], "source": "PV:Config:2"},
            "config_key_3": {"dtype": "string", "shape": [], "source": "PV:Config:3"},
            "config_key_4": {"dtype": "number", "shape": [], "source": "PV:Config:4"},
        },
    }

    secondary_descriptor = descriptors["secondary"]
    print(f"secondary_descriptor: {secondary_descriptor}")
    assert len(secondary_descriptor["configuration"]["flyer-detector"]["data"]) == 4
    assert secondary_descriptor["configuration"] == primary_descriptor["configuration"]

    trivial_flyer_descriptor = descriptors["stream_name"]
    print(f"trivial flyer descriptor: {trivial_flyer_descriptor}")
    assert len(trivial_flyer_descriptor["configuration"]) == 1
    assert "trivial_flyer" in trivial_flyer_descriptor["object_keys"]


def test_device_redundent_config_reading(RE):
    flyer = FlyerDevice(name="flyer-detector")
    RE(
        [
            Msg("open_run"),
            Msg("kickoff", flyer, group="foo"),
            Msg("wait", group="foo"),
            Msg("collect", flyer),
            Msg("collect", flyer),
            Msg("collect", flyer),
            Msg("complete", flyer, group="bar"),
            Msg("wait", group="bar"),
            Msg("collect", flyer),
            Msg("close_run"),
        ]
    )
    assert flyer.call_counts["collect"] == 4
    assert flyer.call_counts["describe_collect"] == 1
    assert flyer.call_counts["read_configuration"] == 1
    assert flyer.call_counts["describe_configuration"] == 1
    assert flyer.call_counts["kickoff"] == 1
    assert flyer.call_counts["complete"] == 1


class FlyerDetector(Device):
    def __init__(self, name, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def describe_configuration(self):
        return {
            "config_key_1": {"dtype": "string", "shape": [], "source": "PV:Config:1"},
            "config_key_2": {"dtype": "number", "shape": [], "source": "PV:Config:2"},
            "config_key_3": {"dtype": "string", "shape": [], "source": "PV:Config:3"},
            "config_key_4": {"dtype": "number", "shape": [], "source": "PV:Config:4"},
        }

    def read_configuration(self):
        return {
            "config_key_1": {"value": "1", "timestamp": 1},
            "config_key_2": {"value": 2, "timestamp": 2},
            "config_key_3": {"value": "3", "timestamp": 3},
            "config_key_4": {"value": 4, "timestamp": 4},
        }


def call_counter(func):
    @functools.wraps(func)
    def inner(self, *args, **kwargs):
        self.call_counts[func.__name__] += 1
        return func(self, *args, **kwargs)

    return inner


class FlyerDevice(Device):
    detector = Cpt(FlyerDetector, name="flyer-detector")

    def __init__(self, *args, **kwargs):
        self.call_counts = defaultdict(int)
        super().__init__(*args, **kwargs)

    @call_counter
    def kickoff(self):
        return NullStatus()

    @call_counter
    def complete(self):
        return NullStatus()

    def stop(self, *, success=False):
        pass

    @call_counter
    def describe_collect(self):
        return {
            "primary": {
                "data_key_1": {
                    "dims": [],
                    "dtype": "string",
                    "shape": [],
                    "source": "",
                },
                "data_key_2": {
                    "dims": [],
                    "dtype": "number",
                    "shape": [],
                    "source": "",
                },
            },
            "secondary": {
                "data_key_3": {
                    "dims": [],
                    "dtype": "string",
                    "shape": [],
                    "source": "",
                },
                "data_key_4": {
                    "dims": [],
                    "dtype": "number",
                    "shape": [],
                    "source": "",
                },
            },
        }

    @call_counter
    def collect(self):
        yield {
            "data": {"data_key_1": "", "data_key_2": 0},
            "timestamps": {"data_key_1": 0, "data_key_2": 0},
            "time": 0,
        }

        yield {
            "data": {"data_key_3": "", "data_key_4": 0},
            "timestamps": {"data_key_3": 0, "data_key_4": 0},
            "time": 0,
        }

    @call_counter
    def read_configuration(self):
        return super().read_configuration()

    @call_counter
    def describe_configuration(self):
        return super().describe_configuration()


def test_describe_config_optional(RE):
    class Simple:
        """A trivial flyer that omits the configuration methods"""

        name = "simple_flyer"
        parent = None

        def kickoff(self):
            return NullStatus()

        def describe_collect(self):
            return {"stream_name": {}}

        def complete(self):
            return NullStatus()

        def collect(self):
            for i in range(100):
                yield {"data": {}, "timestamps": {}, "time": i, "seq_num": i}

        def stop(self, *, success=False):
            pass

    d = DocCollector()

    flyer = Simple()

    RE(
        [
            Msg("open_run"),
            Msg("kickoff", flyer, group="foo"),
            Msg("wait", group="foo"),
            Msg("complete", flyer, group="bar"),
            Msg("wait", group="bar"),
            Msg("collect", flyer),
            Msg("close_run"),
        ],
        d.insert,
    )

    ((desc,),) = d.descriptor.values()
    assert desc["name"] == "stream_name"

    assert "simple_flyer" in desc["object_keys"]
    assert "simple_flyer" in desc["configuration"]
