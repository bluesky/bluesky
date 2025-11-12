import asyncio
import time
from collections.abc import Coroutine
from typing import Any, Optional
from unittest.mock import ANY

import pytest

import bluesky.plan_stubs as bps
from bluesky.protocols import Reading

from . import requires_ophyd, requires_ophyd_async

SIM_SLEEP_TIME = 0.1


@pytest.fixture()
def sync_and_async_devices():
    import ophyd
    import ophyd_async

    class SlowSyncSignal(ophyd.Signal):
        def read(self, *args, **kwargs):
            time.sleep(SIM_SLEEP_TIME)
            self._metadata["timestamp"] = time.time()
            reading = super().read(*args, **kwargs)
            return reading

    class SyncReadableDevice(ophyd.Device):
        trigger_time: Optional[float] = None
        signal1 = ophyd.Component(SlowSyncSignal, value=1)
        signal2 = ophyd.Component(SlowSyncSignal, value="some_value")

        def trigger(self) -> ophyd.DeviceStatus:
            time.sleep(SIM_SLEEP_TIME)
            self.trigger_time = time.time()
            return super().trigger()

    class SlowAsyncSoftSignalBackend(ophyd_async.core.SoftSignalBackend):
        async def get_reading(self) -> Coroutine[Any, Any, Reading]:  # type: ignore
            await asyncio.sleep(SIM_SLEEP_TIME)
            reading = await super().get_reading()
            reading["timestamp"] = time.time()
            return reading  # type: ignore

    class AsyncReadableDevice(ophyd_async.core.StandardReadable):
        trigger_time: Optional[float] = None

        def __init__(self, name: str = "") -> None:
            super().__init__(name=name)
            with self.add_children_as_readables():
                self.signal1 = ophyd_async.core.SignalRW(SlowAsyncSoftSignalBackend(int, initial_value=1))
                self.signal2 = ophyd_async.core.SignalRW(
                    SlowAsyncSoftSignalBackend(str, initial_value="some_value")
                )

        @ophyd_async.core.AsyncStatus.wrap
        async def trigger(self):
            await asyncio.sleep(SIM_SLEEP_TIME)
            self.trigger_time = time.time()

    return (
        SyncReadableDevice(name="sync_device1"),
        SyncReadableDevice(name="sync_device2"),
        AsyncReadableDevice(name="async_device1"),
        AsyncReadableDevice(name="async_device2"),
    )


@requires_ophyd
@requires_ophyd_async
def test_read_all(RE, sync_and_async_devices):
    output = {"start": [], "descriptor": [], "event": [], "stop": []}

    def plan():
        yield from bps.open_run()
        yield from bps.create(name="primary")
        ret = yield from bps.read_all(sync_and_async_devices)
        assert ret == {
            "async_device1-signal1": {"alarm_severity": 0, "timestamp": ANY, "value": 1},
            "async_device1-signal2": {"alarm_severity": 0, "timestamp": ANY, "value": "some_value"},
            "async_device2-signal1": {"alarm_severity": 0, "timestamp": ANY, "value": 1},
            "async_device2-signal2": {"alarm_severity": 0, "timestamp": ANY, "value": "some_value"},
            "sync_device1_signal1": {"timestamp": ANY, "value": 1},
            "sync_device1_signal2": {"timestamp": ANY, "value": "some_value"},
            "sync_device2_signal1": {"timestamp": ANY, "value": 1},
            "sync_device2_signal2": {"timestamp": ANY, "value": "some_value"},
        }
        yield from bps.save()
        yield from bps.close_run()

    RE(plan(), lambda name, doc: output[name].append(doc))

    assert output["event"] == [
        {
            "data": {
                "async_device1-signal1": 1,
                "async_device1-signal2": "some_value",
                "async_device2-signal1": 1,
                "async_device2-signal2": "some_value",
                "sync_device1_signal1": 1,
                "sync_device1_signal2": "some_value",
                "sync_device2_signal1": 1,
                "sync_device2_signal2": "some_value",
            },
            "descriptor": ANY,
            "filled": {},
            "seq_num": 1,
            "time": ANY,
            "timestamps": {
                "async_device1-signal1": ANY,
                "async_device1-signal2": ANY,
                "async_device2-signal1": ANY,
                "async_device2-signal2": ANY,
                "sync_device1_signal1": ANY,
                "sync_device1_signal2": ANY,
                "sync_device2_signal1": ANY,
                "sync_device2_signal2": ANY,
            },
            "uid": ANY,
        }
    ]


@requires_ophyd
@requires_ophyd_async
def test_trigger_all(RE, sync_and_async_devices):
    sync_device1, sync_device2, async_device1, async_device2 = sync_and_async_devices

    def plan():
        yield from bps.open_run()
        yield from bps.trigger_all(sync_and_async_devices, wait=True)
        yield from bps.close_run()

    assert (
        async_device1.trigger_time,
        async_device2.trigger_time,
        sync_device1.trigger_time,
        sync_device2.trigger_time,
    ) == (None, None, None, None)

    RE(plan())

    assert async_device2.trigger_time - async_device1.trigger_time == pytest.approx(0, abs=0.01)
    assert sync_device2.trigger_time - sync_device1.trigger_time == pytest.approx(SIM_SLEEP_TIME, abs=0.03)


@requires_ophyd
@requires_ophyd_async
def test_one_shot_works_asynchronously(RE, sync_and_async_devices):
    sync_device1, sync_device2, async_device1, async_device2 = sync_and_async_devices

    output = {"start": [], "descriptor": [], "event": [], "stop": []}

    def plan():
        yield from bps.open_run()
        yield from bps.one_shot([sync_device1, async_device1, sync_device2, async_device2])
        yield from bps.one_shot([sync_device1, async_device1, sync_device2, async_device2])
        yield from bps.close_run()

    RE(plan(), lambda name, doc: output[name].append(doc))

    assert len(output["event"]) == 2
    for event in output["event"]:
        timestamps = event["timestamps"]
        assert len(timestamps) == 8

        async_timestamps = [v for k, v in timestamps.items() if k.startswith("async_device")]
        sync_timestamps = [v for k, v in timestamps.items() if k.startswith("sync_device")]

        first_sync_device_timestamps = min(sync_timestamps)
        for v in async_timestamps:
            assert v < first_sync_device_timestamps  # async devices are ran first

        for i in range(1, len(async_timestamps)):
            assert async_timestamps[i] - async_timestamps[0] == pytest.approx(0, abs=0.03)

        for i in range(len(sync_timestamps) - 1):
            assert sync_timestamps[i + 1] - sync_timestamps[i] == pytest.approx(SIM_SLEEP_TIME, abs=0.03)


@requires_ophyd
@requires_ophyd_async
def test_read_all_flattened_structure(RE, sync_and_async_devices):
    sync_device1, sync_device2, async_device1, async_device2 = sync_and_async_devices

    output = {"start": [], "descriptor": [], "event": [], "stop": []}

    def plan():
        yield from bps.open_run()
        yield from bps.create(name="primary")
        ret = yield from bps.read_all([sync_device1, async_device1, sync_device2.signal1, async_device2.signal2])
        assert set(ret) == {
            "async_device1-signal1",
            "async_device1-signal2",
            "async_device2-signal2",
            "sync_device1_signal1",
            "sync_device1_signal2",
            "sync_device2_signal1",
        }
        yield from bps.save()
        yield from bps.close_run()

    RE(plan(), lambda name, doc: output[name].append(doc))
