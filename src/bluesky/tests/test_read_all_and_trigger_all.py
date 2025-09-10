import asyncio
import time
from collections.abc import Coroutine
from typing import Any

import pytest

import bluesky.plan_stubs as bps
from bluesky.protocols import Reading

from . import requires_ophyd, requires_ophyd_async

SIM_SLEEP_TIME = 0.05


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
        trigger_time: float | None = None
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
        trigger_time: float | None = None

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
        timestamps = {k: v % 60 for k, v in event["timestamps"].items()}
        assert len(timestamps) == 8

        async_timestamps = [v for k, v in timestamps.items() if k.startswith("async_device")]
        sync_timestamps = [v for k, v in timestamps.items() if k.startswith("sync_device")]

        first_sync_device_timestamps = min(sync_timestamps)
        for v in async_timestamps:
            assert v < first_sync_device_timestamps  # async devices are ran first

        for i in range(1, len(async_timestamps)):
            assert async_timestamps[i] - async_timestamps[0] == pytest.approx(0, abs=0.01)

        for i in range(len(sync_timestamps) - 1):
            assert sync_timestamps[i + 1] - sync_timestamps[i] == pytest.approx(SIM_SLEEP_TIME, abs=0.01)

    assert None not in (
        async_device1.trigger_time,
        async_device2.trigger_time,
        sync_device1.trigger_time,
        sync_device2.trigger_time,
    )

    assert async_device2.trigger_time - async_device1.trigger_time == pytest.approx(0, abs=0.01)
    assert sync_device2.trigger_time - sync_device1.trigger_time == pytest.approx(SIM_SLEEP_TIME, abs=0.01)
