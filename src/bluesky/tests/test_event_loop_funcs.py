import asyncio

import pytest

import bluesky.plan_stubs as bps
from bluesky.run_engine import autoawait_in_bluesky_event_loop, call_in_bluesky_event_loop, in_bluesky_event_loop


def test_in_bluesky_event_loop(RE):
    def plan():
        yield from bps.null()
        assert in_bluesky_event_loop()

    assert not in_bluesky_event_loop()
    RE(plan())
    assert not in_bluesky_event_loop()


def test_call_in_bluesky_event_loop(RE):
    event_loop = None

    async def check():
        nonlocal event_loop
        event_loop = asyncio.get_running_loop()

    asyncio.run(check())
    assert event_loop and event_loop != RE._loop

    call_in_bluesky_event_loop(check())
    assert event_loop == RE._loop


def test_autoawait_in_bluesky_event_loop(RE, request):
    event_loop = None

    async def check():
        nonlocal event_loop
        event_loop = asyncio.get_running_loop()

    IPython = pytest.importorskip("IPython")

    # IPython's default autoawait machinery creates its own asyncio event loop
    # (cached in ``IPython.core.async_helpers``).  Close it on teardown so it is
    # not leaked (ResourceWarning: unclosed event loop / socket) at shutdown.
    def close_ipython_loop():
        from IPython.core.async_helpers import get_asyncio_loop

        loop = get_asyncio_loop()
        if loop is not None and not loop.is_closed():
            loop.close()

    request.addfinalizer(close_ipython_loop)

    ip = IPython.core.interactiveshell.InteractiveShell(user_ns=locals())
    # Check that without the autoawait we get the wrong event loop
    ip.run_cell("await check()")
    assert event_loop and event_loop != RE._loop
    # Then with it we get the right one
    autoawait_in_bluesky_event_loop(ip)
    ip.run_cell("await check()")
    assert event_loop == RE._loop
