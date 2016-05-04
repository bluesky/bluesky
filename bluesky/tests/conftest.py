import asyncio
from bluesky.run_engine import RunEngine
import pytest


@pytest.fixture(scope='function')
def fresh_RE(request):
    loop = asyncio.new_event_loop()
    loop.set_debug(True)
    return RunEngine({}, loop=loop)
