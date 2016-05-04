from bluesky.run_engine import RunEngine
import pytest


@pytest.fixture(scope='function')
def fresh_RE(request):
    return RunEngine({})
