import bluesky
import ophyd
from bluesky import RunEngine

def test_blueskyversion():
    re = RunEngine()
    print(re.md.get('BLUESKY_VERSION'))
    assert re.md.get('BLUESKY_VERSION') == bluesky.__version__

def test_ophydversion():
    re = RunEngine()
    print(re.md.get('OPHYD_VERSION'))
    assert re.md.get('OPHYD_VERSION') == ophyd.__version__

