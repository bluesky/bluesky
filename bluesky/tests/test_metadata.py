import bluesky
import ophyd


def test_blueskyversion(RE):
    assert RE.md['versions'].get('bluesky') == bluesky.__version__


def test_ophydversion(RE):
    assert RE.md['versions'].get('ophyd') == ophyd.__version__
