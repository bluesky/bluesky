import pytest
# some module level globals.
ophyd = None
reason = ''

try:
    import ophyd
    from ophyd import setup_ophyd
except ImportError as ie:
    # pytestmark = pytest.mark.skip
    ophyd = None
    reason = str(ie)
else:
    setup_ophyd()
    # define the classes only if ophyd is available

# define a skip condition based on if ophyd is available or not
requires_ophyd = pytest.mark.skipif(ophyd is None, reason=reason)
