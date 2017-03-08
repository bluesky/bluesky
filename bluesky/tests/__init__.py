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
    try:
        setup_ophyd()
        # define the classes only if ophyd is available
    except Exception as E:
        ophyd = None
        reason = str(E)

# define a skip condition based on if ophyd is available or not
requires_ophyd = pytest.mark.skipif(ophyd is None, reason=reason)
