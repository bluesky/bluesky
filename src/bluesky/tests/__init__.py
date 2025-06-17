import os
from types import ModuleType
from typing import Optional

import pytest

# some module level globals.
ophyd: Optional[ModuleType]
ophyd = None
ophyd_reason = ""
ophyd_async: Optional[ModuleType]
ophyd_async = None
ophyd_async_reason = ""

try:
    import ophyd  # type: ignore
except ImportError as ie:
    # pytestmark = pytest.mark.skip
    ophyd = None
    ophyd_reason = str(ie)

# define a skip condition based on if ophyd is available or not
requires_ophyd = pytest.mark.skipif(ophyd is None, reason=ophyd_reason)

try:  # type: ignore
    import ophyd_async
except ImportError as ie:
    # pytestmark = pytest.mark.skip
    ophyd_async = None
    ophyd_async_reason = str(ie)

requires_ophyd_async = pytest.mark.skipif(ophyd_async is None, reason=ophyd_async_reason)

uses_os_kill_sigint = pytest.mark.skipif(
    os.name == "nt", reason="os.kill on windows ignores signal argument and kills the entire process."
)
