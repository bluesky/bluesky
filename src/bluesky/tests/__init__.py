import os
from types import ModuleType
from typing import Optional

import pytest

# some module level globals.
ophyd: Optional[ModuleType]
ophyd = None
reason = ""

try:
    import ophyd  # type: ignore
except ImportError as ie:
    # pytestmark = pytest.mark.skip
    ophyd = None
    reason = str(ie)

# define a skip condition based on if ophyd is available or not
requires_ophyd = pytest.mark.skipif(ophyd is None, reason=reason)

uses_os_kill_sigint = pytest.mark.skipif(
    os.name == "nt", reason="os.kill on windows ignores signal argument and kills the entire process."
)
