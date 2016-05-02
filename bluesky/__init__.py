from collections import namedtuple

import logging
logger = logging.getLogger(__name__)


class Msg(namedtuple('Msg_base', ['command', 'obj', 'args', 'kwargs'])):
    __slots__ = ()

    def __new__(cls, command, obj=None, *args, **kwargs):
        return super(Msg, cls).__new__(cls, command, obj, args, kwargs)

    def __repr__(self):
        return '{}: ({}), {}, {}'.format(
            self.command, self.obj, self.args, self.kwargs)

from .run_engine import RunEngine
from .run_engine import RunEngineInterrupted
from .run_engine import IllegalMessageSequence
from .run_engine import FailedStatus

# for back-compat
from .plans import PlanBase
from .plans import Count
from .plans import AbsListScanPlan
from .plans import DeltaListScanPlan
from .plans import AbsScanPlan
from .plans import LogAbsScanPlan
from .plans import DeltaScanPlan
from .plans import LogDeltaScanPlan
from .plans import AdaptiveAbsScanPlan
from .plans import AdaptiveDeltaScanPlan
from .plans import PlanND
from .plans import InnerProductAbsScanPlan
from .plans import InnerProductDeltaScanPlan
from .plans import OuterProductAbsScanPlan
from .plans import OuterProductDeltaScanPlan
from .plans import Tweak

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
