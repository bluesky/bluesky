import logging
from types import SimpleNamespace
logger = logging.getLogger(__name__)

from .utils import Msg
from .utils import RunEngineInterrupted
from .utils import IllegalMessageSequence
from .utils import FailedStatus

from .run_engine import RunEngine
from .plans import SupplementalData

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
from . import plans

plns = SimpleNamespace(
    count=plans.count,
    scan=plans.scan,
    inner_product_scan=plans.inner_product_scan,
    grid_scan=plans.grid_scan,
    scan_nd=plans.scan_nd,
    list_scan=plans.list_scan,
    log_scan=plans.log_scan,
    adaptive_scan=plans.adaptive_scan,
    tune_centroid=plans.tune_centroid,
    spiral=plans.spiral,
    spiral_fermat=plans.spiral_fermat,
    ramp_plan=plans.ramp_plan)

rplns = SimpleNamespace(
    rel_scan=plans.rel_scan,
    relative_inner_product_scan=plans.relative_inner_product_scan,
    rel_grid_scan=plans.rel_grid_scan,
    rel_list_scan=plans.rel_list_scan,
    rel_log_scan=plans.rel_log_scan,
    rel_adaptive_scan=plans.rel_adaptive_scan,
    rel_spiral=plans.rel_spiral,
    rel_spiral_fermat=plans.rel_spiral_fermat)

__version__ = get_versions()['version']
del get_versions
del SimpleNamespace
del logging
