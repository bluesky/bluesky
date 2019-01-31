import bluesky.callbacks.core
from bluesky.callbacks.text import TextTableFactory, heading_printer, BaselinePrinterFactory, new_stream_printer
rr = bluesky.callbacks.core.RunRouter([heading_printer, new_stream_printer, BaselinePrinterFactory(), TextTableFactory('primary'),])
from bluesky import RunEngine, SupplementalData
RE = RunEngine({})
RE.subscribe(rr)
from bluesky.plans import count
from ophyd.sim import det
det.kind = 'hinted'
sd = SupplementalData(baseline=[det])
RE.preprocessors.append(sd)
RE(count([det]))
