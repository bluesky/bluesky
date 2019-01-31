import bluesky.callbacks.core
from bluesky.callbacks.text import TextTableFactory, heading_printer, BaselinePrinterFactory, new_stream_printer
from bluesky.callbacks.mpl_plotting import FigureManager
rr = bluesky.callbacks.core.RunRouter([heading_printer, new_stream_printer, BaselinePrinterFactory(), TextTableFactory('primary'), FigureManager()])
from bluesky import RunEngine, SupplementalData
RE = RunEngine({})
RE.subscribe(rr)
from bluesky.plans import scan
from ophyd.sim import det, motor
det.kind = 'hinted'
sd = SupplementalData(baseline=[det])
RE.preprocessors.append(sd)
RE(scan([det], motor, -1, 1, 10))
