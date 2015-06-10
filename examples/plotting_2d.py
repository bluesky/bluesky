from bluesky.examples import *
from bluesky.standard_config import RE
from matplotlib import pyplot as plt
from xray_vision.backend.mpl.cross_section_2d import CrossSection
import numpy as np
import filestore.api as fsapi
import time as ttime

from filestore.handlers import NpyHandler
fsapi.register_handler('npy', NpyHandler)


def stepscan(motor, det):
    for i in np.linspace(-5, 5, 75):
        yield Msg('create')
        yield Msg('set', motor, i)
        yield Msg('trigger', det)
        yield Msg('read', motor)
        yield Msg('read', det)
        yield Msg('save')


ic = LiveImage('det_2d')
table_callback = LiveTable(fields=[motor._name, det_2d._name])
RE(stepscan(motor, det_2d), subs={'event': ic, 'all': table_callback}, beamline_id='c08i')
