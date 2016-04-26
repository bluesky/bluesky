from bluesky.examples import *
from bluesky.broker_examples import *
from bluesky.callbacks.broker import LiveImage
from bluesky.tests.utils import setup_test_run_engine
from matplotlib import pyplot as plt
from xray_vision.backend.mpl.cross_section_2d import CrossSection
import numpy as np
import filestore.api as fsapi
import time as ttime

from filestore.handlers import NpyHandler
fsapi.register_handler('npy', NpyHandler)


def stepscan(det, motor):
    for i in np.linspace(-5, 5, 75):
        yield Msg('open_run')
        yield Msg('create')
        yield Msg('set', motor, i)
        yield Msg('trigger', det)
        yield Msg('read', motor)
        yield Msg('read', det)
        yield Msg('save')
        yield Msg('close_run')


ic = LiveImage('det_2d')
table_callback = LiveTable(fields=[motor.name, det_2d.name])
RE = setup_test_run_engine()
RE(stepscan(det_2d, motor), subs={'event': ic, 'all': table_callback}, beamline_id='c08i')
