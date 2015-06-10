from bluesky.examples import *
from bluesky.standard_config import RE
from matplotlib import pyplot as plt
from xray_vision.backend.mpl.cross_section_2d import CrossSection
import numpy as np
import filestore.api as fsapi
import time as ttime

from filestore.handlers import NpyHandler
fsapi.register_handler('npy', NpyHandler)

class ImageCallback(CallbackBase):

    def __init__(self, datakey, fig=None):
        # wheeee MRO
        super(ImageCallback, self).__init__()
        if fig is None:
            fig = plt.figure()
        self.cs = CrossSection(fig)
        self.datakey = datakey
        show_all_figures()

    def event(self, doc):
        uid = doc['data'][self.datakey]
        data = fsapi.retrieve(uid)
        self.cs.update_image(data)
        draw_all_figures()


def show_all_figures():
    for f_mgr in plt._pylab_helpers.Gcf.get_all_fig_managers():
        f_mgr.canvas.figure.show()


def draw_all_figures():
    for f_mgr in plt._pylab_helpers.Gcf.get_all_fig_managers():
        f_mgr.canvas.draw()
        f_mgr.canvas.flush_events()


def stepscan(motor, det):
    for i in np.linspace(-5, 5, 75):
        yield Msg('create')
        yield Msg('set', motor, i)
        yield Msg('trigger', det)
        yield Msg('read', motor)
        yield Msg('read', det)
        yield Msg('save')


ic = ImageCallback('det_2d')
table_callback = LiveTable(fields=[motor._name, det_2d._name])
RE(stepscan(motor, det_2d), subs={'event': ic, 'all': table_callback}, beamline_id='c08i')
