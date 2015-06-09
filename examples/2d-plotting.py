from bluesky.examples import *
from bluesky.standard_config import RE
from matplotlib import pyplot as plt
import matplotlib
from xray_vision.backend.mpl.cross_section_2d import CrossSection
import numpy as np


class ImageCallback(CallbackBase):

    def __init__(self, datakey):
        # wheeee MRO
        super(ImageCallback, self).__init__()
        self.cs = CrossSection(plt.figure())
        self.datakey = datakey

    def event(self, doc):
        data = doc['data'][self.datakey]
        self.cs.update_image(data)


ic = ImageCallback('det_2d')
table_callback = LiveTable(fields=[motor._name, det_2d._name])

RE(stepscan(motor, det_2d), subs={'event': ic, 'all': table_callback}, beamline_id='c08i')
