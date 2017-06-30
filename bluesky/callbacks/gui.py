from matplotlib.widgets import Slider
from databroker_browser.qt._cross_section_2d import CrossSection
import numpy as np
import matplotlib.pyplot as plt


class StackViewer(object):
    """
    Parameters
    ----------
    viewer : object
        expected to have update_image method and fig attribute
    images : array-like
        must support integer indexing and return a 2D array
    """

    def __init__(self, viewer, images=None, update_upon_add=True):
        self.update_upon_add = update_upon_add
        if images is None:
            images = []
        self.viewer = viewer
        self.images = images
        self.slider_ax = None
        self.slider = None
        length = len(self.images)
        fig = self.viewer._fig
        if length > 0:
            self.create_slider(length)
        fig.show()

    def update(self, val):
        if not isinstance(val, int):
            self.slider.set_val(int(round(val)))
            # sends up through 'update' again
        self.viewer.update_image(self.images[int(val)])

    def create_slider(self, length):
        if self.slider_ax:
            self.slider_ax.remove()
        self.slider_ax = fig.add_axes([0.1, 0.01, 0.8, 0.02])
        self.slider = Slider(self.slider_ax, 'Frame', 0, length - 1, 0,
                             valfmt='%d/{}'.format(length - 1))
        self.slider.on_changed(self.update)

    def add_image(self, img):
        self.images.append(img)
        self.create_slider(len(self.images))
        if self.update_upon_add:
            self.update(len(self.images) - 1)
            self.slider.set_val(len(self.images) - 1)

if __name__ == '__main__':
    fig = plt.figure()
    imgs = (np.random.random((10, 10)) for i in range(10))
    v = CrossSection(fig, cmap='viridis')
    a = StackViewer(v)
    input()
    for img in imgs:
        a.add_image(img)
        input()