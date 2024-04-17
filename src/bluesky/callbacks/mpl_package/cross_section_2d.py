# ######################################################################
# Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
#                                                                      #
# Redistribution and use in source and binary forms, with or without   #
# modification, are permitted provided that the following conditions   #
# are met:                                                             #
#                                                                      #
# * Redistributions of source code must retain the above copyright     #
#   notice, this list of conditions and the following disclaimer.      #
#                                                                      #
# * Redistributions in binary form must reproduce the above copyright  #
#   notice this list of conditions and the following disclaimer in     #
#   the documentation and/or other materials provided with the         #
#   distribution.                                                      #
#                                                                      #
# * Neither the name of the Brookhaven Science Associates, Brookhaven  #
#   National Laboratory nor the names of its contributors may be used  #
#   to endorse or promote products derived from this software without  #
#   specific prior written permission.                                 #
#                                                                      #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS  #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT    #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS    #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE       #
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,           #
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES   #
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR   #
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)   #
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,  #
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OTHERWISE) ARISING   #
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE   #
# POSSIBILITY OF SUCH DAMAGE.                                          #
########################################################################

import logging

from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure

from bluesky.callbacks.mpl_package.AbstractDataView import AbstractMPLDataView
from bluesky.callbacks.mpl_package.AbstractDataView1d import AbstractDataView2D
from bluesky.callbacks.mpl_package.cross_section import CrossSection
from bluesky.callbacks.mpl_package.interpolation import InterpolationEnum

logger = logging.getLogger(__name__)


class CrossSection2DView(AbstractDataView2D, AbstractMPLDataView):
    """
    CrossSection2DView docstring

    """

    # list of valid options for the interpolation parameter. The first one is
    # the default value.
    interpolation = [e.value for e in InterpolationEnum]

    def __init__(
        self,
        base_figure: Figure,
        data_list,
        key_list,
        cmap: str | Colormap | None = None,
        normalization_function: Normalize | None = None,
        limit_func=None,
        interpolation=None,
        **kwargs,
    ):
        """
        Sets up figure with cross section viewer

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure object to build the class on, will clear
            current contents
        clim_percentile : float or None
           percentile away from 0, 100 to put the max/min limits at
           ie, clim_percentile=5 -> vmin=5th percentile vmax=95th percentile
        interpolation : str, optional
            Interpolation method to use. List of valid options can be found in
            CrossSection2DView.interpolation
        """
        # call up the inheritance chain
        super().__init__(
            fig=base_figure, data_list=data_list, ordered_key_list=key_list, norm=normalization_function, cmap=cmap
        )
        self._cross_section = CrossSection(
            base_figure, colormap=self._cmap, norm=self._norm, limit_func=limit_func, interpolation=interpolation
        )

    def update_color_map(self, cmap):
        self._cross_section.update_cmap(cmap)

    def update_image(self, img_idx):
        self._cross_section.update_image(self._data_dict[self._key_list[img_idx]])

    def replot(self):
        """
        Update the image displayed by the main axes

        Parameters
        ----------
        new_image : 2D ndarray
           The new image to use
        """
        self._cross_section._update_artists()

    def update_normalization(self, new_norm):
        """
        Update the way that matplotlib normalizes the image. Default is linear
        """
        self._cross_section.update_norm(new_norm)

    def set_limit_func(self, limit_func):
        """
        Set the function to use to determine the color scale

        """
        self._cross_section.update_limit_func(limit_func)

    def update_interpolation(self, interpolation: InterpolationEnum):
        """
        Update the way that matplotlib interpolates the image. Default is none

        Parameters
        ----------
        interpolation : str
            Interpolation method to use. List of valid options can be found in
            CrossSection2DView.interpolation
        """
        self._cross_section.update_interpolation(interpolation)
