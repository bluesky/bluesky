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
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import numpy as np

from .. import QtCore, QtGui
from . import AbstractMPLDataView
from .. import AbstractDataView2D

import logging
logger = logging.getLogger(__name__)


class ContourView(AbstractDataView2D, AbstractMPLDataView):
    """
    The ContourView provides a UI widget for viewing a number of 1-D
    data sets as a contour plot, starting from dataset 0 at y = 0
    """

    def __init__(self, fig, data_list=None, cmap=None, norm=None, *args,
                 **kwargs):
        """
        __init__ docstring

        Parameters
        ----------
        fig : figure to draw the artists on
        x_data : list
            list of vectors of x-coordinates
        y_data : list
            list of vectors of y-coordinates
        lbls : list
            list of the names of each data set
        cmap : colormap that matplotlib understands
        norm : mpl.colors.Normalize
        """
        # set some defaults
        # no defaults yet

        # call the parent constructors
        super(ContourView, self).__init__(data_list=data_list, fig=fig,
                                          cmap=cmap, norm=norm, *args,
                                          **kwargs)

        # create the matplotlib axes
        self._ax = self._fig.add_subplot(1, 1, 1)
        self._ax.set_aspect('equal')

        # plot the data
        self.replot()

    def replot(self):
        """
        Override
        Replot the data after modifying a display parameter (e.g.,
        offset or autoscaling) or adding new data
        """
        # TODO: This class was originally written to convert a 1-D stack into a
        # 2-D contour.  Rewrite this replot method

        # get the keys from the dict
        keys = list(six.iterkeys(self._data))
        # number of datasets in the data dict
        num_keys = len(keys)
        # cannot plot data if there are no keys
        if num_keys < 1:
            return
        # set the local counter
        counter = num_keys - 1
        # @tacaswell Should it be required that all datasets are the same
        # length?
        num_coords = len(self._data[keys[0]][0])
        # declare the array
        self._data_arr = np.zeros((num_keys, num_coords))
        # add the data to the main axes
        for key in self._data.keys():
            # get the (x,y) data from the dictionary
            (x, y) = self._data[key]
            # add the data to the array

            self._data_arr[counter] = y
            # decrement the counter
            counter -= 1
        # get the first dataset to get the x axis and number of y datasets
        x, y = self._data[keys[0]]
        y = np.arange(len(keys))
        # TODO: Colormap initialization is not working properly.
        self._ax.contourf(x, y, self._data_arr)  # , cmap=colors.Colormap(self._cmap))
