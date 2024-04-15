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
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

from matplotlib import cm

from ...backend import AbstractDataView
from .. import QtCore, QtGui

logger = logging.getLogger(__name__)


class AbstractMPLDataView(object):
    """
    Class docstring
    """
    _default_cmap = 'gray'
    _default_norm = cm.colors.Normalize(vmin=0, vmax=1)

    def __init__(self, fig, cmap=None, norm=None, *args, **kwargs):
        """
        Docstring

        Parameters
        ----------
        fig : mpl.Figure
        """
        # call up the inheritance chain
        super(AbstractMPLDataView, self).__init__(*args, **kwargs)

        # set some defaults
        if cmap is None:
            cmap = self._default_cmap
        if norm is None:
            norm = self._default_norm

        # stash the parameters not taken care of by the inheritance chain
        self._cmap = cmap
        self._norm = norm
        self._fig = fig

        # clean the figure
        self._fig.clf()

    def replot(self):
        raise NotImplementedError("This method must be implemented by "
                                  "daughter classes")

    def update_cmap(self, cmap):
        """
        Update the color map used to display the image
        Parameters
        ----------
        cmap : mpl.cm.colors.Colormap
        """
        self._cmap = cmap

    def update_norm(self, norm):
        """
        Updates the normalization function used for the color mapping

        Parameters
        ----------
        norm : mpl.cm.colors.Normalize
        """
        self._norm = norm

    def draw(self):
        self._fig.canvas.draw()