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
from collections import defaultdict

import numpy as np
from matplotlib import cm
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


class AbstractMPLDataView:
    _default_cmap = "gray"
    _default_norm = cm.colors.Normalize(vmin=0, vmax=1)

    def __init__(self, fig: Figure, cmap=None, norm=None, *args, **kwargs):
        # call up the inheritance chain
        super().__init__(*args, **kwargs)

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
        raise NotImplementedError("This method must be implemented by " "daughter classes")

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


class AbstractDataView:
    """
    AbstractDataView class docstring.  Defaults to a single matplotlib axes
    """

    default_dict_type = defaultdict
    default_list_type = list

    def __init__(self, data_list, key_list, *args, **kwargs):
        """
        Parameters
        ----------
        data_list : list
            The data stored as a list
        key_list : list
            The order of keys to plot
        """
        super().__init__(*args, **kwargs)

        if len(data_list) != len(key_list):
            raise ValueError(f"lengths of data ({len(data_list)}) and keys ({len(key_list)}) must be the" " same")
        if data_list is None:
            raise ValueError("data_list cannot have a value of None. It must " "be, at minimum, an empty list")
        if key_list is None:
            raise ValueError("key_list cannot have a value of None. It must " "be, at minimum, an empty list")
        # init the data dictionary
        data_dict = self.default_dict_type()
        if len(data_list) > 0:
            # but only give it values if the data_list has any entries
            for k, v in zip(key_list, data_list):
                data_dict[k] = v

        # stash the dict and keys
        self._data_dict = data_dict
        self._key_list = key_list

    def replot(self):
        """
        Do nothing in the abstract base class. Needs to be implemented
        in the concrete classes
        """
        raise NotImplementedError("Must override the replot() method in " "the concrete base class")

    def clear_data(self):
        """
        Clear all data
        """
        self._data_dict.clear()
        self._key_list[:] = []

    def remove_data(self, lbl_list):
        """
        Remove the key:value pair from the dictionary as specified by the
        labels in lbl_list

        Parameters
        ----------
        lbl_list : list
            String
            name(s) of dataset to remove
        """
        for lbl in lbl_list:
            try:
                del self._data_dict[lbl]
                self._key_list.remove(lbl)
            except KeyError:
                # do nothing
                pass


class AbstractDataView1D(AbstractDataView):
    """
    AbstractDataView1D class docstring.
    """

    # no init because AbstractDataView1D contains no new attributes

    def add_data(self, lbl_list, x_list, y_list, position=None):
        """
        add data with the name 'lbl'.  Will overwrite data if
        'lbl' already exists in the data dictionary

        Parameters
        ----------
        lbl : String
            Name of the data set
        x : np.ndarray
            single vector of x-coordinates
        y : np.ndarray
            single vector of y-coordinates
        position: int
            The position in the key list to begin inserting the data.
            Default (None) behavior is to append to the end of the list
        """
        # loop over the data passed in
        if position is None:
            position = len(self._key_list)
        for counter, (lbl, x, y) in enumerate(zip(lbl_list, x_list, y_list)):
            self._data_dict[lbl] = (x, y)
            self._key_list.insert(position + counter, lbl)

    def append_data(self, lbl_list, x_list, y_list):
        """
        Append (x, y) coordinates to a dataset.  If there is no dataset
        called 'lbl', add the (x_data, y_data) tuple to a new entry
        specified by 'lbl'

        Parameters
        ----------
        lbl : list
            str
            name of data set to append
        x : list
            np.ndarray
            single vector of x-coordinates to add.
            x_data must be the same length as y_data
        y : list
            np.ndarray
            single vector of y-coordinates to add.
            y_data must be the same length as x_data
        """
        lbl_to_add = []
        x_to_add = []
        y_to_add = []
        for lbl, x, y in zip(lbl_list, x_list, y_list):
            lbl = str(lbl)
            if lbl in self._data_dict:
                # get the current vectors at 'lbl'
                (prev_x, prev_y) = self._data_dict[lbl]
                # set the concatenated data to 'lbl'
                self._data_dict[lbl] = (np.concatenate((prev_x, x)), np.concatenate((prev_y, y)))
            else:
                # key doesn't exist, append the data to lists
                lbl_to_add.append(lbl)
                x_to_add.append(x)
                y_to_add.append(y)
        if len(lbl_to_add) > 0:
            self.add_data(lbl_list=lbl_to_add, x_list=x_to_add, y_list=y_to_add)


class AbstractDataView2D(AbstractDataView):
    def __init__(self, data_list, key_list, *args, **kwargs):
        """
        Parameters
        ----------
        data_dict : Dict
            k:v pairs of data
        key_list : List
            ordered key list which defines the order that images appear in the
            stack
        corners_dict : Dict
            k:v pairs of the location of the corners of each image
            (x0, y0, x1, y1)
        """
        # super().__init__(data_list=data_list, key_list=key_list, *args, **kwargs)
        super().__init__(data_list=data_list, key_list=key_list, **kwargs)

    def add_data(self, lbl_list, xy_list, corners_list=None, position=None):
        """
        add data with the name 'lbl'.  Will overwrite data if
        'lbl' already exists in the data dictionary

        Parameters
        ----------
        lbl : String
            Name of the data set
        x : np.ndarray
            single vector of x-coordinates
        y : np.ndarray
            single vector of y-coordinates
        position: int
            The position in the key list to begin inserting the data.
            Default (None) behavior is to append to the end of the list
        """
        # check for default corners_list behavior
        if corners_list is None:
            corners_list = self.default_list_type()
            for xy in xy_list:
                corners_list.append(self.find_corners(xy))
        # declare a local loop index
        counter = 0
        # loop over the data passed in
        for lbl, xy, corners in zip(lbl_list, xy_list, corners_list):
            # stash the data
            self._data_dict[lbl] = xy
            # stash the corners
            self._corners_dict[lbl] = corners
            # insert the key into the desired position in the keys list
            if position is None:
                self._key_list.append(lbl)
            else:
                self._key_list.insert(i=position + counter, x=lbl)
                counter += 1

    def append_data(self, lbl_list, xy_list, axis=None, append_to_end=None):
        """
        Append (x, y) coordinates to a dataset.  If there is no dataset
        called 'lbl', add the (x_data, y_data) tuple to a new entry
        specified by 'lbl'

        Parameters
        ----------
        lbl : list
            str
            name of data set to append
        xy : list
            np.ndarray
            List of 2D arrays
        axis : list
            int
            axis == 0 is appending in the horizontal direction
            axis == 1 is appending in the vertical direction
        append_to_end : list
            bool
            if false, prepend to the dataset
        """
        if axis is None:
            axis = []
        if append_to_end is None:
            append_to_end = []
        for lbl, xy, ax, end in zip(lbl_list, xy_list, axis, append_to_end):
            try:
                # set the concatenated data to 'lbl'
                if end:
                    self._data_dict[lbl] = np.r_[str(ax), self._data_dict[lbl], xy]
                    # TODO: Need to update the corners_list also...
                else:
                    self._data_dict[lbl] = np.r_[str(ax), xy, self._data_dict[lbl]]
                    # TODO: Need to update the corners_list also...
            except KeyError:
                # key doesn't exist, add data to a new entry called 'lbl'
                self.add_data(lbl, xy)

    def add_datum(self, lbl_list, x_list, y_list, val_list):
        """
        Add a single data point to an array

        Parameters
        ----------
        lbl : list
            str
            name of the dataset to add one datum to
        x : list
            int
            index of x coordinate
        y : list
            int
            index of y coordinate
        val : list
            float
            value of datum at the coordinates specified by (x,y)
        """
        raise NotImplementedError("Not yet implemented")
