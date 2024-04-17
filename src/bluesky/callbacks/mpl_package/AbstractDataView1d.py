import logging

import numpy as np

from bluesky.callbacks.mpl_package import AbstractDataView

logger = logging.getLogger(__name__)


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

    def append_data(self, lbl_list, xy_list, axis=None, append_to_end=None) -> None:
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
