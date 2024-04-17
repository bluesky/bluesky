import logging
from collections import defaultdict

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

    def update_color_map(self, cmap):
        """
        Update the color map used to display the image
        Parameters
        ----------
        cmap : mpl.cm.colors.Colormap
        """
        self._cmap = cmap

    def update_normalization(self, norm):
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
