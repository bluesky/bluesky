import itertools
from functools import partial
import re

from .mpl_plotting import Grid, Trajectory
import matplotlib.pyplot as plt
import numpy as np


def find_figure(dim_fields, columns, overplot=False, fig_factory=None):
    '''Determines where the figure should be plotted and returns the figure and
    axes references.
    This section checks to see if a current figure exists that the plots can be
    added too, and if not creates the figure to be plotted on. It is designed
    to be used in all of the plotting related callback_factories.

    Parameters
    ----------
    start_doc : dict
        the start document issued by the `Run_Engine` when starting a
    new 'run'.

    Returns
    -------
    fig : matplotlib fig reference
        the reference to the figure we should use to plot.
    axes : list
        list of matplotlib axes references.
    '''
    # I am proposing placing here all of the mechanisms associated with
    # checking if we can use an existing figure, how many subplots should be
    # used and how to arragne them. I prefer this as otherwise there would be
    # a lot of similar code being employed in each of the figure based
    # callback_factories. In addition most of this will probably end up living
    # on the databrowser what do people think?

    # Create a figure or reuse an existing one.

    fig_name = '{} vs {}'.format(' '.join(sorted(columns)),
                                 ' '.join(sorted(dim_fields)))
    if overplot and len(dim_fields) == 1:
        # If any open figure matches 'figname {number}', use it. If there
        # are multiple, the most recently touched one will be used.
        pat1 = re.compile('^' + fig_name + '$')
        pat2 = re.compile('^' + fig_name + r' \d+$')
        for label in plt.get_figlabels():
            if pat1.match(label) or pat2.match(label):
                fig_name = label
                break
    else:
        if plt.fignum_exists(fig_name):
            # Generate a unique name by appending a number.
            for number in itertools.count(2):
                new_name = '{} {}'.format(fig_name, number)
                if not plt.fignum_exists(new_name):
                    fig_name = new_name
                    break
    ndims = len(dim_fields)
    if not 0 < ndims < 3:
        # we need 1 or 2 dims to do anything, do not make empty figures
        return

    if fig_factory:
        fig = fig_factory(fig_name)
    else:
        fig = plt.figure(fig_name)
    if not fig.axes:
        # This is apparently a fresh figure. Make axes.
        # The complexity here is due to making a shared x axis. This can be
        # simplified when Figure supports the `subplots` method in a future
        # release of matplotlib.
        fig.set_size_inches(6.4, min(950, len(columns) * 400) / fig.dpi)
        for i in range(len(columns)):
            if i == 0:
                ax = fig.add_subplot(len(columns), 1, 1 + i)
                if ndims == 1:
                    share_kwargs = {'sharex': ax}
                elif ndims == 2:
                    share_kwargs = {'sharex': ax, 'sharey': ax}
                else:
                    raise NotImplementedError("we now support 3D?!")
            else:
                ax = fig.add_subplot(len(columns), 1, 1 + i,
                                     **share_kwargs)
    axes = fig.axes
    return fig, axes


def hinted_fields(descriptor):
    # Figure out which columns to put in the table and/or to plot.
    obj_names = list(descriptor['object_keys'])
    # We will see if these objects hint at whether
    # a subset of their data keys ('fields') are interesting. If they
    # did, we'll use those. If these didn't, we know that the RunEngine
    # *always* records their complete list of fields, so we can use
    # them all unselectively.
    columns = []
    for obj_name in obj_names:
        try:
            fields = descriptor.get('hints', {}).get(obj_name, {})['fields']
        except KeyError:
            fields = descriptor['object_keys'][obj_name]
        columns.extend(fields)
    return columns


def independent_variables(start_doc):
    hints = start_doc.get('hints', {})
    all_dim_names = [field
                     for fields, stream_name in hints['dimension']
                     for field in fields]
    dim_names = [fields[0]
                 for fields, stream_name in hints['dimensions']]

    return all_dim_names, dim_names


def grid_factory_descriptor(all_dim_names, dim_names, shape,
                            extent, descriptor, origin='lower'):
    columns = hinted_fields(descriptor)
    fig, axes = find_figure(dim_names, columns)
    callbacks = []
    I_names = [c for c in columns
               if c not in all_dim_names]
    for I_name, ax in zip(I_names, axes):

        # This section defines the function for the grid callback
        def func(self, bulk_event):
            '''This functions takes in a bulk event and returns x_coords,
            y_coords, I_vals lists.
            '''
            # start by working out the scaling between grid pixels and axes
            # units
            data_range = np.array([float(np.diff(e)) for e in self.extent])
            y_step, x_step = data_range / [max(1, s - 1) for s in
                                           self.shape]
            x_min = self.extent[0]
            y_min = self.extent[2]
            # define the lists of relevant data from the bulk_event
            x_vals = bulk_event['data'][dim_names[1]]
            y_vals = bulk_event['data'][dim_names[0]]
            I_vals = bulk_event['data'][I_name]
            x_coords = []
            y_coords = []

            for x_val, y_val in zip(x_vals, y_vals):
                x_coords.append((x_val - x_min) / x_step)
                y_coords.append((y_val - y_min) / y_step)
            return x_coords, y_coords, I_vals  # lists to be returned

        grid_callback = Grid(func, shape, ax=ax,
                             extent=extent, origin=origin)
        callbacks.append(grid_callback)

        # This section defines the callback for the overlayed path.
        def trajectory_func(self, bulk_event):
            '''This functions takes in a bulk event and returns x_vals, y_vals
            lists.
            '''
            x_vals = bulk_event['data'][dim_names[1]]
            y_vals = bulk_event['data'][dim_names[0]]
            return x_vals, y_vals

        # if x_trajectory is not None:
        #     trajectory_callback = Trajectory(start_doc, trajectory_func,
        #                                      x_trajectory, y_trajectory,
        #                                      ax=ax)
        #
        #     callbacks.append(trajectory_callback)
    return callbacks


def grid_factory_start(start_doc):
    '''
    This is a callback factory for 'grid' or 'image' plots. It takes in a
    start_doc and returns a list of callbacks that have been initialized based
    on its contents.
    '''
    shape = start_doc['shape']
    # If this isn't a 2D thing don't even bother
    if len(shape) != 2:
        return [], []
    extent = start_doc['extents']

    # define some required parameters for setting up the grid plot.
    # NOTE: THIS NEEDS WORK, in order to allow for plotting of non-grid type
    # scans the following parameters need to be passed down to here from the RE
    # This is the minimum information required to create the grid plot.

    # This section adjusts extents so that the values are centered on the grid
    # pixels
    data_range = np.array([float(np.diff(e)) for e in extent])
    y_step, x_step = data_range / [max(1, s - 1) for s in shape]
    adjusted_extent = [extent[1][0] - x_step / 2,
                       extent[1][1] + x_step / 2,
                       extent[0][0] - y_step / 2,
                       extent[0][1] + y_step / 2]

    # This section is where the scan path is defined, if the x_trajectory and
    # y_trajectory are 'None' it is not overlayed.
    # NOTE: we need to decide how to pass this info down to here.

    # x_trajectory = None  # This should be able to take in the path info here.
    # y_trajectory = None  # This should be able to take in the path info here.

    return [], [partial(grid_factory_descriptor,
                        *independent_variables(start_doc),
                        shape,
                        adjusted_extent,)
                ]
