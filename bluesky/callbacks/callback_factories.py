from .mpl_plotting import Grid, Trajectory
import numpy as np


def find_figure(start_doc):
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


def grid_factory(start_doc):
    '''
    This is a callback factory for 'grid' or 'image' plots. It takes in a
    start_doc and returns a list of callbacks that have been initialized based
    on its contents.
    '''
    hints = start_doc.get('hints', {})
    callbacks = []

    # The next line is supposed to take care of where to plot, it can get that
    # info from anywhere (like the new proposed data-browser) I am expecting
    # the length of axes to match the number of self.I_names below.
    fig, axes = find_figure(start_doc)

    # below are some preliminary values required to generate the required
    # parameters.
    all_dim_names = [field
                     for fields, stream_name in hints['dimension']
                     for field in fields]  # find all dimension field names.

    # define some required parameters for setting up the grid plot.
    # NOTE: THIS NEEDS WORK, in order to allow for plotting of non-grid type
    # scans the following parameters need to be passed down to here from the RE
    # This is the minimum information required to create the grid plot.
    dim_names = [fields[0]
                 for fields, stream_name in hints['dimensions']]
    I_names = [c for c in hinted_fields(start_doc)
               if c not in all_dim_names]
    extent = start_doc['extents']
    shape = start_doc['shape']
    origin = 'lower'

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
    x_trajectory = None  # This should be able to take in the path info here.
    y_trajectory = None  # This should be able to take in the path info here.

    for I_name, ax in zip(I_names, axes):

        # This section defines the function for the grid callback
        def func(self, bulk_event):
            '''This functions takes in a bulk event and returns x_coords,
            y_coords, I_vals lists.
            '''
            # start by working out the scaling between grid pixels and axes
            # units
            data_range = np.array([float(np.diff(e)) for e in self.extent])
            y_step, x_step = data_range / [max(1, s - 1) for s in self.shape]
            x_min = self.extent[0]
            y_min = self.extent[2]
            # define the lists of relevant data from the bulk_event
            x_vals = bulk_event['data'][dim_names[1]]
            y_vals = bulk_event['data'][dim_names[0]]
            I_vals = bulk_event['data'][I_name]
            x_coords = []
            y_coords = []

            for x_val, y_val in zip(x_vals, y_vals):
                x_coords.append((x_val-x_min)/x_step)
                y_coords.append((y_val-y_min)/y_step)
            return x_coords, y_coords, I_vals  # lists to be returned

        grid_callback = Grid(start_doc, func, shape, ax=ax,
                             extent=adjusted_extent, origin=origin)
        callbacks.append(grid_callback)

        # This section defines the callback for the overlayed path.
        def trajectory_func(self, bulk_event):
            '''This functions takes in a bulk event and returns x_vals, y_vals
            lists.
            '''
            x_vals = bulk_event['data'][dim_names[1]]
            y_vals = bulk_event['data'][dim_names[0]]
            return x_vals, y_vals

        if x_trajectory is not None:
            trajectory_callback = Trajectory(start_doc, trajectory_func,
                                             x_trajectory, y_trajectory, ax=ax)

            callbacks.append(trajectory_callback)

    return callbacks
