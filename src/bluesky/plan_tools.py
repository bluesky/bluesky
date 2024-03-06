import bluesky.simulators as _bs
from warnings import warn


def print_summary(plan):
    """Print summary of plan

    Prints a minimal version of the plan, showing only moves and
    where events are created.

    Parameters
    ----------
    plan : iterable
        Must yield `Msg` objects
    """
    warn("bluesky.plan_tools.print_summary is deprecated. Use "
         "bluesky.simulators.summarize_plan instead.")
    return _bs.summarize_plan(plan)


def print_summary_wrapper(plan):
    """Print summary of plan as it goes by

    Prints a minimal version of the plan, showing only moves and
    where events are created.  Yields the `Msg` unchanged.

    Parameters
    ----------
    plan : iterable
        Must yield `Msg` objects

    Yields
    ------
    msg : `Msg`
    """
    warn("The bluesky.plan_tools module is deprecated. Use "
         "bluesky.simulators instead.")
    yield from _bs.print_summary_wrapper(plan)


def plot_raster_path(plan, x_motor, y_motor, ax=None, probe_size=None, lw=2):
    """Plot the raster path for this plan

    Parameters
    ----------
    plan : iterable
       Must yield `Msg` objects and not be a co-routine

    x_motor, y_motor : str
       Names of the x and y motors

    ax : matplotlib.axes.Axes
       The axes to plot to, if none, make new figure + axes

    probe_size : float, optional
       If not None, use as radius of probe (in same units as motor positions)

    lw : float, optional
        Width of lines drawn between points
    """
    warn("The bluesky.plan_tools module is deprecated. Use "
         "bluesky.simulators instead.")
    return _bs.plot_raster_path(plan, x_motor, y_motor, ax=ax,
                                probe_size=probe_size, lw=lw)
