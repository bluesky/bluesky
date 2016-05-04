import types
import matplotlib.pyplot as plt
from matplotlib import collections as mcollections
from matplotlib import patches as mpatches


PLAN_TYPES = (types.GeneratorType,)
try:
    from types import CoroutineType
except ImportError:
    # < py35
    pass
else:
    PLAN_TYPES = PLAN_TYPES + (CoroutineType, )
    del CoroutineType


def ensure_generator(plan):
    """
    Ensure that the input is a generator.

    Parameters
    ----------
    plan : iterable or iterator

    Returns
    -------
    gen : coroutine
    """
    gen = iter(plan)  # no-op on generators; needed for classes
    if not isinstance(gen, PLAN_TYPES):
        # If plan does not support .send, we must wrap it in a generator.
        gen = (msg for msg in gen)

    return gen


def plot_raster_path(plan, x_motor, y_motor, ax=None, probe_size=None):
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
    """
    if ax is None:
        ax = plt.subplots()[1]
    ax.set_aspect('equal')

    cur_x = cur_y = None
    traj = []
    for msg in plan:
        cmd = msg.command
        if cmd == 'set':
            if msg.obj.name == x_motor:
                cur_x = msg.args[0]
            if msg.obj.name == y_motor:
                cur_y = msg.args[0]
        elif cmd == 'save':
            traj.append((cur_x, cur_y))

    x, y = zip(*traj)
    path, = ax.plot(x, y, marker='', linestyle='-', lw=2)
    if probe_size is None:
        read_points = ax.scatter(x, y, marker='o', lw=2)
    else:
        circles = [mpatches.Circle((_x, _y), probe_size,
                                   facecolor='black', alpha=0.5)
                   for _x, _y in traj]

        read_points = mcollections.PatchCollection(circles,
                                                   match_original=True)
        ax.add_collection(read_points)
    return {'path': path, 'events': read_points}


def print_summary(plan):
    """Print summary of plan

    Prints a minimal version of the plan, showing only moves and
    where events are created.

    Parameters
    ----------
    plan : iterable
        Must yield `Msg` objects
    """

    read_cache = []
    for msg in plan:
        cmd = msg.command
        if cmd == 'open_run':
            print('{:=^80}'.format(' Open Run '))
        elif cmd == 'close_run':
            print('{:=^80}'.format(' Close Run '))
        elif cmd == 'set':
            print('{motor.name} -> {args[0]}'.format(motor=msg.obj,
                                                     args=msg.args))
        elif cmd == 'create':
            pass
        elif cmd == 'read':
            read_cache.append(msg.obj.name)
        elif cmd == 'save':
            print('  Read {}'.format(read_cache))
            read_cache = []
