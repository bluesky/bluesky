import functools
import operator

import numpy as np
from cycler import cycler
try:
    # cytools is a drop-in replacement for toolz, implemented in Cython
    from cytools import partition
except ImportError:
    from toolz import partition

from .utils import snake_cyclers


def spiral(x_motor, y_motor, x_start, y_start, x_range, y_range, dr, nth, *,
           dr_y=None, tilt=0.0):
    '''Spiral scan, centered around (x_start, y_start)

    Parameters
    ----------
    x_motor : object, optional
        any 'setable' object (motor, temp controller, etc.)
    y_motor : object, optional
        any 'setable' object (motor, temp controller, etc.)
    x_start : float
        x center
    y_start : float
        y center
    x_range : float
        x width of spiral
    y_range : float
        y width of spiral
    dr : float
        Delta radius along the minor axis of the ellipse.
    dr_y : float, optional
        Delta radius along the major axis of the ellipse, if not specifed defaults to dr
    nth : float
        Number of theta steps
    tilt : float, optional
        Tilt angle in radians, default 0.0

    Returns
    -------
    cyc : cycler
    '''
    if dr_y is None:
        dr_aspect = 1
    else:
        dr_aspect = dr_y / dr
    
    half_x = x_range / 2
    half_y = y_range / (2 * dr_aspect)

    r_max = np.sqrt(half_x ** 2 + half_y ** 2)
    num_ring = 1 + int(r_max / dr)
    tilt_tan = np.tan(tilt + np.pi / 2.)

    x_points, y_points = [], []

    for i_ring in range(1, num_ring + 2):
        radius = i_ring * dr
        angle_step = 2. * np.pi / (i_ring * nth)

        for i_angle in range(int(i_ring * nth)):
            angle = i_angle * angle_step
            x = radius * np.cos(angle)
            y = radius * np.sin(angle) * dr_aspect
            if ((abs(x - (y / dr_aspect) / tilt_tan) <= half_x) and 
                    (abs(y / dr_aspect) <= half_y)):
                x_points.append(x_start + x)
                y_points.append(y_start + y)

    cyc = cycler(x_motor, x_points)
    cyc += cycler(y_motor, y_points)
    return cyc


def spiral_square_pattern(x_motor, y_motor, x_center, y_center,
                          x_range, y_range, x_num, y_num):
    '''
    Square spiral scan, centered around (x_start, y_start)

    Parameters
    ----------
    x_motor : object, optional
        any 'setable' object (motor, temp controller, etc.)
    y_motor : object, optional
        any 'setable' object (motor, temp controller, etc.)
    x_center : float
        x center
    y_center : float
        y center
    x_range : float
        x width of spiral
    y_range : float
        y width of spiral
    x_num : float
        number of x axis points
    y_num : float
        number of y axis points
    Returns
    -------
    cyc : cycler
    '''
    x_points, y_points = [], []

    # checks if x_num/y_num is even or odd and sets the required offset
    # parameter for the start point from the centre point.
    if x_num % 2 == 0:
        x_offset = 0.5
    else:
        x_offset = 0

    if y_num % 2 == 0:
        y_offset = -0.5
    else:
        y_offset = 0

    num_ring = max(x_num, y_num)
    x_delta = x_range / (x_num - 1)
    y_delta = y_range / (y_num - 1)

    # include the first point, as it is the first 'ring' to include.
    x_points.append(x_center - x_delta * x_offset)
    y_points.append(y_center - y_delta * y_offset)

    # set the number of found points to 0
    num_pnts_fnd = 1

    # step through each of the rings required to map out the entire area.
    for i_ring in range(2, num_ring+1, 1):
        # step through each of the 'sides' of the ring if the constant value
        # for each side is within the range to plot and
        # that we have not already found all the required points.
        # SIDE 1
        if (abs(i_ring - 1 - x_offset) <= x_num / 2) and \
           (num_pnts_fnd < x_num * y_num):
            for n in range(i_ring-2, -i_ring, -1):
                # Ensure that the variable value for this side is within the
                # range to plot and that we have not already
                # found all the required points.
                if (abs(n - y_offset) < y_num / 2) and \
                   (num_pnts_fnd < y_num * x_num):

                    x = x_center - x_delta * x_offset + x_delta * (i_ring - 1)
                    y = y_center - y_delta * y_offset + y_delta * n
                    num_pnts_fnd += 1

                    x_points.append(x)
                    y_points.append(y)

        # SIDE 2
        if (abs(-i_ring + 1 - y_offset) < y_num / 2) and \
           (num_pnts_fnd < x_num * y_num):
            for n in range(i_ring - 2, -i_ring, -1):
                # Ensure that the variable value for this side is within the
                # range to plot and that we have not already
                # found all the required points.
                if (abs(n - x_offset) < x_num / 2) and \
                   (num_pnts_fnd < y_num * x_num):

                    x = x_center - x_delta * x_offset + x_delta * n
                    y = y_center - y_delta * y_offset + y_delta * (-i_ring + 1)

                    num_pnts_fnd += 1

                    x_points.append(x)
                    y_points.append(y)

        # SIDE 3
        if (abs(-i_ring + 1 - x_offset) < x_num / 2) and \
           (num_pnts_fnd < x_num * y_num):
            for n in range(-i_ring + 2, i_ring, 1):
                # Ensure that the variable value for this side is within the
                # range to plot and that we have not already
                # found all the required points.
                if (abs(n - y_offset) < y_num / 2) and \
                   (num_pnts_fnd < y_num * x_num):

                    x = x_center - x_delta * x_offset + x_delta * (-i_ring + 1)
                    y = y_center - y_delta * y_offset + y_delta * n
                    num_pnts_fnd += 1

                    x_points.append(x)
                    y_points.append(y)

        # SIDE 4
        if (abs(i_ring - 1 - y_offset) < y_num / 2) and \
           (num_pnts_fnd < x_num * y_num):
            for n in range(-i_ring + 2, i_ring, 1):
                # Ensure that the variable value for this side is within the
                # range to plot and that we have not already
                # found all the required points.
                if (abs(n - x_offset) < x_num / 2) and \
                   (num_pnts_fnd < y_num * x_num):

                    x = x_center - x_delta * x_offset + x_delta * n
                    y = y_center - y_delta * y_offset + y_delta * (i_ring - 1)

                    num_pnts_fnd += 1

                    x_points.append(x)
                    y_points.append(y)

    cyc = cycler(x_motor, x_points)
    cyc += cycler(y_motor, y_points)

    return cyc


def spiral_fermat(x_motor, y_motor, x_start, y_start, x_range, y_range, dr,
                  factor, *, dr_y=None, tilt=0.0):
    '''Absolute fermat spiral scan, centered around (x_start, y_start)

    Parameters
    ----------
    x_motor : object, optional
        any 'setable' object (motor, temp controller, etc.)
    y_motor : object, optional
        any 'setable' object (motor, temp controller, etc.)
    x_start : float
        x center
    y_start : float
        y center
    x_range : float
        x width of spiral
    y_range : float
        y width of spiral
    dr : float
        delta radius along the minor axis of the ellipse.
    dr_y : float, optional
        Delta radius along the major axis of the ellipse, if not specifed defaults to dr
    factor : float
        radius gets divided by this
    tilt : float, optional
        Tilt angle in radians, default 0.0

    Returns
    -------
    cyc : cycler
    '''
    if dr_y is None:
        dr_aspect = 1
    else:
        dr_aspect = dr_y / dr
    
    phi = 137.508 * np.pi / 180.

    half_x = x_range / 2
    half_y = y_range / (2 * dr_aspect)
    tilt_tan = np.tan(tilt + np.pi / 2.)

    x_points, y_points = [], []

    diag = np.sqrt(half_x ** 2 + half_y ** 2)
    num_rings = int((1.5 * diag / (dr / factor)) ** 2)
    for i_ring in range(1, num_rings):
        radius = np.sqrt(i_ring) * dr / factor
        angle = phi * i_ring
        x = radius * np.cos(angle)
        y = radius * np.sin(angle) * dr_aspect

        if ((abs(x - (y / dr_aspect) / tilt_tan) <= half_x) and (abs(y) <= half_y)):
            x_points.append(x_start + x)
            y_points.append(y_start + y)

    cyc = cycler(x_motor, x_points)
    cyc += cycler(y_motor, y_points)
    return cyc


def inner_product(num, args):
    '''Scan over one multi-motor trajectory.

    Parameters
    ----------
    num : integer
        number of steps
    args : list of {Positioner, Positioner, int}
        patterned like (``motor1, start1, stop1, ..., motorN, startN, stopN``)
        Motors can be any 'setable' object (motor, temp controller, etc.)

    Returns
    -------
    cyc : cycler
    '''
    if len(args) % 3 != 0:
        raise ValueError("wrong number of positional arguments")

    cyclers = []
    for motor, start, stop, in partition(3, args):
        steps = np.linspace(start, stop, num=num, endpoint=True)
        c = cycler(motor, steps)
        cyclers.append(c)
    return functools.reduce(operator.add, cyclers)


def chunk_outer_product_args(args):
    '''Scan over a mesh; each motor is on an independent trajectory.

    Parameters
    ----------
    args
        patterned like (``motor1, start1, stop1, num1,```
                        ``motor2, start2, stop2, num2, snake2,``
                        ``motor3, start3, stop3, num3, snake3,`` ...
                        ``motorN, startN, stopN, numN, snakeN``)

        The first motor is the "slowest", the outer loop. For all motors
        except the first motor, there is a "snake" argument: a boolean
        indicating whether to following snake-like, winding trajectory or a
        simple left-to-right trajectory.

    See Also
    --------
    `bluesky.plan_patterns.outer_product`

    Yields
    ------
    (motor, start, stop, num, snake)
    '''
    args = list(args)
    # The first (slowest) axis is never "snaked." Insert False to
    # make it easy to iterate over the chunks or args..
    args.insert(4, False)
    if len(args) % 5 != 0:
        raise ValueError("wrong number of positional arguments")

    yield from partition(5, args)


def outer_product(args):
    '''Scan over a mesh; each motor is on an independent trajectory.

    Parameters
    ----------
    args
        patterned like (``motor1, start1, stop1, num1,```
                        ``motor2, start2, stop2, num2, snake2,``
                        ``motor3, start3, stop3, num3, snake3,`` ...
                        ``motorN, startN, stopN, numN, snakeN``)

        The first motor is the "slowest", the outer loop. For all motors
        except the first motor, there is a "snake" argument: a boolean
        indicating whether to following snake-like, winding trajectory or a
        simple left-to-right trajectory.

    See Also
    --------
    `bluesky.plan_patterns.inner_product`

    Returns
    -------
    cyc : cycler
    '''
    shape = []
    extents = []
    snaking = []
    cyclers = []
    for motor, start, stop, num, snake in chunk_outer_product_args(args):
        shape.append(num)
        extents.append([start, stop])
        snaking.append(snake)
        steps = np.linspace(start, stop, num=num, endpoint=True)
        c = cycler(motor, steps)
        cyclers.append(c)

    return snake_cyclers(cyclers, snaking)
