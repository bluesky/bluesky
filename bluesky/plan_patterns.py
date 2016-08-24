import functools
import operator

import numpy as np
from cycler import cycler
from boltons.iterutils import chunked

from .utils import snake_cyclers


def spiral(x_motor, y_motor, x_start, y_start, x_range, y_range, dr, nth, *,
           tilt=0.0):
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
        Delta radius
    nth : float
        Number of theta steps
    tilt : float, optional
        Tilt angle in radians, default 0.0

    Returns
    -------
    cyc : cycler
    '''
    half_x = x_range / 2
    half_y = y_range / 2

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
            y = radius * np.sin(angle)
            if ((abs(x - y / tilt_tan) <= half_x) and (abs(y) <= half_y)):
                x_points.append(x_start + x)
                y_points.append(y_start + y)

    cyc = cycler(x_motor, x_points)
    cyc += cycler(y_motor, y_points)
    return cyc


def spiral_fermat(x_motor, y_motor, x_start, y_start, x_range, y_range, dr,
                  factor, *, tilt=0.0):
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
        delta radius
    factor : float
        radius gets divided by this
    tilt : float, optional
        Tilt angle in radians, default 0.0

    Returns
    -------
    cyc : cycler
    '''
    phi = 137.508 * np.pi / 180.

    half_x = x_range / 2
    half_y = y_range / 2
    tilt_tan = np.tan(tilt + np.pi / 2.)

    x_points, y_points = [], []

    diag = np.sqrt(half_x ** 2 + half_y ** 2)
    num_rings = int((1.5 * diag / (dr / factor)) ** 2)
    for i_ring in range(1, num_rings):
        radius = np.sqrt(i_ring) * dr / factor
        angle = phi * i_ring
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)

        if ((abs(x - y / tilt_tan) <= half_x) and (abs(y) <= half_y)):
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
    for motor, start, stop, in chunked(args, 3):
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

    yield from chunked(args, 5)


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
