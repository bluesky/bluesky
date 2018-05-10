# These are experimental IPython magics, providing quick shortcuts for simple
# tasks. None of these save any data.

# To use, run this in an IPython shell:
# ip = get_ipython()
# ip.register_magics(BlueskyMagics)

import asyncio
from bluesky.utils import ProgressBarManager
from bluesky import RunEngine, RunEngineInterrupted
from IPython.core.magic import Magics, magics_class, line_magic
import numpy as np
import collections
from operator import attrgetter
from . import plans as bp
from . import plan_stubs as bps
from bluesky.utils import separate_devices

try:
    # cytools is a drop-in replacement for toolz, implemented in Cython
    from cytools import partition
except ImportError:
    from toolz import partition


@magics_class
class BlueskyMagics(Magics):
    """
    IPython magics for bluesky.

    To install:

    >>> ip = get_ipython()
    >>> ip.register_magics(BlueskyMagics)

    Optionally configure default detectors and positioners by setting
    the class attributes:

    * ``BlueskyMagics.detectors``
    * ``BlueskyMagics.positioners``

    For more advanced configuration, access the magic's RunEngine instance and
    ProgressBarManager instance:

    * ``BlueskyMagics.RE``
    * ``BlueskyMagics.pbar_manager``
    """
    RE = RunEngine({}, loop=asyncio.new_event_loop())
    pbar_manager = ProgressBarManager()

    def _ensure_idle(self):
        if self.RE.state != 'idle':
            print('The RunEngine invoked by magics cannot be resumed.')
            print('Aborting...')
            self.RE.abort()

    @line_magic
    def mov(self, line):
        if len(line.split()) % 2 != 0:
            raise TypeError("Wrong parameters. Expected: "
                            "%mov motor position (or several pairs like that)")
        args = []
        for motor, pos in partition(2, line.split()):
            args.append(eval(motor, self.shell.user_ns))
            args.append(eval(pos, self.shell.user_ns))
        plan = bps.mv(*args)
        self.RE.waiting_hook = self.pbar_manager
        try:
            self.RE(plan)
        except RunEngineInterrupted:
            ...
        self.RE.waiting_hook = None
        self._ensure_idle()
        return None

    @line_magic
    def movr(self, line):
        if len(line.split()) % 2 != 0:
            raise TypeError("Wrong parameters. Expected: "
                            "%mov motor position (or several pairs like that)")
        args = []
        for motor, pos in partition(2, line.split()):
            args.append(eval(motor, self.shell.user_ns))
            args.append(eval(pos, self.shell.user_ns))
        plan = bps.mvr(*args)
        self.RE.waiting_hook = self.pbar_manager
        try:
            self.RE(plan)
        except RunEngineInterrupted:
            ...
        self.RE.waiting_hook = None
        self._ensure_idle()
        return None

    detectors = []

    @line_magic
    def ct(self, line):
        if line.strip():
            dets = eval(line, self.shell.user_ns)
        else:
            dets = self.detectors
        plan = bp.count(dets)
        print("[This data will not be saved. "
              "Use the RunEngine to collect data.]")
        try:
            self.RE(plan, _ct_callback)
        except RunEngineInterrupted:
            ...
        self._ensure_idle()
        return None

    positioners = []
    FMT_PREC = 6

    @line_magic
    def detectors(self, line):
        ''' List all available detectors.'''
        # also make sure it has a name for printing
        label='detector'
        devices = _labeled_devices(user_ns=self.shell.user_ns)
        cols = ["Python name", "Ophyd Name"]
        print("{:20s} \t {:20s}".format(*cols))
        print("="*40)
        for name, obj in devices[label]:
            print("{:20s} \t {:20s}".format(name, str(obj.name)))

    @line_magic
    def motors(self, line):
        ''' List all available motors.'''
        # also make sure it has a name for printing
        label='motor'
        devices = _labeled_devices(user_ns=self.shell.user_ns)
        # ignore the first key
        positioners = [positioner[1] for positioner in devices[label]]
        _print_positioners(positioners, precision=self.FMT_PREC)

    @line_magic
    def wa(self, line):
        "List positioner info. 'wa' stands for 'where all'."
        if line.strip():
            positioners = eval(line, self.shell.user_ns)
        else:
            positioners = self.positioners
        _print_positioners(positioners, precision=self.FMT_PREC)


def _print_positioners(positioners, sort=True, precision=6):
    '''
        This will take a list of positioners and try to print them.

        Parameters
        ----------
        positioners : list
            list of positioners

        sort : bool, optional
            whether or not to sort the list

        precision: int, optional
            The precision to use for numbers
    '''
    # sort first
    if sort:
        positioners = sorted(set(positioners), key=attrgetter('name'))

    values = []
    for p in positioners:
        try:
            values.append(p.position)
        except Exception as exc:
            values.append(exc)

    headers = ['Positioner', 'Value', 'Low Limit', 'High Limit', 'Offset']
    LINE_FMT = '{: <30} {: <11} {: <11} {: <11} {: <11}'
    lines = []
    lines.append(LINE_FMT.format(*headers))
    for p, v in zip(positioners, values):
        if not isinstance(v, Exception):
            try:
                prec = p.precision
            except Exception:
                prec = precision
            value = np.round(v, decimals=prec)
            try:
                low_limit, high_limit = p.limits
            except Exception as exc:
                low_limit = high_limit = exc.__class__.__name__
            else:
                low_limit = np.round(low_limit, decimals=prec)
                high_limit = np.round(high_limit, decimals=prec)
            try:
                offset = p.user_offset.get()
            except Exception as exc:
                offset = exc.__class__.__name__
            else:
                offset = np.round(offset, decimals=prec)
        else:
            value = v.__class__.__name__  # e.g. 'DisconnectedError'
            low_limit = high_limit = offset = ''

        lines.append(LINE_FMT.format(p.name, value, low_limit, high_limit,
                                     offset))
    print('\n'.join(lines))


def _labeled_devices(user_ns=None, maxdepth=6):
    ''' Returns list of all devices that are labeled.

        Parameters
        ----------
        user_ns : dict, optional
            The namespace to search on

        maxdepth: int, optional
            max recursion depth

        Returns
        -------
            A dictionary of (name, ophydobject) tuple indexed by device label.

        Examples
        --------
        Read devices labeled as motors:
            objs = _labeled_devices()
            my_motors = objs['motors']
    '''
    # could be set but lists are more common for users
    obj_list = collections.defaultdict(list)

    if maxdepth <= 0:
        print("Recursion limit exceeded")
        return obj_list

    if user_ns is None:
        user_ns = get_ipython().user_ns

    for key, obj in user_ns.items():
        # ignore objects beginning with "_"
        # (mainly for ipython stored objs from command line
        # return of commands)
        # also check its a subclass of desired classes
        if not key.startswith("_"):
            if is_parent(obj):
                labels = getattr(obj, '_ophyd_labels_', set())
                obj_list.update(_labeled_devices(user_ns=obj.__dict__,
                                              maxdepth=maxdepth-1,))
            else:
                if hasattr(obj, '_ophyd_labels_'):
                    # inherit parent labels
                    labels = obj._ophyd_labels_
                    for label in labels:
                        obj_list[label].append((key, obj))

    return obj_list


def is_parent(dev):
    # return whether a node is a parent
    # should not have component_names, or if yes, should be empty
    # read_attrs needed to check it's an instance and not class itself
    return (hasattr(dev, 'component_names') and len(dev.component_names) > 0
            and hasattr(dev, 'read_attrs'))

def get_children(dev):
    children = list()
    if hasattr(dev, 'component_names') and len(dev.component_names) > 0:
        for comp_name in dev.component_names:
            children.append(getattr(dev, comp_name))
    return children

def _ct_callback(name, doc):
    if name != 'event':
        return
    for k, v in doc['data'].items():
        print('{: <30} {}'.format(k, v))
