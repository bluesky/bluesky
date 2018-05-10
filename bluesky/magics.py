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
        devices = _which_devices(labels=[label],
                                 user_ns=self.shell.user_ns)
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
        devices = _which_devices(labels=[label],
                                 user_ns=self.shell.user_ns)
        # ignore the first key
        positioners = [positioner[1] for positioner in devices[label]]
        _print_positioners(positioners, precision=self.FMT_PREC)

    @line_magic
    def signals(self, line):
        ''' List all ophyd signals.'''
        devices = _which_devices(labels=None, user_ns=self.shell.user_ns)
        cols = ["Python name", "Ophyd Name"]
        print("{:20s} \t {:20s}".format(*cols))
        print("="*40)
        for label, objs in devices.items():
            for name, obj in devices:
                print("{:20s} \t {:20s}".format(name, str(obj.name)))


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


def _which_devices(user_ns=None, max_depth=6):
    ''' Returns list of all devices according to the classes listed.

        Parameters
        ----------
        user_ns : dict, optional
            The namespace to search on

        max_dept: int, optional
            max recursion depth

        Examples
        --------
        Read from everything except EpicsMotor's:
            objs = _which_devices()
            objs['motors']
    '''
    # could be set but lists are more common for users
    obj_list = collections.defaultdict(list)

    if user_ns is None:
        user_ns = get_ipython().user_ns

    for key, obj in user_ns.items():
        # ignore objects beginning with "_"
        # (mainly for ipython stored objs from command line
        # return of commands)
        # also check its a subclass of desired classes
        if not key.startswith("_"):
            if hasattr(obj, '_ophyd_labels_'):
                for label in obj._ophyd_labels_:
                    obj_list[label].append((key, obj))

    return obj_list



def _ct_callback(name, doc):
    if name != 'event':
        return
    for k, v in doc['data'].items():
        print('{: <30} {}'.format(k, v))
