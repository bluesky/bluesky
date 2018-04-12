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
from operator import attrgetter
from . import plans as bp
from . import plan_stubs as bps

from ophyd.areadetector.base import ADBase

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
        devices = _which_devices(cls_whitelist=[ADBase], cls_blacklist=None)
        cols = ["Python name", "Ophyd Name"]
        print("{:20s} \t {:20s}".format(*cols))
        print("="*40)
        for name, obj in devices:
            print("{:20s} \t {:20s}".format(name, obj.name))


    @line_magic
    def wa(self, line):
        "List positioner info. 'wa' stands for 'where all'."
        if line.strip():
            positioners = eval(line, self.shell.user_ns)
        else:
            positioners = self.positioners
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
                    prec = self.FMT_PREC
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


def _which_devices(cls_whitelist=None, cls_blacklist=None):
    ''' Returns list of all devices according to the classes listed.

        Parameters
        ----------
        cls_whitelist : tuple or list, optional
            the class of PV's to search for
            defaults to [Device, Signal]
        cls_blacklist : tuple or list, optional

        Examples
        --------
        Read from everything except EpicsMotor's:
            objs = _which_devices(cls_blacklist=[EpicsMotor])
    '''
    if cls_whitelist is None:
        cls_whitelist = [Device, Signal]
    if cls_blacklist is None:
        cls_blacklist = []

    user_ns = get_ipython().user_ns

    obj_list = list()
    for key, obj in user_ns.items():
        # ignore objects beginning with "_"
        # (mainly for ipython stored objs from command line
        # return of commands)
        # also check its a subclass of desired classes
        if not key.startswith("_") and \
                isinstance(obj, tuple(cls_whitelist)) and \
                not isinstance(tuple(cls_blacklist)):
            obj_list.append((key, obj))

    return obj_list


def _ct_callback(name, doc):
    if name != 'event':
        return
    for k, v in doc['data'].items():
        print('{: <30} {}'.format(k, v))
