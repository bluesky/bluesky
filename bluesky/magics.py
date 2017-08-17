# These are experimental IPython magics, providing quick shortcuts for simple
# tasks. None of these save any data.

# To use, run this in an IPython shell:
# ip = get_ipython()
# ip.register_magics(BlueskyMagics)

from ast import literal_eval as le
import asyncio
import bluesky.plans as bp
from bluesky import RunEngine
from IPython.core.magic import Magics, magics_class, line_magic
import numpy as np
from operator import attrgetter

try:
    # cytools is a drop-in replacement for toolz, implemented in Cython
    from cytools import partition
except ImportError:
    from toolz import partition


@magics_class
class BlueskyMagics(Magics):

    RE = RunEngine({}, loop = asyncio.new_event_loop())

    @line_magic
    def mov(self, line):
        if len(line.split()) % 2 != 0:
            raise TypeError("Wrong parameters. Expected: "
                            "%mov motor position (or several pairs like that)")
        args = []
        strs = []
        for motor, pos in partition(2, line.split()):
            args.extend([self.shell.user_ns[motor], le(pos)])
        plan = bp.mv(*args)
        self.RE(plan)
        return None

    @line_magic
    def movr(self, line):
        if len(line.split()) % 2 != 0:
            raise TypeError("Wrong parameters. Expected: "
                            "%mov motor position (or several pairs like that)")
        args = []
        strs = []
        for motor, pos in partition(2, line.split()):
            args.extend([self.shell.user_ns[motor], le(pos)])
        plan = bp.mvr(*args)
        self.RE(plan)
        return None

    dets = []

    @line_magic
    def ct(self, line):
        if line.strip():
            dets = eval(line, self.shell.user_ns)
        else:
            dets = self.dets  # default is SPECMagic.dets
        plan = bp.count(dets)
        print("[This data will not be saved. "
              "Use the RunEngine to collect data.]")
        self.RE(plan, _ct_callback)
        return None

    positioners = []
    FMT_PREC = 6

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

        headers = ['Positioner', 'Value', 'Low Limit', 'High Limit']
        LINE_FMT = '{: <30} {: <15} {: <15} {: <15}'
        lines = []
        lines.append(LINE_FMT.format(*headers))
        for p, v in zip(positioners, values):
            if not isinstance(v, Exception):
                try:
                    prec = p.precision
                except Exception as exc:
                    prec = self.FMT_PREC
                value = np.round(v, decimals=prec)
            else:
                value = exc.__class__.__name__  # e.g. 'DisconnectedError'
            try:
                low_limit, high_limit = p.low_limit, p.high_limit
            except Exception as exc:
                low_limit = high_limit = exc.__class__.__name__

            lines.append(LINE_FMT.format(p.name, value, low_limit, high_limit))
        print('\n'.join(lines))


def _ct_callback(name, doc):
    if name != 'event':
        return
    for k, v in doc['data'].items():
        print('{: <30} {}'.format(k, v))
