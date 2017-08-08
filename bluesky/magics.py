# These are experimental IPython magics for executing a plan using a syntax
# familiar to users of SPEC.

# To use, run this in an IPython shell:
# ip = get_ipython()
# ip.register_magics(BlueskyMagics)
# ip.register_magics(SPECMagics)

from IPython.core.magic import Magics, magics_class, line_magic
from ast import literal_eval as le
import bluesky.plans as bp
import numpy as np
from operator import attrgetter

try:
    # cytools is a drop-in replacement for toolz, implemented in Cython
    from cytools import partition
except ImportError:
    from toolz import partition


# Wrap the user namespace to provide newbie-friendly KeyError messages.
class Namespace:
    def __init__(self, ns):
        self.ns = ns

    def __getitem__(self, key):
        try:
            return self.ns[key]
        except KeyError:
            raise KeyError("No variable named {!r} is defined in the "
                           "user namespace. Define it and try again."
                           "".format(key))

@magics_class
class BlueskyMagics(Magics):

    @line_magic
    def mov(self, line):
        if len(line.split()) % 2 != 0:
            raise TypeError("Wrong parameters. Expected: "
                            "%mov motor position (or several pairs like that)")
        ns = Namespace(self.shell.user_ns)
        RE = ns['RE']
        args = []
        strs = []
        for motor, pos in partition(2, line.split()):
            args.extend([ns[motor], le(pos)])
            strs.extend([motor, repr(le(pos))])
        print('---> RE(mv({}))'.format(', '.join(strs)))
        plan = bp.mv(*args)
        return RE(plan)

    @line_magic
    def movr(self, line):
        if len(line.split()) % 2 != 0:
            raise TypeError("Wrong parameters. Expected: "
                            "%mov motor position (or several pairs like that)")
        ns = Namespace(self.shell.user_ns)
        RE = ns['RE']
        args = []
        strs = []
        for motor, pos in partition(2, line.split()):
            args.extend([ns[motor], le(pos)])
            strs.extend([motor, repr(le(pos))])
        print('---> RE(mvr({}))'.format(', '.join(strs)))
        plan = bp.mvr(*args)
        return RE(plan)

@magics_class
class SPECMagics(Magics):

    @line_magic
    def ct(self, line):
        if line.split():
            raise TypeError("No parameters expected, just %ct")
        print('---> RE(count(dets))')
        ns = Namespace(self.shell.user_ns)
        RE = ns['RE']
        plan = bp.count(ns['dets'])
        return RE(plan)

    @line_magic
    def ascan(self, line):
        # SPEC's ascan expects number of *intervals*, as opposed to the number
        # of points (intervals = points - 1). We follow SPEC's API but try to
        # make the difference clear in the string we print.
        try:
            motor, start, stop, intervals = line.split()
        except ValueError:
            raise TypeError("Wrong parameters. Expected: "
                            "%ascan motor start stop intervals")
        print('---> RE(scan(dets, {}, {}, {}, {} + 1))'.format(*line.split()))
        ns = Namespace(self.shell.user_ns)
        RE = ns['RE']
        plan = bp.scan(ns['dets'], ns[motor],
                       le(start), le(stop), 1 + le(intervals))
        return RE(plan)

    @line_magic
    def dscan(self, line):
        try:
            motor, start, stop, intervals = line.split()
        except ValueError:
            raise TypeError("Wrong parameters. Expected: "
                            "%dscan motor start stop intervals")
        print('---> RE(relative_scan(dets, {}, {}, {}, {} + 1))'
              ''.format(*line.split()))
        ns = Namespace(self.shell.user_ns)
        RE = ns['RE']
        plan = bp.relative_scan(ns['dets'], ns[motor],
                                le(start), le(stop), 1 + le(intervals))
        return RE(plan)

    @line_magic
    def a2scan(self, line):
        try:
            (motor1, start1, stop1,
             motor2, start2, stop2,
             intervals) = line.split()
        except ValueError:
            raise TypeError("Wrong parameters. Expected: "
                            "%a2scan motor1 start1 stop1 "
                            "motor2 start2 stop2 intervals")
        print('---> RE(inner_product_scan(dets, {} + 1, {}, {}, {}, '
              '{}, {}, {}))'.format(intervals,
                                    motor1, start1, stop1,
                                    motor2, start2, stop2))
        ns = Namespace(self.shell.user_ns)
        RE = ns['RE']
        plan = bp.inner_product_scan(ns['dets'], 1 + le(intervals),
                                     ns[motor1], le(start1), le(stop1),
                                     ns[motor2], le(start2), le(stop2))
        return RE(plan)

    @line_magic
    def d2scan(self, line):
        try:
            (motor1, start1, stop1,
             motor2, start2, stop2,
             intervals) = line.split()
        except ValueError:
            raise TypeError("Wrong parameters. Expected: "
                            "%d2scan motor1 start1 stop1 "
                            "motor2 start2 stop2 intervals")
        print('---> RE(relative_inner_product_scan(dets, {} + 1, {}, {}, {}, '
              '{}, {}, {}))'.format(intervals,
                                    motor1, start1, stop1,
                                    motor2, start2, stop2))
        ns = Namespace(self.shell.user_ns)
        RE = ns['RE']
        plan = bp.relative_inner_product_scan(
            ns['dets'], 1 + le(intervals),
            ns[motor1], le(start1), le(stop1),
            ns[motor2], le(start2), le(stop2))
        return RE(plan)

    @line_magic
    def mesh(self, line):
        try:
            (motor1, start1, stop1, intervals1,
             motor2, start2, stop2, intervals2) = line.split()
        except ValueError:
            raise TypeError("Wrong parameters. Expected: "
                            "%mesh motor1 start1 stop1 intervals1"
                            "motor2 start2 stop2 intervals2")
        print('---> RE(outer_product_scan(dets, {} + 1, {}, {}, {}, '
              '{}, {}, {}, False))'.format(motor1, start1, stop1, intervals1,
                                    motor2, start2, stop2, intervals2))
        ns = Namespace(self.shell.user_ns)
        RE = ns['RE']
        plan = bp.outer_product_scan(
            ns['dets'],
            ns[motor1], le(start1), le(stop1), 1 + le(intervals1),
            ns[motor2], le(start2), le(stop2), 1 + le(intervals2), False)
        return RE(plan)

    positioners = []
    FMT_PREC = 6

    @line_magic
    def wa(self, line):
        "List positioner info. 'wa' stands for 'where all'."
        if line.split():
            raise TypeError("No parameters expected, just %wa")
        positioners = sorted(set(self.positioners), key=attrgetter('name'))
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
