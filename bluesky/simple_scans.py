"""
These "scans" bundle a Message generator with an instance of the RunEngine,
combining two separate concepts -- instructions and execution -- into one
object. This makes the interface less flexible and somewhat less "Pythonic"
but more condensed.

This module is meant to be run in a namespace where several global
variables have been defined. If some variables are left undefined, the
associated scans will be not usable.

    DETS  # list of detectors
    MASTER_DET  # detector to use for tw
    MASTER_DET_FIELD  # detector field to use for tw
    H_MOTOR
    K_MOTOR
    L_MOTOR
    TH_MOTOR
    TTH_MOTOR
    TEMP_CONTROLLER

Page numbers in the code comments refer to the SPEC manual at
http://www.certif.com/downloads/css_docs/spec_man.pdf
"""
from inspect import signature
from bluesky import scans
from bluesky.callbacks import LiveTable, LivePlot
from boltons.iterutils import chunked
from bluesky.standard_config import gs


class _PrimitiveScan:

    def __init__(self):
        self.params = list(signature(self._scan_class).parameters.keys())

    def __call__(self, *args, subs=None, **kwargs):
        scan_kwargs = dict()
        # Any kwargs valid for the scan go to the scan, not the RE.
        for k, v in kwargs.items():
            if k in self.params:
                scan_kwargs[k] = kwargs.pop(k)
        from bluesky.standard_config import gs

        RE_params = list(signature(gs.RE.__call__).parameters.keys())
        if set(RE_params) & set(self.params):
            raise AssertionError("The names of the scan's arguments clash "
                                 "the RunEngine arguments. Use different "
                                 "names. Avoid: {0}".format(RE_params))

        self.scan = self._scan_class(gs.DETS, *args, **scan_kwargs)
        if subs is None:
            subs = self.subs
        # Any remainging kwargs go the RE. To be safe, no args are passed
        # to RE; RE args effectively become keyword-only arguments.
        return gs.RE(self.scan, subs, **kwargs)



### Mid-level base classes ###

# These are responsible for popping off the time arg and adjusting the
# interval. SPEC counts "bonds;" idiomatic Python counts "sites."

class _OuterProductScan(_PrimitiveScan):
    def __call__(self, *args, subs=None, **kwargs):
        args = list(args)
        time = args.pop(-1)
        original_times = _set_acquire_time(time)
        for i, _ in enumerate(chunked(args, 4)):
            # intervals -> intervals + 1
            args[4*i + 3] += 1
        result = super().__call__(*args, subs=subs, **kwargs)
        _unset_acquire_time(original_times)
        return result

    @property
    def subs(self):
        table = LiveTable(self.scan.motors + gs.TABLE_COLS)
        plot = LivePlot(gs.PLOT_Y, self.scan.motors[0]._name)
        return [table, plot]


class _InnerProductScan(_PrimitiveScan):
    def __call__(self, *args, subs=None, **kwargs):
        args = list(args)
        time = args.pop(-1)
        intervals = args.pop(-1) + 1
        original_times = _set_acquire_time(time)
        result = super().__call__(intervals, *args, subs=subs, **kwargs)
        _unset_acquire_time(original_times)
        return result

    @property
    def subs(self):
        table = LiveTable(self.scan.motors + gs.TABLE_COLS)
        plot = LivePlot(gs.PLOT_Y, self.scan.motors[0]._name)
        return [table, plot]


class _StepScan(_PrimitiveScan):

    def __call__(self, motor, start, finish, intervals, time,
                 subs=None, **kwargs):
        original_times = _set_acquire_time(time)
        result = super().__call__(motor, start, finish, intervals + 1,
                                  subs=subs, **kwargs)
        _unset_acquire_time(original_times)
        return result

    @property
    def subs(self):
        table = LiveTable([self.scan.motor] + gs.TABLE_COLS)
        plot = LivePlot(gs.PLOT_Y, self.scan.motor._name)
        return [table, plot]


class _HardcodedMotorStepScan(_PrimitiveScan):
    # Subclasses must define self._motor as a property.

    def __call__(self, start, finish, intervals, time, subs=None,
                 **kwargs):
        original_times = _set_acquire_time(time)
        result = super().__call__(self._motor, start, finish,
                                  intervals + 1, subs=subs, **kwargs)
        _unset_acquire_time(original_times)
        return result

    @property
    def subs(self):
        table = LiveTable([self.scan.motor] + gs.TABLE_COLS)
        plot = LivePlot(gs.PLOT_Y, self.scan.motor._name)
        return [table, plot]


### Counts (p. 140) ###


class Count(_PrimitiveScan):
    "ct"
    _scan_class = scans.Count

    def __call__(self, time=None, subs=None, **kwargs):
        if subs is None:
            table = LiveTable(gs.TABLE_COLS)
            subs = [table]
        original_times = _set_acquire_time(time)
        result = super().__call__(subs=subs, **kwargs)
        _unset_acquire_time(original_times)
        return result


### Motor Scans (p. 146) ###


class AbsoluteScan(_StepScan):
    "ascan"
    _scan_class = scans.AbsScan


class OuterProductAbsoluteScan(_OuterProductScan):
    "mesh"
    _scan_class = scans.OuterProductAbsScan


class InnerProductAbsoluteScan(_InnerProductScan):
    "a2scan, a3scan, etc."
    _scan_class = scans.InnerProductAbsScan


class DeltaScan(_StepScan):
    "dscan (also known as lup)"
    _scan_class = scans.DeltaScan


class InnerProductDeltaScan(_InnerProductScan):
    "d2scan, d3scan, etc."
    _scan_class = scans.InnerProductDeltaScan


class ThetaTwoThetaScan(_InnerProductScan):
    "th2th"
    _scan_class = scans.InnerProductDeltaScan

    def __call__(self, start, finish, intervals, time=None, **kwargs):
        TTH_MOTOR = gs.TTH_MOTOR
        TH_MOTOR = gs.TH_MOTOR
        original_times = _set_acquire_time(time)
        result = super().__call__(TTH_MOTOR, start, finish,
                                  TH_MOTOR, start/2, finish/2,
                                  intervals, time, **kwargs)
        _unset_acquire_time(original_times)


### Temperature Scans (p. 148) ###


class _TemperatureScan(_HardcodedMotorStepScan):

    def __call__(self, start, finish, intervals, time=None, sleep=0,
                 **kwargs):
        self._sleep = sleep
        original_times = _set_acquire_time(time)
        self._motor.settle_time = sleep
        result = super().__call__(self._motor, start, finish, intervals + 1,
                                  **kwargs)
        _unset_acquire_time(original_times)
        return result

    @property
    def _motor(self):
        from bluesky.standard_config import gs
        return gs.TEMP_CONTROLLER


class AbsoluteTemperatureScan(_TemperatureScan):
    "tscan"
    _scan_class = scans.AbsScan


class DeltaTemperatureScan(_TemperatureScan):
    "dtscan"
    _scan_class = scans.DeltaScan


### Basic Reciprocal Space Scans (p. 147) ###


class HScan(_HardcodedMotorStepScan):
    "hscan"
    _scan_class = scans.AbsScan

    @property
    def _motor(self):
        from bluesky.standard_config import gs
        return gs.H_MOTOR


class KScan(_HardcodedMotorStepScan):
    "kscan"
    _scan_class = scans.AbsScan

    @property
    def _motor(self):
        from bluesky.standard_config import gs
        return gs.K_MOTOR


class LScan(_HardcodedMotorStepScan):
    "lscan"
    _scan_class = scans.AbsScan

    @property
    def _motor(self):
        from bluesky.standard_config import gs
        return gs.L_MOTOR


class OuterProductHKLScan(_PrimitiveScan):
    "hklmesh"
    _scan_class = scans.OuterProductAbsScan

    def __call__(self, Q1, start1, finish1, intervals1, Q2, start2, finish2,
                 intervals2, time=None, **kwargs):
        # To be clear, like all other functions in this module, this
        # eye-gouging API is for compatbility with SPEC, not the author's
        # idea of good Python code.
        from bluesky.standard_config import gs
        H_MOTOR = gs.H_MOTOR
        K_MOTOR = gs.K_MOTOR
        L_MOTOR = gs.L_MOTOR
        original_times = _set_acquire_time(time)
        _motor_mapping = {'H': H_MOTOR, 'K': K_MOTOR, 'L': L_MOTOR}
        motor1 = _motor_mapping[Q1]
        motor2 = _motor_mapping[Q2]
        # Note that intervals + 1 is handled in the base class.
        result = super().__call__(motor1, start1, finish1, intervals1,
                                  motor2, start2, finish2, intervals2,
                                  **kwargs)
        _unset_acquire_time(original_times)
        return result


class InnerProductHKLScan(_PrimitiveScan):
    "hklscan"
    _scan_class = scans.InnerProductAbsScan

    def __call__(self, start_h, finish_h, start_k, finish_k, start_l,
                 finish_l, intervals, time=None, **kwargs):
        from bluesky.standard_config import gs
        H_MOTOR = gs.H_MOTOR
        K_MOTOR = gs.K_MOTOR
        L_MOTOR = gs.L_MOTOR
        original_times = _set_acquire_time(time)
        result = super().__call__(intervals, start_h, finish_h, start_k,
                                  finish_k, start_l, finish_l, **kwargs)
        _unset_acquire_time(original_times)


### Special Reciprocal Space Scans ###

# TODO:
# klradial
# hlradial
# hkradial
# klcircle
# hlcircle
# hkcircle


class Tweak(_PrimitiveScan):
    "tw"
    _scan_class = scans.Tweak

    def __call__(motor, step, **kwargs):
        from bluesky.standard_config import gs
        MASTER_DET = gs.MASTER_DET
        MASTER_DET_FIELD = gs.MASTER_DET_FIELD
        return super().__call__(MASTER_DET, MASTER_DET_FIELD, motor,
                                step, **kwargs)


def _set_acquire_time(time):
    from bluesky.standard_config import gs
    if time is None:
        time = gs.COUNT_TIME
    original_times = {}
    for det in gs.DETS:
        if hasattr(det, 'count_time'):
            original_times[det] = det.count_time
            det.count_time = time
    return original_times


def _unset_acquire_time(original_times):
    for det, time in original_times.items():
        det.count_time = time
