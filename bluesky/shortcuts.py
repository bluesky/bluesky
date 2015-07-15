"""
These "scans" bundle a Message generator with an instance of the RunEngine,
combining two separate concepts -- instructions and execution -- into one
object. This makes the interface less flexible and somewhat less "Pythonic"
but more condensed.

This module is meant to be run in a namespace where several global
variables have been defined. If some variables are left undefined, the
associated scans will be not usable.

    dets  # list of detectors
    master_det  # detector to use for tw
    master_det_field  # detector field to use for tw
    h_motor
    k_motor
    l_motor
    th_motor
    tth_motor
    temp_controller

Page numbers in the code comments refer to the SPEC manual at
http://www.certif.com/downloads/css_docs/spec_man.pdf
"""
from inspect import signature
from bluesky import scans
from boltons.iterutils import chunked


class _PrimitiveScan:

    def __init__(self, RE):
        self.RE = RE
        self.subs = None
        self.params = list(signature(self._scan_class).parameters.keys())
        RE_params = list(signature(RE.__call__).parameters.keys())
        if set(RE_params) & set(self.params):
            raise AssertionError("The names of the scan's arguments clash "
                                 "the RunEngine arguments. Use different "
                                 "names. Avoid: {0}".format(RE_params))

    def __call__(self, *args, subs=None, **kwargs):
        global dets
        scan_kwargs = dict()
        # Any kwargs valid for the scan go to the scan, not the RE.
        for k, v in kwargs.items():
            if k in self.params:
                scan_kwargs[k] = kwargs.pop(k)
        scan = self._scan_class(dets, *args, **scan_kwargs)
        scan.subs = self.subs
        self.RE(scan, **kwargs)  # to pass to RE, you must use kwargs


### Mid-level base classes ###

# These are responsible for popping off the time arg and adjusting the
# interval. SPEC counts "bonds;" idiomatic Python counts "sites."

class _OuterProductScan(_PrimitiveScan):
    def __call__(self, *args, **kwargs):
        time = args.pop(-1)
        _set_acquire_time(time)
        for i, _ in enumerate(chunked(args, 4)):
            # intervals -> intervals + 1
            args[4*i + 3] += 1
        super.__call__(*args, **kwargs)

class _InnerProductScan(_PrimitiveScan):
    def __call__(self, *args, **kwargs):
        time = args.pop(-1)
        intervals = args.pop(-1) + 1
        _set_acquire_time(time)
        super.__call__(intervals, *args, **kwargs)


class _StepScan(_PrimitiveScan):

    def __call__(self, motor, start, finish, intervals, time, **kwargs):
        _set_acquire_time(time)
        super.__call__(motor, start, finish, intervals + 1, **kwargs)


class _HardcodedMotorStepScan(_PrimitiveScan):
    # Subclasses must define self._motor as a property.

    def __call__(self, start, finish, intervals, time, **kwargs):
        _set_acquire_time(time)
        super.__call__(self._motor, start, finish, intervals + 1, **kwargs)


### Counts (p. 140) ###


class Count(_PrimitiveScan):
    "ct"
    _scan_class = scans.Count
    
    def __call__(self, time=1, **kwargs):
        _set_acquire_time(time)
        super.__call__(**kwargs)


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

    def __call__(self, start, finish, intervals, time, **kwargs):
        global th_motor
        global tth_motor
        super.__call__(tth_motor, start, finish,
                       th_motor, start/2, finish/2, intervals, time, **kwargs)


### Temperature Scans (p. 148) ###


class _TemperatureScan(_HardcodedMotorStepScan):

    def __call__(self, start, finish, intervals, time, sleep=0, **kwargs):
        self._sleep = sleep
        _set_acquire_time(time)
        self._motor.settle_time = sleep
        super.__call__(self._motor, start, finish, intervals + 1, **kwargs)

    @property
    def _motor(self):
        global temp_controller
        return temp_controller


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
        global h_motor
        return h_motor


class KScan(_HardcodedMotorStepScan):
    "kscan"
    _scan_class = scans.AbsScan

    @property
    def _motor(self):
        global l_motor
        return l_motor


class LScan(_HardcodedMotorStepScan):
    "lscan"
    _scan_class = scans.AbsScan

    @property
    def _motor(self):
        global l_motor
        return l_motor


class OuterProductHKLScan(_PrimitiveScan):
    "hklmesh"
    _scan_class = scans.OuterProductAbsScan

    def __call__(self, Q1, start1, finish1, intervals1, Q2, start2, finish2,
                 intervals2, time, **kwargs):
        # To be clear, like all other functions in this module, this
        # eye-gouging API is for compatbility with SPEC, not the author's
        # idea of good Python code.
        global h_motor, k_motor, l_motor
        _set_acquire_time(time)
        _motor_mapping = {'H': h_motor, 'K': k_motor, 'L': l_motor}
        motor1 = _motor_mapping[Q1]
        motor2 = _motor_mapping[Q2]
        # Note that intervals + 1 is handled in the base class.
        super.__call__(motor1, start1, finish1, intervals1,
                       motor2, start2, finish2, intervals2, **kwargs)


class InnerProductHKLScan(_PrimitiveScan):
    "hklscan"
    _scan_class = scans.InnerProductAbsScan

    def __call__(self, start_h, finish_h, start_k, finish_k, start_l, finish_l,
                 intervals, time, **kwargs):
        global h_motor, k_motor, l_motor
        _set_acquire_time(time)
        super.__call__(self, intervals, start_h, finish_h, start_k, finish_k,
                       start_l, finish_l, **kwargs)


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
        global master_det, master_det_field
        super().__call__(master_det, master_det_field, motor, step, **kwargs)


def _set_acquire_time(time):
    global dets
    for det in dets:
        det.acquire_time = time
