import warnings
import time as ttime
import numpy as np
from .utils import Msg


try:
    import ophyd.sim  # noqa: F401
except ImportError:
    raise ImportError("""
The simulated hardware objects in the bluesky.examples module have been
moved to ophyd.sim. A version of ophyd v0.8.0 or greater is required.""")
else:
    warnings.warn("""
The simulated hardware objects in the bluesky.examples module have been
moved to ophyd.sim. Update imports to suppress this warning.
    """, stacklevel=2)
    from ophyd.sim import *  # noqa: F403, F401


# These are very old, raw implementations of 'plans' the pre-date the
# bluesky.plans module. They are not documented or (we hope) used in
# production, but they are still used in tests.

def simple_scan(motor):
    yield Msg('open_run')
    yield Msg('set', motor, 5)
    yield Msg('read', motor)
    yield Msg('close_run')


def conditional_break(det, motor, threshold):
    """Set, trigger, read until the detector reads intensity < threshold"""
    i = 0
    yield Msg('open_run')
    while True:
        print("LOOP %d" % i)
        yield Msg('set', motor, i)
        yield Msg('trigger', det)
        reading = yield Msg('read', det)
        if reading[det.name]['value'] < threshold:
            print('DONE')
            yield Msg('close_run')
            break
        i += 1


def sleepy(det, motor):
    "Set, trigger motor, sleep for a fixed time, trigger detector, read"
    yield Msg('open_run')
    yield Msg('set', motor, 5)
    yield Msg('sleep', None, 2)  # units: seconds
    yield Msg('trigger', det)
    yield Msg('read', det)
    yield Msg('close_run')


def do_nothing(timeout=5):
    "Generate 'checkpoint' messages until timeout."
    t = ttime.time()
    yield Msg('open_run')
    while True:
        if ttime.time() > t + timeout:
            break
        ttime.sleep(0.1)
        yield Msg('checkpoint')
    yield Msg('close_run')


def checkpoint_forever():
    # simplest pauseable scan
    yield Msg('open_run')
    while True:
        ttime.sleep(0.1)
        yield Msg('checkpoint')
    yield Msg('close_run')


def wait_one(det, motor):
    "Set, trigger, read"
    yield Msg('open_run')
    yield Msg('set', motor, 5, group='A')  # Add to group 'A'.
    yield Msg('wait', None,
              group='A')  # Wait for everything in group 'A' to finish.
    yield Msg('trigger', det)
    yield Msg('read', det)
    yield Msg('close_run')


def wait_multiple(det, motors):
    "Set motors, trigger all motors, wait for all motors to move."
    yield Msg('open_run')
    for motor in motors:
        yield Msg('set', motor, 5, group='A')
    # Wait for everything in group 'A' to report done.
    yield Msg('wait', None, group='A')
    yield Msg('trigger', det)
    yield Msg('read', det)
    yield Msg('close_run')


def wait_complex(det, motors):
    "Set motors, trigger motors, wait for all motors to move in groups."
    # Same as above...
    yield Msg('open_run')
    for motor in motors[:-1]:
        yield Msg('set', motor, 5, group='A')

    # ...but put the last motor is separate group.
    yield Msg('set', motors[-1], 5, group='B')
    # Wait for everything in group 'A' to report done.
    yield Msg('wait', None, group='A')
    yield Msg('trigger', det)
    yield Msg('read', det)

    # Wait for everything in group 'B' to report done.
    yield Msg('wait', None, group='B')
    yield Msg('trigger', det)
    yield Msg('read', det)
    yield Msg('close_run')


def conditional_pause(det, motor, defer, include_checkpoint):
    yield Msg('open_run')
    for i in range(5):
        if include_checkpoint:
            yield Msg('checkpoint')
        yield Msg('set', motor, i)
        yield Msg('trigger', det)
        reading = yield Msg('read', det)
        if reading['det']['value'] < 0.2:
            yield Msg('pause', defer=defer)
        print("I'm not pausing yet.")
    yield Msg('close_run')


def simple_scan_saving(det, motor):
    "Set, trigger, read"
    yield Msg('open_run')
    yield Msg('declare_stream', None, motor, det, name='primary')
    yield Msg('create', name='primary')
    yield Msg('set', motor, 5)
    yield Msg('read', motor)
    yield Msg('trigger', det)
    yield Msg('read', det)
    yield Msg('save')
    yield Msg('close_run')


def stepscan(det, motor):
    yield Msg('open_run')
    yield Msg('declare_stream', None, motor, det, name='primary')
    for i in range(-5, 5):
        yield Msg('create', name='primary')
        yield Msg('set', motor, i)
        yield Msg('trigger', det)
        yield Msg('read', motor)
        yield Msg('read', det)
        yield Msg('save')
    yield Msg('close_run')


def cautious_stepscan(det, motor):
    yield Msg('open_run')
    yield Msg('declare_stream', None, motor, det, name='primary')
    for i in range(-5, 5):
        yield Msg('checkpoint')
        yield Msg('create', name='primary')
        yield Msg('set', motor, i)
        yield Msg('trigger', det)
        ret_m = yield Msg('read', motor)
        ret_d = yield Msg('read', det)
        yield Msg('save')
        print("Value at {m} is {d}. Pausing.".format(
            m=ret_m[motor.name]['value'], d=ret_d[det.name]['value']))
        yield Msg('pause', None, defer=True)
    yield Msg('close_run')


def fly_gen(flyer):
    yield Msg('open_run')
    yield Msg('kickoff', flyer, group='fly-kickoff')
    yield Msg('wait', None, group='fly-kickoff')
    yield Msg('complete', flyer, group='fly-complete')
    yield Msg('wait', None, group='fly-complete')
    yield Msg('collect', flyer)
    yield Msg('kickoff', flyer, group='fly-kickoff2')
    yield Msg('wait', None, group='fly-kickoff2')
    yield Msg('complete', flyer, group='fly-complete2')
    yield Msg('wait', None, group='fly-complete2')
    yield Msg('collect', flyer)
    yield Msg('close_run')


def multi_sample_temperature_ramp(detector, sample_names, sample_positions,
                                  scan_motor, start, stop, step,
                                  temp_controller, tstart, tstop, tstep):
    def read_and_store_temp():
        yield Msg('create', name='primary')
        yield Msg('read', temp_controller)
        yield Msg('save')

    peak_centers = [-1 + 3 * n for n in range(len((sample_names)))]
    detector.noise = True

    for idx, temp in enumerate(np.arange(tstart, tstop, tstep)):
        # todo would be cute to have the temperature reduce peak noise
        yield Msg('set', temp_controller, temp)
        for sample, sample_position, peak_pos in zip(sample_names,
                                                     sample_positions,
                                                     peak_centers):
            yield Msg('open_run', sample_name=sample, target_temp=temp)
            detector.center = peak_pos
            detector.sigma = .5 + .25 * idx
            detector.noise_factor = .05 + idx * 0.1
            yield Msg('declare_stream', None, scan_motor, detector, temp_controller, name='primary')
            for scan_pos in np.arange(start, stop, step):
                yield Msg('set', scan_motor, scan_pos)
                # be super paranoid about the temperature. Grab it before and
                # after each trigger!
                # Capturing the temperature data before and after each
                # trigger is resulting in unintended behavior. Uncomment the
                # two `yield from` statements and run this example to see
                # what I'm talking about.
                # yield from read_and_store_temp()
                yield Msg('trigger', detector)
                # yield from read_and_store_temp()
                yield Msg('create', name='primary')
                yield Msg('read', scan_motor)
                yield Msg('read', detector)
                yield Msg('read', temp_controller)
                # yield Msg('sleep', None, .1)
                yield Msg('save')
            # generate the end of the run document
            yield Msg('close_run')
