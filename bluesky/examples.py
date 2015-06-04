import threading
import time as ttime
from . import Msg
from .run_engine import Msg
from collections import deque
import numpy as np
from lmfit.models import GaussianModel, LinearModel


def simple_scan(motor):
    yield Msg('set', motor, 5)
    yield Msg('read', motor)


def conditional_break(motor, det, threshold):
    """Set, trigger, read until the detector reads intensity < threshold"""
    i = 0
    while True:
        print("LOOP %d" % i)
        yield Msg('set', motor, i)
        yield Msg('trigger', det)
        reading = yield Msg('read', det)
        if reading['intensity']['value'] < threshold:
            print('DONE')
            break
        i += 1


def sleepy(motor, det):
    "Set, trigger motor, sleep for a fixed time, trigger detector, read"
    yield Msg('set', motor, 5)
    yield Msg('sleep', None, 2)  # units: seconds
    yield Msg('trigger', det)
    yield Msg('read', det)

def checkpoint_forever():
    # simplest pauseable scan
    while True:
        ttime.sleep(0.1)
        yield Msg('checkpoint')


def wait_one(motor, det):
    "Set, trigger, read"
    yield Msg('set', motor, 5, block_group='A')  # Add to group 'A'.
    yield Msg('wait', None, 'A')  # Wait for everything in group 'A' to finish.
    yield Msg('trigger', det)
    yield Msg('read', det)


def wait_multiple(motors, det):
    "Set motors, trigger all motors, wait for all motors to move."
    for motor in motors:
        yield Msg('set', motor, 5, block_group='A')
    # Wait for everything in group 'A' to report done.
    yield Msg('wait', None, 'A')
    yield Msg('trigger', det)
    yield Msg('read', det)


def wait_complex(motors, det):
    "Set motors, trigger motors, wait for all motors to move in groups."
    # Same as above...
    for motor in motors[:-1]:
        yield Msg('set', motor, 5, block_group='A')

    # ...but put the last motor is separate group.
    yield Msg('set', motors[-1], 5, block_group='B')
    # Wait for everything in group 'A' to report done.
    yield Msg('wait', None, 'A')
    yield Msg('trigger', det)
    yield Msg('read', det)

    # Wait for everything in group 'B' to report done.
    yield Msg('wait', None, 'B')
    yield Msg('trigger', det)
    yield Msg('read', det)


def conditional_pause(motor, det, hard, include_checkpoint):
    for i in range(5):
        if include_checkpoint:
            yield Msg('checkpoint')
        yield Msg('set', motor, i)
        yield Msg('trigger', det)
        reading = yield Msg('read', det)
        if reading['intensity']['value'] < 0.2:
            yield Msg('pause', hard=hard)
        yield Msg('set', motor, i + 0.5)


class PausingAgent:
    def __init__(self, RE, name):
        self.RE = RE
        self.name = name

    def issue_request(self, hard, delay=0):
        def callback():
            return self.permission

        def requester():
            ttime.sleep(delay)
            self.permission = False
            self.RE.request_pause(hard, self.name, callback)

        thread = threading.Thread(target=requester)
        thread.start()

    def revoke_request(self, delay=0):
        ttime.sleep(delay)
        self.permission = True

def panic_timer(RE, delay):
    def f():
        ttime.sleep(delay)
        RE.panic()

    thread = threading.Thread(target=f)
    thread.start()

def simple_scan_saving(motor, det):
    "Set, trigger, read"
    yield Msg('create')
    yield Msg('set', motor, 5)
    yield Msg('read', motor)
    yield Msg('trigger', det)
    yield Msg('read', det)
    yield Msg('save')


def stepscan(motor, det):
    for i in range(-5, 5):
        yield Msg('create')
        yield Msg('set', motor, i)
        yield Msg('trigger', det)
        yield Msg('read', motor)
        yield Msg('read', det)
        yield Msg('save')


def live_scalar_plotter(ax, y, x):
    x_data, y_data = [], []
    line, = ax.plot([], [], 'ro', markersize=10)

    def update_plot(doc):
        # Update with the latest data.
        x_data.append(doc['data'][x])
        y_data.append(doc['data'][y])
        line.set_data(x_data, y_data)
        # Rescale and redraw.
        ax.relim(visible_only=True)
        ax.autoscale_view(tight=True)
        ax.figure.canvas.draw()
    return update_plot


def MoveRead_gen(motor, detector):
    try:
        for j in range(10):
            yield Msg('create')
            yield Msg('set', motor, {'x': j})
            yield Msg('trigger', detector)
            yield Msg('read', detector)
            yield Msg('read', motor)
            yield Msg('save')
    finally:
        print('Generator finished')


def SynGauss_gen(syngaus, motor_steps, motor_limit=None):
    try:
        for x in motor_steps:
            yield Msg('create')
            yield Msg('set', syngaus, {syngaus.motor_name: x})
            yield Msg('trigger', syngaus)
            yield Msg('sleep', None, .1)
            ret = yield Msg('read', syngaus)
            yield Msg('save')
            if motor_limit is not None:
                if ret[syngaus.motor_name] > motor_limit:
                    break
    finally:
        print('generator finished')


def find_center_gen(syngaus, initial_center, initial_width,
                    output_mutable):
    tol = .01
    seen_x = deque()
    seen_y = deque()

    for x in np.linspace(initial_center - initial_width,
                         initial_center + initial_center,
                         5, endpoint=True):
        yield Msg('set', syngaus, {syngaus.motor_name: x})
        yield Msg('trigger', syngaus)
        yield Msg('sleep', None, .1,)
        ret = yield Msg('read', syngaus)
        seen_x.append(ret[syngaus.motor_name])
        seen_y.append(ret[syngaus.det_name])
    model = GaussianModel() + LinearModel()
    guesses = {'amplitude': np.max(seen_y),
               'center': initial_center,
               'sigma': initial_width,
               'slope': 0, 'intercept': 0}
    while True:
        x = np.asarray(seen_x)
        y = np.asarray(seen_y)
        res = model.fit(y, x=x, **guesses)
        old_guess = guesses
        guesses = res.values

        if np.abs(old_guess['center'] - guesses['center']) < tol:
            break

        yield Msg('set', syngaus, {syngaus.motor_name: guesses['center']})
        yield Msg('trigger', syngaus)
        yield Msg('sleep', None, .1)
        ret = yield Msg('read', syngaus)
        seen_x.append(ret[syngaus.motor_name])
        seen_y.append(ret[syngaus.det_name])

    output_mutable.update(guesses)


def fly_gen(flyer):
    yield Msg('kickoff', flyer)
    yield Msg('collect', flyer)
    yield Msg('kickoff', flyer)
    yield Msg('collect', flyer)


def adaptive_scan(motor, detector, motor_name, detector_name, start,
                  stop, min_step, max_step, target_dI):
    next_pos = start
    step = (max_step - min_step) / 2

    past_I = None
    cur_I = None
    while next_pos < stop:
        yield Msg('set', motor, {motor_name: next_pos})
        yield Msg('sleep', None, .1)
        yield Msg('create')
        yield Msg('trigger', motor)
        yield Msg('trigger', detector)
        cur_det = yield Msg('read', detector)
        yield Msg('read', motor)
        yield Msg('save')

        cur_I = cur_det[detector_name]['value']

        # special case first first loop
        if past_I is None:
            past_I = cur_I
            next_pos += step
            continue

        dI = np.abs(cur_I - past_I)

        slope = dI / step

        new_step = np.clip(target_dI / slope, min_step, max_step)
        # if we over stepped, go back and try again
        if new_step < step * 0.8:
            next_pos -= step
        else:
            past_I = cur_I
        print(step, new_step)
        step = new_step

        next_pos += step
        print('********', next_pos, step, past_I, slope, dI,  '********')
