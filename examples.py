from bs import Msg
from collections import deque
import numpy as np
from lmfit.models import GaussianModel, LinearModel


def MoveRead_gen(motor, detector):
    try:
        for j in range(10):
            yield Msg('create')
            yield Msg('set', motor, {'x': j})
            yield Msg('trigger', motor)
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
