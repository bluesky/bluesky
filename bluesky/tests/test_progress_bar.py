from bluesky.utils import ProgressBar, ProgressBarManager
from bluesky.plans import mv
from bluesky import RunEngine
from bluesky.examples import NullStatus, SimpleStatus, Mover
from collections import OrderedDict
import time


def test_status_without_watch():
    st = NullStatus()
    ProgressBar([st])


def test_status_with_name():
    st = SimpleStatus()
    pbar = ProgressBar([st])
    st._finished()

    st = SimpleStatus()
    pbar = ProgressBar([st])
    assert pbar.delay_draw == 0.2
    time.sleep(0.25)
    st._finished()

def test_mv_progress(fresh_RE):
    RE = fresh_RE
    RE.waiting_hook = ProgressBarManager()
    motor1 = Mover('motor1', OrderedDict([('motor1', lambda x: x),
                                        ('motor1_setpoint', lambda x: x)]),
                {'x': 0})
    motor2 = Mover('motor2', OrderedDict([('motor2', lambda x: x),
                                        ('motor2_setpoint', lambda x: x)]),
                {'x': 0})

    assert RE.waiting_hook.delay_draw == 0.2

    # moving time > delay_draw
    motor1._fake_sleep = 0.5
    motor1._fake_sleep = 0.5
    RE(mv(motor1, 0, motor2, 0))

    # moving time < delay_draw
    motor1._fake_sleep = 0.01
    motor1._fake_sleep = 0.01
    RE(mv(motor1, 0, motor2, 0))


def test_draw_before_update():
    class Status:
        done = False
        def watch(self, func):
            ...

    # Test that the default meter placeholder is valid to draw.
    pbar = ProgressBar([Status()])
    pbar.draw()
