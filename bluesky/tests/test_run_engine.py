import asyncio
import pytest
from bluesky.run_engine import (RunEngine, RunEngineStateMachine,
                                TransitionError)
from bluesky import Msg
from bluesky.examples import det, Mover

def test_states():
    assert RunEngineStateMachine.States.states() == ['idle', 'running', 'paused']


def test_verbose(fresh_RE):
    fresh_RE.verbose = True
    fresh_RE.verbose == True
    # Emit all four kinds of document, exercising the logging.
    fresh_RE([Msg('open_run'), Msg('create'), Msg('read', det), Msg('save'),
              Msg('close_run')])


def test_reset(fresh_RE):
    fresh_RE([Msg('open_run'), Msg('pause')])
    assert fresh_RE._run_start_uid != None
    fresh_RE.reset()
    assert fresh_RE._run_start_uid == None


def test_running_from_paused_state_raises(fresh_RE):
    fresh_RE([Msg('pause')])
    assert fresh_RE.state == 'paused'
    with pytest.raises(RuntimeError):
        fresh_RE([Msg('null')])
    fresh_RE.resume()
    assert fresh_RE.state == 'idle'
    fresh_RE([Msg('null')])


def test_resuming_from_idle_state_raises(fresh_RE):
    with pytest.raises(RuntimeError):
        fresh_RE.resume()
    fresh_RE([Msg('pause')])
    assert fresh_RE.state == 'paused'
    fresh_RE.resume()
    assert fresh_RE.state == 'idle'
    with pytest.raises(RuntimeError):
        fresh_RE.resume()


def test_stopping_from_idle_state_raises(fresh_RE):
    with pytest.raises(TransitionError):
        fresh_RE.stop()


def test_aborting_from_idle_state_raises(fresh_RE):
    with pytest.raises(TransitionError):
        fresh_RE.abort()


def test_register(fresh_RE):
    mutable = {}
    fresh_RE.verbose = True

    @asyncio.coroutine
    def func(msg):
        mutable['flag'] = True

    def plan():
        yield Msg('custom-command')

    fresh_RE.register_command('custom-command', func)
    fresh_RE(plan())
    assert 'flag' in mutable
    # Unregister command; now the Msg should not be recognized.
    fresh_RE.unregister_command('custom-command')
    with pytest.raises(KeyError):
        fresh_RE([Msg('custom-command')])


def test_stop_motors_and_log_any_errors(fresh_RE):
    # test that if stopping one motor raises an error, we can carry on
    stopped = {}

    class MoverWithFlag(Mover):
        def stop(self):
            print('HELLO')
            stopped['non_broken'] = True

    class BrokenMoverWithFlag(Mover):
        def stop(self):
            print('HELLO')
            stopped['broken'] = True
            raise Exception


    motor = MoverWithFlag('a', ['a'])
    broken_motor = BrokenMoverWithFlag('b', ['b'])

    fresh_RE([Msg('set', broken_motor, 1), Msg('set', motor, 1), Msg('pause')])
    assert 'broken' in stopped
    assert 'non_broken' in stopped
    fresh_RE.stop()

    fresh_RE([Msg('set', motor, 1), Msg('set', broken_motor, 1), Msg('pause')])
    assert 'broken' in stopped
    assert 'non_broken' in stopped
    fresh_RE.stop()
