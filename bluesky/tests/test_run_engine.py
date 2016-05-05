import asyncio
import pytest
from bluesky.run_engine import (RunEngine, RunEngineStateMachine,
                                TransitionError, IllegalMessageSequence)
from bluesky import Msg
from bluesky.examples import det, Mover

class DummyFlyer:
    def kickoff(self):
        pass
    def describe(self):
        return []
    def collect(self):
        return {}

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
            stopped[self.name] = True

    class BrokenMoverWithFlag(Mover):
        def stop(self):
            stopped[self.name] = True
            raise Exception


    motor = MoverWithFlag('a', ['a'])
    broken_motor = BrokenMoverWithFlag('b', ['b'])

    fresh_RE([Msg('set', broken_motor, 1), Msg('set', motor, 1), Msg('pause')])
    assert 'a' in stopped
    assert 'b' in stopped
    fresh_RE.stop()

    fresh_RE([Msg('set', motor, 1), Msg('set', broken_motor, 1), Msg('pause')])
    assert 'a' in stopped
    assert 'b' in stopped
    fresh_RE.stop()


def test_collect_uncollected_and_log_any_errors(fresh_RE):
    # test that if stopping one motor raises an error, we can carry on
    collected = {}

    class DummyFlyerWithFlag(DummyFlyer):
        def collect(self):
            collected[self.name] = True
            super().collect()

    class BrokenDummyFlyerWithFlag(DummyFlyerWithFlag):
        def collect(self):
            super().collect()
            raise Exception


    flyer1 = DummyFlyerWithFlag()
    flyer1.name = 'flyer1'
    flyer2 = DummyFlyerWithFlag()
    flyer2.name = 'flyer2'

    collected.clear()
    fresh_RE([Msg('open_run'), Msg('kickoff', flyer1), Msg('kickoff', flyer2)])
    assert 'flyer1' in collected
    assert 'flyer2' in collected

    collected.clear()
    fresh_RE([Msg('open_run'), Msg('kickoff', flyer2), Msg('kickoff', flyer1)])
    assert 'flyer1' in collected
    assert 'flyer2' in collected


def test_unstage_and_log_errors(fresh_RE):
    unstaged = {}

    class MoverWithFlag(Mover):
        def unstage(self):
            unstaged[self.name] = True
            return [self]

    class BrokenMoverWithFlag(Mover):
        def unstage(self):
            unstaged[self.name] = True
            return [self]

    a = MoverWithFlag('a', ['a'])
    b = BrokenMoverWithFlag('b', ['b'])

    unstaged.clear()
    fresh_RE([Msg('stage', a), Msg('stage', b)])
    assert 'a' in unstaged
    assert 'b' in unstaged

    unstaged.clear()
    fresh_RE([Msg('stage', b), Msg('stage', a)])
    assert 'a' in unstaged
    assert 'b' in unstaged


def test_open_run_twice_is_illegal(fresh_RE):
    with pytest.raises(IllegalMessageSequence):
        fresh_RE([Msg('open_run'), Msg('open_run')])


def test_saving_without_an_open_bundle_is_illegal(fresh_RE):
    with pytest.raises(IllegalMessageSequence):
        fresh_RE([Msg('open_run'), Msg('save')])


def test_opening_a_bundle_without_a_run_is_illegal(fresh_RE):
    with pytest.raises(IllegalMessageSequence):
        fresh_RE([Msg('create')])


def test_checkpoint_inside_a_bundle_is_illegal(fresh_RE):
    with pytest.raises(IllegalMessageSequence):
        fresh_RE([Msg('open_run'), Msg('create'), Msg('checkpoint')])


def test_flying_outside_a_run_is_illegal(fresh_RE):

    flyer = DummyFlyer()

    # This is normal, legal usage.
    fresh_RE([Msg('open_run'), Msg('kickoff', flyer), Msg('collect', flyer),
              Msg('close_run')])

    # It is not legal to kickoff outside of a run.
    with pytest.raises(IllegalMessageSequence):
        fresh_RE([Msg('kickoff', flyer)])

    # And of course you can't collect a flyer you never kicked off.
    with pytest.raises(IllegalMessageSequence):
        fresh_RE([Msg('open_run'), Msg('collect', flyer), Msg('close_run')])


def test_empty_bundle(fresh_RE):
    mutable = {}

    def cb(name, doc):
        mutable['flag'] = True

    # In this case, an Event should be emitted.
    mutable.clear()
    fresh_RE([Msg('open_run'), Msg('create'), Msg('read', det), Msg('save')],
             subs={'event': cb})
    assert 'flag' in mutable

    # In this case, an Event should not be emitted because the bundle is
    # emtpy (i.e., there are no readings.)
    mutable.clear()
    fresh_RE([Msg('open_run'), Msg('create'), Msg('save')], subs={'event': cb})
    assert 'flag' not in mutable


def test_dispatcher_unsubscribe_all(fresh_RE):
    def count_callbacks(RE):
        return sum([len(cbs) for cbs in
                    RE.dispatcher.cb_registry.callbacks.values()])

    def cb(name, doc):
        pass

    fresh_RE.subscribe('all', cb)
    assert count_callbacks(fresh_RE) == 5 
    fresh_RE.dispatcher.unsubscribe_all()
    assert count_callbacks(fresh_RE) == 0


def test_stage_and_unstage_are_optional_methods(fresh_RE):
    class Dummy:
        pass

    dummy = Dummy()

    fresh_RE([Msg('stage', dummy), Msg('unstage', dummy)])
