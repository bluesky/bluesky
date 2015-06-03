"""Tests for, uh, the state machine of the run engine"""
from nose.tools import raises, assert_raises
from bluesky import RunEngineStateMachine
from super_state_machine.errors import TransitionError
from .utils import goto_state, define_state_machine_transitions


sm = RunEngineStateMachine()


def _transitioner(state_machine, state_from, state_to, valid_transition):
    print('current state = [[%s]]' % state_machine.state)
    print('attempting to go to state [[%s]]' % state_to)
    if valid_transition:
        state_machine.set_(state_to)
    else:
        assert_raises(TransitionError, state_machine.set_, state_to)


def test_state_machine_transitions():
    transition_map = define_state_machine_transitions(sm)
    for state_to, transitions in transition_map.items():
        # set the state correctly
        for state_from, valid_transition in transitions:
            goto_state(sm, state_from)
            yield _transitioner, sm, state_from, state_to, valid_transition
