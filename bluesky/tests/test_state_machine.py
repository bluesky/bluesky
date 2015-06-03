"""Tests for, uh, the state machine of the run engine"""
from nose.tools import raises
from nose.plugins.logcapture import log
from bluesky import RunEngineStateMachine
from super_state_machine.errors import TransitionError

# path to various states
idle = ['panicked', 'idle']
running = idle + ['running']
soft_pausing = running + ['soft_pausing']
hard_pausing = running + ['hard_pausing']
paused = soft_pausing + ['paused']
aborting = paused + ['aborting']
panicked = ['panicked']

state_paths = {'idle': idle,
               'running': running,
               'aborting': aborting,
               'soft_pausing': soft_pausing,
               'hard_pausing': hard_pausing,
               'paused': paused,
               'panicked': panicked}


@raises(TransitionError)
def _invalid_state_transitioner(state_machine, state_to):
    print('current state = [[%s]]' % state_machine.state)
    print('attempting to go to state [[%s]]' % state_to)
    state_machine.set_(state_to)

def _valid_state_transitioner(state_machine, state_to):
    print('current state = [[%s]]' % state_machine.state)
    print('attempting to go to state [[%s]]' % state_to)
    state_machine.set_(state_to)

def test_state_machine_transitions():
    sm = RunEngineStateMachine()
    all_states = set(sm.States.states())
    transition_map = []
    for tup in sm.Meta.named_transitions:
        name = tup[0]
        to_state = tup[1]
        try:
            valid_from_states = set(tup[2])
        except IndexError:
            valid_from_states = all_states
        for from_state in all_states:
            allowed_transition = True
            if from_state not in valid_from_states:
                allowed_transition = False

            transition_map.append((to_state, from_state,
                                   allowed_transition))
    for state_to, state_from, valid_transition in transition_map:
        # set the state correctly
        get_to_valid_state = state_paths[state_from]
        print("\nNavigating state machine to state [[%s]]" % state_from)
        for state in get_to_valid_state:
            print('current state = [[%s]]' % sm.state)
            print('attempting to go to state [[%s]]' % state)
            sm.set_(state)
        if valid_transition:
            yield _valid_state_transitioner, sm, state_to
        else:
            yield _invalid_state_transitioner, sm, state_to
