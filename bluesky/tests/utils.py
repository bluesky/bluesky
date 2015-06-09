from collections import defaultdict
from itertools import product
from bluesky.run_engine import RunEngine


# path to various states
idle = ['panicked', 'idle']
running = idle + ['running']
soft_pausing = running + ['soft_pausing']
hard_pausing = running + ['hard_pausing']
paused = soft_pausing + ['paused']
aborting = paused + ['aborting']
panicked = ['panicked']

state_paths_dict = {
    'idle': idle,
    'running': running,
    'aborting': aborting,
    'soft_pausing': soft_pausing,
    'hard_pausing': hard_pausing,
    'paused': paused,
    'panicked': panicked}


def goto_state(state_machine, desired_state):
    print("\nNavigating state machine to state [[%s]]" % desired_state)
    for state in state_paths_dict[desired_state]:
        print('current state = [[%s]]' % state_machine.state)
        print('attempting to go to state [[%s]]' % state)
        state_machine.set_(state)


def tautologically_define_state_machine_transitions(state_machine):
    """Create a mapping of all transitions in ``state_machine``

    Parameters
    ----------
    state_machine : super_state_machine.machines.StateMachine
        The state machine you want a complete map of

    Returns
    -------
    dict
        Dictionary of all transitions in ``state_machine``
        Structured as
        {from_state1: [(to_state, allowed), ...],
         from_state2: [(to_state, allowed), ...],
        }
        where
        - ``allowed`` is a boolean
        - ``from_stateN`` is a string
        - ``to_state`` is a string
    """
    transitions_as_enum = state_machine.__class__._meta['transitions']
    transitions_as_names = {
        to_state.value: [from_state.value for from_state in from_states]
        for to_state, from_states in transitions_as_enum.items()}
    transition_map = defaultdict(list)
    all_states = set(state_machine.States.states())
    for to_state, from_states in transitions_as_names.items():
        for from_state in all_states:
            allowed = True
            if from_state not in from_states:
                allowed = False
            transition_map[to_state].append((from_state, allowed))

    return transition_map


def define_state_machine_transitions_from_class(state_machine):
    all_states = set(state_machine.States.states())
    transitions = {from_state: set(to_states) for from_state, to_states
                   in state_machine.Meta.transitions.items()}
    for transition in state_machine.Meta.named_transitions:
        try:
            transition_name, to_state, from_states = transition
        except ValueError:
            transition_name, to_state = transition
            from_states = all_states
        for from_state in from_states:
            transitions[from_state].add(to_state)
    transition_map = defaultdict(list)
    for from_state, to_states in transitions.items():
        for to_state in all_states:
            allowed = True
            if to_state not in to_states:
                allowed = False
            transition_map[from_state].append((to_state, allowed))
    return transition_map


def setup_run_engine():
    RE = RunEngine()
    RE.memory['owner'] = 'test_owner'
    RE.memory['group'] = 'Grant No. 12345'
    RE.memory['config'] = {'detector_model': 'XYZ', 'pxiel_size': 10}
    RE.memory['beamline_id'] = 'test_beamline'
    return RE


if __name__ == "__main__":
    from bluesky import RunEngineStateMachine
    sm = RunEngineStateMachine()
    transition_map = dict(tautologically_define_state_machine_transitions(sm))
    from pprint import pprint
    pprint(transition_map)
