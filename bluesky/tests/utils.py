
from collections import defaultdict
from itertools import product


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


def define_state_machine_transitions(state_machine):
    all_states = set(state_machine.States.states())
    print('all_states [[%s]]' % all_states)
    transitions = state_machine.Meta.transitions
    for state in all_states:
        if state not in transitions:
            transitions[state] = list(all_states)
    print('transitions')
    print(transitions)
    transition_map = defaultdict(list)
    for (from_state, to_state) in product(all_states, all_states):
        valid_transition = to_state in transitions[from_state]
        transition_map[from_state].append((to_state, valid_transition))
    return transition_map
