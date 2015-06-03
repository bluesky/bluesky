from bluesky import RunEngineStateMachine
from collections import defaultdict
from itertools import product
from pprint import pprint

sm = RunEngineStateMachine()


def define_state_machine_transitions():
    all_states = set(sm.States.states())
    print('all_states [[%s]]' % all_states)
    transitions = sm.Meta.transitions
    for state in all_states:
        if state not in transitions:
            transitions[state] = list(all_states)
    print('transitions')
    pprint(transitions)
    transition_map = defaultdict(list)
    for (from_state, to_state) in product(all_states, all_states):
        valid_transition = from_state in transitions[to_state]
        transition_map[to_state].append((from_state, valid_transition))
    return transition_map

transition_map = dict(define_state_machine_transitions())
print('temperature map')
pprint(transition_map)

sm.state = 'running'
sm.state = 'soft'
sm.state = 'running'
