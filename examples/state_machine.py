
from bluesky import RunEngineStateMachine

sm = RunEngineStateMachine()
print(sm.state)

print('state.name: state.value')
print('-----------------------')
for state in sm.States:
    print('%s: %s' % (state.name, state.value))

print('get the name of the state of the state machine')
print(sm.state)

print('or get the enum value of the state machine')
print(sm.actual_state)


# Basic commands
#
# - Set the state by using `set_VALUE()`
# - Query if the state machine is in a specific state with `is_VALUE`
# - Query if a transition is possible with `can_be_VALUE`
#
#    - Note that these transitions are defined in the `named_transitions` list of the class Meta bit of the implementation of the `RunEngineStateMachine` in `bluesky/run_engine.py`)
#
# where VALUE is the enum value of all states of the state machine
#
# - idle
# - running
# - aborted
# - soft_pausing
# - hard_pausing
# - paused
# - panicked

sm.set_running()
print('current state is [[%s]]' % sm.state)

print(sm.is_running)

print(sm.can_be_running)

print('Other ways to transition the state machine')

print('use a named transition')
sm.stop()
print('current state is [[%s]]' % sm.state)

print('assign a new state directly to the state of the state machine')
sm.state = 'panicked'
print('current state is [[%s]]' % sm.state)

print('assign a new state directly to the state of the state machine by '
      'providing the minimum amount of information to uniquely identify a '
      'state')
sm.state = 'i'
print('current state is [[%s]]' % sm.state)

sm.state = 'r'
print('current state is [[%s]]' % sm.state)

sm.state = 's'
print('current state is [[%s]]' % sm.state)

print('Here is error you get when the selection is ambiguous')
sm.state = 'p'
print('current state is [[%s]]' % sm.state)
