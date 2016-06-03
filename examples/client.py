from bluesky.service_client import execute_on_remote

plan = [{'msg': {'command': 'null', 'obj': None, 'args': [], 'kwargs': {}}},

        # a pause message pauses the RunEngine
        {'msg': {'command': 'pause', 'obj': None, 'args': [], 'kwargs': {}}},
        # and it can be resumed
        {'change_state': 'resume'},

        # objects like motor are mapped with string idenfiers
        {'msg': {'command': 'set', 'obj': 'motor', 'args': [5], 'kwargs': {}}},

        # pause again, and this time abort
        {'msg': {'command': 'pause', 'obj': None, 'args': [], 'kwargs': {}}},
        {'change_state': 'abort'},
        # the RunEngine is idle, but the server is ready to send it new
        # messages and put it back into a running state
        {'msg': {'command': 'null', 'obj': None, 'args': [], 'kwargs': {}}},

        # this raises a TypeError, but the server recovers
        {'msg': {'command': 'set', 'obj': 'motor', 'args': [], 'kwargs': {}}},
        # ... and this is processed
        {'msg': {'command': 'null', 'obj': None, 'args': [], 'kwargs': {}}}]


def main():
    execute_on_remote(plan)


if __name__ == "__main__":
    main()
