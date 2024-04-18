import enum

from bluesky._vendor.super_state_machine import extras, machines


class Lock(machines.StateMachine):
    class States(enum.Enum):
        OPEN = "open"
        LOCKED = "locked"

    class Meta:
        allow_empty = False
        initial_state = "open"
        named_transitions = [
            ("lock", "locked"),
            ("open", "open"),
        ]


def test_property_machine():
    class Door:
        lock1 = extras.PropertyMachine(Lock)
        lock2 = extras.PropertyMachine(Lock)

    door = Door()
    assert door.lock1 == "open"
    assert door.lock2 == "open"
    door.lock1.lock()
    assert door.lock1 == "locked"
    assert door.lock2 == "open"
    door.lock2.state_machine.lock()
    assert door.lock1 == "locked"
    assert door.lock2 == "locked"
    door.lock1.open()
    door.lock2.open()
    assert door.lock1 == "open"
    assert door.lock2 == "open"
