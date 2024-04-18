from enum import Enum

import pytest

from bluesky._vendor.super_state_machine import errors, machines


class StatesEnum(Enum):
    ONE = "one"
    TWO = "two"
    THREE = "three"
    FOUR = "four"


class OtherEnum(Enum):
    ONE = "one"


def test_transitions():
    class Machine(machines.StateMachine):
        States = StatesEnum
        state = StatesEnum.ONE

        class Meta:
            transitions = {
                StatesEnum.ONE: [StatesEnum.TWO, StatesEnum.THREE],
                StatesEnum.TWO: [StatesEnum.ONE, StatesEnum.THREE],
                StatesEnum.THREE: [StatesEnum.TWO],
            }

    sm = Machine()
    assert sm.is_one is True
    sm.set_two()
    sm.set_three()

    with pytest.raises(errors.TransitionError):
        sm.set_one()
    assert sm.is_three is True

    with pytest.raises(errors.TransitionError):
        sm.set_("one")
    assert sm.is_three is True

    with pytest.raises(errors.TransitionError):
        sm.state = "one"

    assert sm.is_three is True

    sm.set_two()
    sm.set_one()
    assert sm.is_one is True


def test_transitions_checkers():
    class Machine(machines.StateMachine):
        States = StatesEnum
        state = "one"

        class Meta:
            transitions = {
                "one": ["two", "three"],
                "two": ["one", "three"],
                "three": ["two", "three"],
            }

    sm = Machine()
    assert sm.can_be_one is False
    assert sm.can_be_two is True
    assert sm.can_be_three is True
    assert sm.can_be_four is False
    assert sm.can_be_("one") is False
    assert sm.can_be_("two") is True
    assert sm.can_be_("three") is True
    assert sm.can_be_("four") is False

    with pytest.raises(errors.TransitionError):
        sm.set_one()

    sm.set_two()
    assert sm.can_be_one is True
    assert sm.can_be_two is False
    assert sm.can_be_three is True
    assert sm.can_be_four is False

    sm.set_three()
    assert sm.can_be_one is False
    assert sm.can_be_two is True
    assert sm.can_be_three is True
    assert sm.can_be_four is False

    sm.set_three()


def test_transitions_checkers_with_complete_graph():
    class Machine(machines.StateMachine):
        States = StatesEnum
        state = "one"

        class Meta:
            complete = True
            transitions = {
                "one": ["two", "three"],
                "two": ["one", "three"],
                "three": ["two", "three"],
            }

    sm = Machine()
    sm.set_three()
    assert sm.can_be_one is True
    assert sm.can_be_two is True
    assert sm.can_be_three is True
    assert sm.can_be_four is True
    sm.set_four()
    assert sm.is_four is True


def test_named_transitions_checkers():
    class Machine(machines.StateMachine):
        States = StatesEnum

        class Meta:
            initial_state = "one"
            transitions = {
                "one": ["two", "three"],
                "two": ["one", "three"],
                "three": ["two", "three"],
            }
            named_checkers = [
                ("can_go_to_one", "one"),
                ("can_become_two", StatesEnum.TWO),
            ]

        @property
        def can_one(self):
            return self.can_be_("one")

    sm = Machine()
    sm.set_two()
    assert sm.can_be_one is True
    assert sm.can_one is True
    assert sm.can_go_to_one is True
    assert sm.can_become_two is False


def test_named_transitions_checkers_cant_overwrite_methods():
    with pytest.raises(ValueError):

        class Machine(machines.StateMachine):
            States = StatesEnum

            class Meta:
                named_checkers = [
                    ("can_one", "one"),
                ]

            @property
            def can_one(self):
                return self.can_be_("one")


def test_named_checkers_cant_overwrite_generated_methods():
    with pytest.raises(ValueError):

        class Machine(machines.StateMachine):
            States = StatesEnum

            class Meta:
                named_checkers = [
                    ("can_be_one", "one"),
                ]

            @property
            def can_one(self):
                return self.can_be_("one")


def test_named_checkers_dont_accept_wrong_values():
    with pytest.raises(ValueError):

        class Machine(machines.StateMachine):
            States = StatesEnum

            class Meta:
                named_checkers = [
                    ("can_become_five", "five"),
                ]

            @property
            def can_one(self):
                return self.can_be_("one")


def test_named_checkers_dont_accept_wrong_enums():
    with pytest.raises(ValueError):

        class Machine(machines.StateMachine):
            States = StatesEnum

            class Meta:
                named_checkers = [
                    ("can_be_other_one", OtherEnum.ONE),
                ]

            @property
            def can_one(self):
                return self.can_be_("one")


def test_transitions_with_wrong_enum():
    with pytest.raises(ValueError):

        class Machine(machines.StateMachine):
            States = StatesEnum

            class Meta:
                transitions = {
                    OtherEnum.ONE: [StatesEnum.TWO, StatesEnum.THREE],
                    StatesEnum.TWO: [StatesEnum.ONE, StatesEnum.THREE],
                    StatesEnum.THREE: [StatesEnum.TWO],
                }


def test_named_transitions():
    class Machine(machines.StateMachine):
        States = StatesEnum
        state = "one"

        class Meta:
            transitions = {
                "one": ["two", "three"],
                "two": ["one", "three"],
                "three": ["two", "three"],
            }
            named_transitions = [
                ("run", "two"),
                ("confirm", "three"),
                ("cancel", StatesEnum.FOUR),
            ]

    sm = Machine()
    sm.run()
    assert sm.is_two is True
    sm.confirm()
    assert sm.is_three is True
    sm.cancel()
    assert sm.is_four is True


def test_named_transitions_collisions():
    with pytest.raises(ValueError):

        class Machine(machines.StateMachine):
            States = StatesEnum
            state = "one"

            class Meta:
                transitions = {
                    "o": ["tw", "th"],
                    "tw": ["o", "th"],
                    "th": ["tw"],
                }
                named_transitions = [
                    ("run", "one"),
                    ("confirm", "two"),
                    ("cancel", "three"),
                ]

            def run():
                pass


def test_named_transitions_collisions_with_auto_generated_methods():
    with pytest.raises(ValueError):

        class Machine(machines.StateMachine):
            States = StatesEnum
            state = "one"

            class Meta:
                named_transitions = [
                    ("is_one", "one"),
                    ("confirm", "two"),
                    ("cancel", "three"),
                ]


def test_named_transitions_wrong_value():
    with pytest.raises(ValueError):

        class Machine(machines.StateMachine):
            States = StatesEnum
            state = "one"

            class Meta:
                named_transitions = [
                    ("run", "five"),
                    ("confirm", "two"),
                    ("cancel", "three"),
                ]


def test_named_transitions_wrong_enum():
    with pytest.raises(ValueError):

        class Machine(machines.StateMachine):
            States = StatesEnum
            state = "one"

            class Meta:
                named_transitions = [
                    ("run", OtherEnum.ONE),
                    ("confirm", "two"),
                    ("cancel", "three"),
                ]


def test_named_transitions_are_in_state_graph():
    class Machine(machines.StateMachine):
        States = StatesEnum
        state = "one"

        class Meta:
            named_transitions = [
                ("run", "one"),
                ("confirm", "two"),
                ("cancel", "three"),
            ]

    sm = Machine()
    assert sm.can_be_one is True
    assert sm.can_be_two is True
    assert sm.can_be_three is True
    assert sm.can_be_four is False


def test_named_transitions_with_restricted_source():
    class Machine(machines.StateMachine):
        States = StatesEnum
        state = "one"

        class Meta:
            named_transitions = [
                ("confirm", "two", "one"),
                ("cancel", "three", ["one", "two"]),
                ("surprise", "four", []),
            ]

    sm = Machine()
    assert sm.can_be_two is True
    assert sm.can_be_three is True
    sm.confirm()
    assert sm.can_be_one is False
    assert sm.can_be_three is True
    assert sm.can_be_four is False
    assert sm.can_be_two is False
    sm.cancel()
    assert sm.is_three is True


def test_restricted_source_proper_value():
    with pytest.raises(ValueError):

        class Machine(machines.StateMachine):
            States = StatesEnum

            class Meta:
                named_transitions = [
                    ("run", "o", None),
                    ("confirm", "two", "o"),
                    ("cancel", "three", ["o", "tw"]),
                    ("surprise", "four", ["five"]),
                ]


def test_restricted_source_proper_enum():
    with pytest.raises(ValueError):

        class Machine(machines.StateMachine):
            States = StatesEnum

            class Meta:
                named_transitions = [
                    ("run", "o", None),
                    ("confirm", "two", "o"),
                    ("cancel", "three", ["o", "tw"]),
                    ("surprise", "four", OtherEnum.ONE),
                ]


def test_force_set():
    class Machine(machines.StateMachine):
        States = StatesEnum
        state = "one"

        class Meta:
            transitions = {
                "one": ["two", "three"],
            }

    machine = Machine()
    assert machine.can_be_four is False
    with pytest.raises(errors.TransitionError):
        machine.set_four()
    assert machine.state == "one"
    machine.force_set("four")
    assert machine.state == "four"


def test_force_set_name_collision():
    with pytest.raises(ValueError):

        class Machine(machines.StateMachine):
            States = StatesEnum

            def force_set():
                pass


def test_force_set_name_collision_with_generated_methods():
    with pytest.raises(ValueError):

        class Machine(machines.StateMachine):
            States = StatesEnum

            class Meta:
                named_transitions = [("force_set", "one")]


def test_force_set_accepts_only_proper_values():
    class Machine(machines.StateMachine):
        States = StatesEnum
        state = StatesEnum.ONE

        class Meta:
            complete = False

    machine = Machine()
    assert machine.can_be_four is False
    machine.force_set("four")
    machine.force_set(StatesEnum.FOUR)
    machine.force_set("four")
    with pytest.raises(ValueError):
        machine.force_set("fourtyfour")
    with pytest.raises(ValueError):
        machine.force_set(OtherEnum.ONE)
