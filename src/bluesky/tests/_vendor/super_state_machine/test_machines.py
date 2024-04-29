from enum import Enum

import pytest

from bluesky._vendor.super_state_machine import machines


class StatesEnum(Enum):
    ONE = "one"
    TWO = "two"
    THREE = "three"


class OtherEnum(Enum):
    ONE = "one"


def test_state_machine_is_always_scalar():
    class Machine(machines.StateMachine):
        class States(Enum):
            ONE = "one"
            TWO = "two"
            THREE = "three"

        state = States.ONE

    sm = Machine()
    assert sm.state == "one"


def test_state_machine_allow_scalars_on_init():
    class Machine(machines.StateMachine):
        class States(Enum):
            ONE = "one"
            TWO = "two"
            THREE = "three"

        state = "two"

    sm = Machine()
    assert sm.is_two is True


def test_state_machine_accepts_enums_only_from_proper_source():
    class OtherEnum(Enum):
        FOUR = "four"

    with pytest.raises(ValueError):

        class Machine(machines.StateMachine):
            class States(Enum):
                ONE = "one"
                TWO = "two"
                THREE = "three"

            state = OtherEnum.FOUR


def test_state_machine_doesnt_allow_wrong_scalars():
    with pytest.raises(ValueError):

        class Machine(machines.StateMachine):
            class States(Enum):
                ONE = "one"
                TWO = "two"
                THREE = "three"

            state = "four"


def test_state_machine_accepts_only_unique_enums():
    with pytest.raises(ValueError):

        class Machine(machines.StateMachine):
            class States(Enum):
                ONE = "one"
                TWO = "one"
                THREE = "three"


def test_state_machine_allows_to_change_and_check_state():
    class Machine(machines.StateMachine):
        state = "one"

        class States(Enum):
            ONE = "one"
            TWO = "two"
            THREE = "three"

    sm = Machine()
    assert sm.is_("one") is True
    assert sm.state == "one"
    assert sm.is_("two") is False
    sm.set_("two")
    assert sm.is_("two") is True


def test_state_machine_allows_to_change_and_check_state_by_methods():
    class Machine(machines.StateMachine):
        state = "one"

        class States(Enum):
            ONE = "one"
            TWO = "two"
            THREE = "three"

    sm = Machine()
    assert sm.is_one is True
    assert sm.state == "one"
    assert sm.is_two is False
    sm.set_two()
    assert sm.is_two is True


def test_name_collistion_for_checker_raises_exception():
    with pytest.raises(ValueError):

        class Machine(machines.StateMachine):
            class States(Enum):
                ONE = "one"
                TWO = "two"
                THREE = "three"

            def is_one():
                pass


def test_name_collistion_for_setter_raises_exception():
    with pytest.raises(ValueError):

        class Machine(machines.StateMachine):
            class States(Enum):
                ONE = "one"
                TWO = "two"
                THREE = "three"

            def set_one():
                pass


def test_states_enum_can_be_predefined():
    class Machine(machines.StateMachine):
        States = StatesEnum
        state = StatesEnum.ONE

    sm = Machine()
    assert sm.is_one is True


def test_disallow_empty():
    class Machine(machines.StateMachine):
        States = StatesEnum
        state = StatesEnum.ONE

    sm = Machine()
    with pytest.raises(AttributeError):
        del sm.state


def test_states_enum_is_always_given():
    with pytest.raises(ValueError):

        class Machine(machines.StateMachine):
            pass


def test_states_enum_is_always_enum():
    with pytest.raises(ValueError):

        class Machine(machines.StateMachine):
            States = "something"


def test_disallow_empty_without_initial_value():
    with pytest.raises(ValueError):

        class Machine(machines.StateMachine):
            States = StatesEnum

            class Meta:
                allow_empty = False


def test_initial_value():
    class Machine(machines.StateMachine):
        States = StatesEnum
        state = StatesEnum.ONE

        class Meta:
            initial_state = StatesEnum.TWO

    sm = Machine()
    assert sm.state == "two"
    assert sm.is_two is True


def test_wrong_initial_value_from_class_is_ignored():
    class Machine(machines.StateMachine):
        States = StatesEnum
        state = "wrong"

        class Meta:
            initial_state = StatesEnum.TWO

    sm = Machine()
    assert sm.state == "two"
    assert sm.is_two is True


def test_initial_value_from_meta_and_disallowed_empty():
    class Machine(machines.StateMachine):
        States = StatesEnum

        class Meta:
            allow_empty = False
            initial_state = StatesEnum.TWO

    sm = Machine()
    assert sm.state == "two"
    assert sm.is_two is True


def test_disallow_assignation_of_wrong_value():
    class Machine(machines.StateMachine):
        state = "one"

        class States(Enum):
            ONE = "one"
            TWO = "two"
            THREE = "three"

    sm = Machine()
    sm.set_one()
    assert sm.is_one is True
    sm.set_(Machine.States.TWO)
    assert sm.is_two is True
    with pytest.raises(ValueError):
        sm.set_(StatesEnum.TWO)
    with pytest.raises(ValueError):
        sm.set_("four")


def test_checker_getter_and_setter_wrong_values_and_enums():
    class Machine(machines.StateMachine):
        States = StatesEnum
        state = "one"

    sm = Machine()

    sm.is_("one")
    sm.can_be_("one")
    sm.set_("one")

    with pytest.raises(ValueError):
        sm.is_("five")
    with pytest.raises(ValueError):
        sm.is_(OtherEnum.ONE)
    with pytest.raises(ValueError):
        sm.can_be_("five")
    with pytest.raises(ValueError):
        sm.can_be_(OtherEnum.ONE)
    with pytest.raises(ValueError):
        sm.set_("five")
    with pytest.raises(ValueError):
        sm.set_(OtherEnum.ONE)


def test_get_actual_state_as_enum():
    class Machine(machines.StateMachine):
        States = StatesEnum
        state = "one"

    sm = Machine()
    assert sm.actual_state is StatesEnum.ONE
    sm.set_two()
    assert sm.actual_state is StatesEnum.TWO


def test_actual_state_name_collision():
    with pytest.raises(ValueError):

        class Machine(machines.StateMachine):
            States = StatesEnum

            def actual_state():
                pass


def test_actual_state_name_collision_with_generated_methods():
    with pytest.raises(ValueError):

        class Machine(machines.StateMachine):
            States = StatesEnum

            class Meta:
                named_transitions = [
                    ("actual_state", "one"),
                ]


def test_as_enum():
    class Machine(machines.StateMachine):
        States = StatesEnum
        state = "one"

    sm = Machine()
    assert sm.as_enum is StatesEnum.ONE
    sm.set_two()
    assert sm.as_enum is StatesEnum.TWO


def test_as_enum_name_collision():
    with pytest.raises(ValueError):

        class Machine(machines.StateMachine):
            States = StatesEnum

            def as_enum():
                pass


def test_as_enum_name_collision_with_generated_methods():
    with pytest.raises(ValueError):

        class Machine(machines.StateMachine):
            States = StatesEnum

            class Meta:
                named_transitions = [
                    ("as_enum", "one"),
                ]


def test_custom_states_enum():
    class Machine(machines.StateMachine):
        trololo = StatesEnum
        state = "one"

        class Meta:
            states_enum_name = "trololo"

    machine = Machine()
    assert machine.is_one is True


def test_initial_state_is_required():
    with pytest.raises(ValueError):

        class Machine(machines.StateMachine):
            States = StatesEnum

            class Meta:
                pass


def test_state_machine_allows_only_strings():
    with pytest.raises(ValueError):

        class Machine(machines.StateMachine):
            class States(Enum):
                ONE = 1
                TWO = "two"
                THREE = "three"
