"""Utilities for core."""

from enum import Enum, unique
from functools import wraps

from .errors import TransitionError


def is_(self, state):
    """Check if machine is in given state."""
    translator = self._meta['translator']
    state = translator.translate(state)
    return self.actual_state == state


def can_be_(self, state):
    """Check if machine can transit to given state."""
    translator = self._meta['translator']
    state = translator.translate(state)

    if self._meta['complete']:
        return True

    if self.actual_state is None:
        return True

    transitions = self._meta['transitions'][self.actual_state]
    return state in transitions


def force_set(self, state):
    """Set new state without checking if transition is allowed."""
    translator = self._meta['translator']
    state = translator.translate(state)
    attr = self._meta['state_attribute_name']
    setattr(self, attr, state)


def set_(self, state):
    """Set new state for machine."""
    if not self.can_be_(state):
        state = self._meta['translator'].translate(state)
        raise TransitionError(
            "Cannot transit from '{actual_value}' to '{value}'."
            .format(actual_value=self.actual_state.value, value=state.value)
        )

    self.force_set(state)


def state_getter(self):
    """Get actual state as value."""
    try:
        return self.actual_state.value
    except AttributeError:
        return None


def state_setter(self, value):
    """Set new state for machine."""
    self.set_(value)


def generate_getter(value):
    """Generate getter for given value."""
    @property
    @wraps(is_)
    def getter(self):
        return self.is_(value)

    return getter


def generate_checker(value):
    """Generate state checker for given value."""
    @property
    @wraps(can_be_)
    def checker(self):
        return self.can_be_(value)

    return checker


def generate_setter(value):
    """Generate setter for given value."""
    @wraps(set_)
    def setter(self):
        self.set_(value)

    return setter

state_property = property(state_getter, state_setter)


@property
def actual_state(self):
    """Actual state as `None` or `enum` instance."""
    attr = self._meta['state_attribute_name']
    return getattr(self, attr)


@property
def as_enum(self):
    """Return actual state as enum."""
    return self.actual_state


class EnumValueTranslator(object):

    """Helps to find enum element by its value."""

    def __init__(self, base_enum):
        """Init.

        :param enum base_enum: Enum, to which elements values are translated.

        """
        base_enum = unique(base_enum)
        self.base_enum = base_enum
        self.generate_search_table()

    def generate_search_table(self):
        self.search_table = dict(
            (item.value, item) for item in list(self.base_enum)
        )

    def translate(self, value):
        """Translate value to enum instance.

        If value is already enum instance, check if this value belongs to base
        enum.

        """
        if self._check_if_already_proper(value):
            return value

        try:
            return self.search_table[value]
        except KeyError:
            raise ValueError("Value {value} doesn't match any state.".format(
                value=value
            ))

    def _check_if_already_proper(self, value):
        if isinstance(value, Enum):
            if value in self.base_enum:
                return True
            raise ValueError(
                "Given value ('{value}') doesn't belong to states enum."
                .format(value=value)
            )
        return False
