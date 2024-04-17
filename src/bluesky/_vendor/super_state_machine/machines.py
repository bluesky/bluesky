"""State machine core."""

from enum import Enum
from functools import partial

import six

from . import utils


NotSet = object()


class DefaultMeta(object):

    """Default configuration values."""

    states_enum_name = 'States'


class AttributeDict(dict):

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        return self[key]


class StateMachineMetaclass(type):

    """Metaclass for state machine, to build all its logic."""

    def __new__(cls, name, bases, attrs):
        """Create state machine and add all logic and methods to it."""
        cls._set_up_context()
        new_class = super(cls, cls).__new__(cls, name, bases, attrs)
        cls.context.new_class = new_class

        parents = [b for b in bases if isinstance(b, cls)]
        if not parents:
            return cls.context.new_class

        cls._set_up_config_getter()
        cls._check_states_enum()
        cls._check_if_states_are_strings()
        cls._set_up_translator()
        cls._calculate_state_name()
        cls._check_state_value()
        cls._add_standard_attributes()
        cls._generate_standard_transitions()
        cls._generate_standard_methods()
        cls._generate_named_checkers()
        cls._generate_named_transitions()
        cls._add_new_methods()
        cls._set_complete_option()
        cls._complete_meta_for_new_class()

        new_class = cls.context.new_class
        del cls.context
        return new_class

    @classmethod
    def _set_up_context(cls):
        """Create context to keep all needed variables in."""
        cls.context = AttributeDict()
        cls.context.new_meta = {}
        cls.context.new_transitions = {}
        cls.context.new_methods = {}

    @classmethod
    def _check_states_enum(cls):
        """Check if states enum exists and is proper one."""
        states_enum_name = cls.context.get_config('states_enum_name')
        try:
            cls.context['states_enum'] = getattr(
                cls.context.new_class, states_enum_name)
        except AttributeError:
            raise ValueError('No states enum given!')

        proper = True
        try:
            if not issubclass(cls.context.states_enum, Enum):
                proper = False
        except TypeError:
            proper = False

        if not proper:
            raise ValueError(
                'Please provide enum instance to define available states.')

    @classmethod
    def _check_if_states_are_strings(cls):
        """Check if all states are strings."""
        for item in list(cls.context.states_enum):
            if not isinstance(item.value, six.string_types):
                raise ValueError(
                    'Item {name} is not string. Only strings are allowed.'
                    .format(name=item.name)
                )

    @classmethod
    def _check_state_value(cls):
        """Check initial state value - if is proper and translate it.

        Initial state is required.
        """
        state_value = cls.context.get_config('initial_state', None)
        state_value = state_value or getattr(
            cls.context.new_class, cls.context.state_name, None
        )

        if not state_value:
            raise ValueError(
                "Empty state is disallowed, yet no initial state is given!"
            )
        state_value = (
            cls.context
            .new_meta['translator']
            .translate(state_value)
        )
        cls.context.state_value = state_value

    @classmethod
    def _add_standard_attributes(cls):
        """Add attributes common to all state machines.

        These are methods for setting and checking state etc.

        """
        setattr(
            cls.context.new_class,
            cls.context.new_meta['state_attribute_name'],
            cls.context.state_value)
        setattr(
            cls.context.new_class,
            cls.context.state_name,
            utils.state_property)

        setattr(cls.context.new_class, 'is_', utils.is_)
        setattr(cls.context.new_class, 'can_be_', utils.can_be_)
        setattr(cls.context.new_class, 'set_', utils.set_)

    @classmethod
    def _generate_standard_transitions(cls):
        """Generate methods used for transitions."""
        allowed_transitions = cls.context.get_config('transitions', {})
        for key, transitions in allowed_transitions.items():
            key = cls.context.new_meta['translator'].translate(key)

            new_transitions = set()
            for trans in transitions:
                if not isinstance(trans, Enum):
                    trans = cls.context.new_meta['translator'].translate(trans)
                new_transitions.add(trans)

            cls.context.new_transitions[key] = new_transitions

        for state in cls.context.states_enum:
            if state not in cls.context.new_transitions:
                cls.context.new_transitions[state] = set()

    @classmethod
    def _generate_standard_methods(cls):
        """Generate standard setters, getters and checkers."""
        for state in cls.context.states_enum:
            getter_name = 'is_{name}'.format(name=state.value)
            cls.context.new_methods[getter_name] = utils.generate_getter(state)

            setter_name = 'set_{name}'.format(name=state.value)
            cls.context.new_methods[setter_name] = utils.generate_setter(state)

            checker_name = 'can_be_{name}'.format(name=state.value)
            checker = utils.generate_checker(state)
            cls.context.new_methods[checker_name] = checker

        cls.context.new_methods['actual_state'] = utils.actual_state
        cls.context.new_methods['as_enum'] = utils.as_enum
        cls.context.new_methods['force_set'] = utils.force_set

    @classmethod
    def _generate_named_checkers(cls):
        named_checkers = cls.context.get_config('named_checkers', None) or []
        for method, key in named_checkers:
            if method in cls.context.new_methods:
                raise ValueError(
                    "Name collision for named checker '{checker}' - this "
                    "name is reserved for other auto generated method."
                    .format(checker=method)
                )

            key = cls.context.new_meta['translator'].translate(key)
            cls.context.new_methods[method] = utils.generate_checker(key.value)

    @classmethod
    def _generate_named_transitions(cls):
        named_transitions = (
            cls.context.get_config('named_transitions', None) or [])

        translator = cls.context.new_meta['translator']
        for item in named_transitions:
            method, key, from_values = cls._unpack_named_transition_tuple(item)

            if method in cls.context.new_methods:
                raise ValueError(
                    "Name collision for transition '{transition}' - this name "
                    "is reserved for other auto generated method."
                    .format(transition=method)
                )

            key = translator.translate(key)
            cls.context.new_methods[method] = utils.generate_setter(key)

            if from_values:
                from_values = [translator.translate(k) for k in from_values]
                for s in cls.context.states_enum:
                    if s in from_values:
                        cls.context.new_transitions[s].add(key)

    @classmethod
    def _unpack_named_transition_tuple(cls, item):
        try:
            method, key = item
            from_values = cls.context['states_enum']
        except ValueError:
            method, key, from_values = item
            if from_values is None:
                from_values = []
            if not isinstance(from_values, list):
                from_values = list((from_values,))

        return method, key, from_values

    @classmethod
    def _add_new_methods(cls):
        """Add all generated methods to result class."""
        for name, method in cls.context.new_methods.items():
            if hasattr(cls.context.new_class, name):
                raise ValueError(
                    "Name collision in state machine class - '{name}'."
                    .format(name)
                )

            setattr(cls.context.new_class, name, method)

    @classmethod
    def _set_complete_option(cls):
        """Check and set complete option."""
        get_config = cls.context.get_config
        complete = get_config('complete', None)
        if complete is None:
            conditions = [
                get_config('transitions', False),
                get_config('named_transitions', False),
            ]
            complete = not any(conditions)

        cls.context.new_meta['complete'] = complete

    @classmethod
    def _set_up_config_getter(cls):
        meta = getattr(cls.context.new_class, 'Meta', DefaultMeta)
        cls.context.get_config = partial(get_config, meta)

    @classmethod
    def _set_up_translator(cls):
        translator = utils.EnumValueTranslator(cls.context['states_enum'])
        cls.context.new_meta['translator'] = translator

    @classmethod
    def _calculate_state_name(cls):
        cls.context.state_name = 'state'
        new_state_name = '_' + cls.context.state_name
        cls.context.new_meta['state_attribute_name'] = new_state_name

    @classmethod
    def _complete_meta_for_new_class(cls):
        cls.context.new_meta['transitions'] = cls.context.new_transitions
        cls.context.new_meta['config_getter'] = cls.context['get_config']
        setattr(cls.context.new_class, '_meta', cls.context['new_meta'])


class StateMachine(six.with_metaclass(StateMachineMetaclass)):

    """State machine."""


def get_config(original_meta, attribute, default=NotSet):
    for meta in [original_meta, DefaultMeta]:
        try:
            return getattr(meta, attribute)
        except AttributeError:
            pass

    if default is NotSet:
        raise

    return default
