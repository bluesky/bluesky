"""
This module creates a singleton object to store settings such as which
RunEngine and detectors the simple scan interface should invoke.
"""
from traitlets import HasTraits, TraitType, Unicode, List, Bool, link, Dict
import itertools
from collections import Iterable
from bluesky import RunEngine
from bluesky.utils import get_history


# Define custom traitlets.
# See http://traitlets.readthedocs.org/en/latest/defining_traits.html


class RunEngineTraitType(TraitType):

    info_text = 'a RunEngine instance'
    default_value = RunEngine(get_history())

    def validate(self, obj, value):
        if not isinstance(value, RunEngine):
            self.error(obj, value)
        return value


class Readable(TraitType):

    info_text = 'a Readable (detector-like) object'

    def validate(self, obj, value):
        try:
            validate_readable(value)
        except TypeError:
            self.error(obj, value)
        return value


class Movable(TraitType):

    info_text = 'a Movable (positioner-like) object'

    def validate(self, obj, value):
        try:
            validate_movable(value)
        except TypeError:
            self.error(obj, value)
        return value


class FlyableList(TraitType):

    info_text = 'a list or iterable of Flyable (flyscan-able) objects'
    default_value = []

    def validate(self, obj, value):
        if not isinstance(value, Iterable):
            self.error(obj, value)
        for flyable in value:
            try:
                validate_flyable(flyable)
            except TypeError:
                self.error(obj, value)
        return value


class ReadableList(TraitType):

    info_text = 'a list or iterable of Readable (detector-like) objects'
    default_value = []

    def validate(self, obj, value):
        if not isinstance(value, Iterable):
            self.error(obj, value)
        for det in value:
            try:
                validate_readable(det)
            except TypeError:
                self.error(obj, value)

            # If dealing with an areadetector, ensure that all of the plugin
            # ports are exposed to Python/ophyd:
            if hasattr(det, 'validate_asyn_ports'):
                det.validate_asyn_ports()

        # Read the scary line below as "flatten data key names"
        data_keys = list(itertools.chain.from_iterable(
            [list(det.describe().keys()) for det in value]))
        # The data keys taken together must be unique.
        if len(set(data_keys)) < len(data_keys):
            self.error(obj, value)
        return value


def method_validator_factory(name, required_methods):
    "Make a validator function that checks that an object has certain methods."
    def f(obj):
        if isinstance(obj, str):
            raise TypeError("{} is a string, not a {} object "
                            "(Do not use quotes)".format(obj, name))
        for method in required_methods:
            if not hasattr(obj, method):
                raise TypeError("{} is not a {} object; "
                                "it does not have a "
                                "'{}' method'".format(obj, name, method))
    return f


validate_readable = method_validator_factory('Readable',
                                             ['read', 'describe', 'trigger'])
validate_movable = method_validator_factory('Movable',
                                            ['read', 'describe', 'trigger',
                                             'set', 'stop'])
validate_flyable = method_validator_factory('Flyable',
                                            ['kickoff', 'collect', 'complete',
                                             'describe_collect', 'stop'])


class GlobalState(HasTraits):
    "A bucket of validated global state used by the simple scan API"
    RE = RunEngineTraitType()
    DETS = ReadableList()
    BASELINE_DEVICES = ReadableList()
    FLYERS = FlyableList()
    MONITORS = ReadableList()
    MASTER_DET = Readable()
    MASTER_DET_FIELD = Unicode()
    TH_MOTOR = Movable()
    TTH_MOTOR = Movable()
    TEMP_CONTROLLER = Movable()
    TABLE_COLS = List()
    PLOT_Y = Unicode()
    OVERPLOT = Bool(True)
    MD_TIME_KEY = Unicode('count_time')
    PS_CONFIG = Dict(default_value=dict(edge_count=None))
    SUB_FACTORIES = Dict(default_value={})


gs = GlobalState()  # a singleton instance
link((gs, 'PLOT_Y'), (gs, 'MASTER_DET_FIELD'))


def get_gs():
    "A guaranteed way to access the singleton GlobalState instance."
    global gs
    return gs


# Convenience functions, analogous to pyplot's set_xlim()

def resume():
    """
    A convenience function to ask the RunEngine to resume after a pause.

    This function closes over the global RunEngine instance, gs.RE.
    """
    return gs.RE.resume()


def abort():
    """
    A convenience function to ask the RunEngine to abort after a pause,
    canceling any further actions by by the scan and marking it as aborted.

    This function closes over the global RunEngine instance, gs.RE.
    """
    return gs.RE.abort()


def stop():
    """
    A convenience function to ask the RunEngine to stop after a pause,
    canceling any further actions by the scan and marking it as successfully
    completed.

    This function closes over the global RunEngine instance, gs.RE.
    """
    return gs.RE.stop()


def state():
    """
    A convenience function to return the state of the RunEngine.

    This function closes over the global RunEngine instance, gs.RE.
    """
    # Sharp edge: RE.state is a property; but in order to close over gs,
    # I am exposing it in the global namespace as a function.
    return gs.RE.state
