"""
This module creates a singleton object to store settings such as which
RunEngine and detectors the simple scan interface should invoke.
"""
from traitlets import HasTraits, TraitType, Unicode, List, Float
from collections import Iterable
from bluesky.run_engine import RunEngine


# Define custom traitlets.
# See http://traitlets.readthedocs.org/en/latest/defining_traits.html


class RunEngineTraitType(TraitType):

    info_text = 'a RunEngine instance'
    default_value = RunEngine(dict())

    def validate(self, obj, value):
        if not isinstance(value, RunEngine):
            self.error(obj, vluae)
        return value


class Readable(TraitType):

    info_text = 'a Readable (detector-like) object'

    def validate(self, obj, value):
        try:
            validate_detector(value)
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


class DetectorList(TraitType):

    info_text = 'a list or iterable of Readable (detector-like) objects'

    def validate(self, obj, value):
        if not isinstance(value, Iterable):
            self.error(obj, value)
        for det in value:
            try:
                validate_detector(det)
            except TypeError:
                self.error(obj, value)
        return value


def validate_detector(det):
    if isinstance(det, str):
        raise TypeError("{0} is a string, not a Readable object "
                        "(Do not use quotes)".format(det))
    required_methods = ['read', 'trigger']
    for method in required_methods:
        if not hasattr(det, method):
            raise TypeError("{0} is not a detector; it does not have "
                            "a '{1}' method'".format(det, method))


def validate_movable(movable):
    if isinstance(movable, str):
        raise TypeError("{0} is a string, not a Movable object "
                        "(Do not use quotes)".format(det))
    required_methods = ['read', 'set']
    for method in required_methods:
        if not hasattr(movable, method):
            raise TypeError("{0} is not a Movable; it does not have "
                            "a '{1}' method'".format(det, method))


class GlobalState(HasTraits):
    "A bucket of validated global state used by the simple scan API"
    RE = RunEngineTraitType()
    DETS = DetectorList()
    MASTER_DET = Readable()
    MASTER_DET_FIELD = Unicode()
    H_MOTOR = Readable()
    K_MOTOR = Readable()
    L_MOTOR = Readable()
    TH_MOTOR = Movable()
    TTH_MOTOR = Movable()
    TEMP_CONTROLLER = Movable()
    TABLE_COLS = List()
    PLOT_Y = Unicode()
    COUNT_TIME = Float(1.0)


gs = GlobalState()  # a singleton instance


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


def panic():
    """
    A convenience function to 'panic' the RunEngine, forbidding it to run
    until all_is_well is called.

    This function closes over the global RunEngine instance, gs.RE.
    """
    return gs.RE.panic()


def all_is_well():
    """
    A convenience function to 'un-panic' the RunEngine, allowing normal
    operation.

    This function closes over the global RunEngine instance, gs.RE.
    """
    return gs.RE.all_is_well()


def state():
    """
    A convenience function to return the state of the RunEngine.

    This function closes over the global RunEngine instance, gs.RE.
    """
    # Sharp edge: RE.state is a property; but in order to close over gs,
    # I am exposing it in the global namespace as a function.
    return gs.RE.state
