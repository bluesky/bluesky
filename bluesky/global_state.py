from weakref import WeakKeyDictionary
from collections import Iterable
from bluesky.run_engine import RunEngine


class BaseDescriptor:
    # Adapted from http://nbviewer.ipython.org/urls/gist.github.com/ChrisBeaumont/5758381/raw/descriptor_writeup.ipynb
    def __init__(self, default):
        self.default = default
        self.contents = WeakKeyDictionary()
        
    def __get__(self, instance, owner):
        return self.contents.get(instance, self.default)
    
    def __set__(self, instance, value):
        self.validate(value)
        self.contents[instance] = value

    def validate(self, value):
        pass


class RunEngineDescriptor(BaseDescriptor):
    def validate(self, value):
        if not isinstance(value, RunEngine):
            raise TypeError("must an instance of the RunEngine")


class DetectorDescriptor(BaseDescriptor):
    def validate(self, value):
        if value is not None:
            validate_detector(value)

    def __get__(self, instance, owner):
        if instance not in self.contents:
            raise("ValueError: not set")


class DetectorListDescriptor(BaseDescriptor):
    def validate(self, value):
        if not isinstance(value, Iterable):
            raise TypeError("must be a list or iterable")
        for det in value:
            validate_detector(det)


def validate_detector(det):
    if isinstance(det, str):
        raise TypeError("{0} is a string, not a Readable object "
                        "(Do not use quotes)").format(det)
    required_methods = ['read', 'trigger']
    for method in required_methods:
        if not hasattr(det, method):
            raise TypeError("{0} is not a detector; it does not have "
                            "a '{1}' method'".format(det, method))


class GlobalState:
    "A bucket of validated global state used by the simple scan API"
    RE = RunEngineDescriptor(RunEngine(dict()))
    DETS = DetectorListDescriptor([])
    MASTER_DET = DetectorDescriptor(None)
    MASTER_DET_FIELD = DetectorDescriptor(None)
    H_MOTOR = DetectorDescriptor(None)
    K_MOTOR = DetectorDescriptor(None)
    L_MOTOR = DetectorDescriptor(None)
    TH_MOTOR = DetectorDescriptor(None)
    TTH_MOTOR = DetectorDescriptor(None)
    TEMP_CONTROLLER = DetectorDescriptor(None)
