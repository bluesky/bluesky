"""
This recreates @stuwilkins' scan api from ophyd using the bluesky message API.
All of these are deprecated. See bluesky/scans.py for new versions.
"""


import warnings
from .scans import LinDscan, LinAscan, Count

def _deprecation_warning(old, new):
    warnings.warn("{old} is deprecated, and it does not provide "
                  "the latest functionality. Use {new} instead.".format(
                      old=old, new=new),
                  UserWarning)


class _LegacyScan:
    _shared_config = {'default_detectors': [],
                      'detectors': []}

    @property
    def default_detectors(self):
        return self._shared_config['default_detectors']

    @default_detectors.setter
    def default_detectors(self, val):
        self._shared_config['default_detectors'] = val

    @property
    def detectors(self):
        return self._shared_config['detectors']

    @detectors.setter
    def detectors(self, val):
        self._shared_config['detectors'] = val

    def __init__(self, RE):
        self.RE = RE

    def _run(self, curried):
        dets = set(self.detectors) | set(self.default_detectors)
        self.RE(curried(dets))


class LegacyAscan(_LegacyScan):
    def __call__(self, motor, start, stop, num):
        _deprecation_warning('ascan', 'LinAscan')
        curried = lambda dets: LinAscan(motor, dets, start, stop, num)
        super()._run(curried)


class LegacyDscan(_LegacyScan):
    def __call__(self, motor, start, stop, num):
        _deprecation_warning('dscan', 'LinDscan')
        dets = set(self.detectors) | set(self.default_detectors)
        curried = lambda dets: LinDscan(motor, dets, start, stop, num)
        super()._run(curried)


class LegacyCount(_LegacyScan):
    def __call__(self):
        _deprecation_warning('ct', 'Count')
        curried = lambda dets: Count(dets)
        super()._run(curried)
