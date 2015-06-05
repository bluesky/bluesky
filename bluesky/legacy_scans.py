"""
This recreates @stuwilkins' scan api from ophyd using the bluesky message API.
All of these are deprecated. See bluesky/scans.py for new versions.
"""


import warnings
from .scans import LinDscan, LinAscan, Count

def _deprecation_warning(replacement):
    warnings.warn("ascan is deprecated, and it does not provide "
                  "the latest functionality. Use %s instead." % replacement,
                  UserWarning)


class _LegacyScan:
    def __init__(self, RE):
        self.RE = RE
        self.default_detectors = []
        self.detectors = []

    def _run(self, curried):
        dets = set(self.detectors) | set(self.default_detectors)
        self.RE(curried(dets))


class LegacyAscan(_LegacyScan):
    def __call__(self, motor, start, stop, num):
        _deprecation_warning('LinAscan')
        curried = lambda dets: LinAscan(motor, dets, start, stop, num)
        super()._run(curried)


class LegacyDscan(_LegacyScan):
    def __call__(self, motor, start, stop, num):
        _deprecation_warning('LinDscan')
        dets = set(self.detectors) | set(self.default_detectors)
        curried = lambda dets: LinDscan(motor, dets, start, stop, num)
        super()._run(curried)


class LegacyCount(_LegacyScan):
    def __call__(self):
        _deprecation_warning('Count')
        curried = lambda dets: Count(dets)
        super()._run(curried)
