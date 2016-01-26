"""
This module is for decorators related to testing.

Much of this code is inspired by the code in matplotlib.  Exact copies
are noted.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six


# copied from matplotlib
class KnownFailureDidNotFailTest(Exception):
    '''Raise this exception to mark a test should have failed but did not.'''
    pass


# This code is copied from numpy
class KnownFailureTest(Exception):
    '''Raise this exception to mark a test as a known failing test.'''
    pass


class KnownFailure(Exception):
    pass
