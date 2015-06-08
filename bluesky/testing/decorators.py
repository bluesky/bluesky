########################################################################
# Copyright (c) 2015, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
#                                                                      #
# Redistribution and use in source and binary forms, with or without   #
# modification, are permitted provided that the following conditions   #
# are met:                                                             #
#                                                                      #
# * Redistributions of source code must retain the above copyright     #
#   notice, this list of conditions and the following disclaimer.      #
#                                                                      #
# * Redistributions in binary form must reproduce the above copyright  #
#   notice this list of conditions and the following disclaimer in     #
#   the documentation and/or other materials provided with the         #
#   distribution.                                                      #
#                                                                      #
# * Neither the name of the Brookhaven Science Associates, Brookhaven  #
#   National Laboratory nor the names of its contributors may be used  #
#   to endorse or promote products derived from this software without  #
#   specific prior written permission.                                 #
#                                                                      #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS  #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT    #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS    #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE       #
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,           #
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES   #
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR   #
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)   #
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,  #
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OTHERWISE) ARISING   #
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE   #
# POSSIBILITY OF SUCH DAMAGE.                                          #
########################################################################
"""
This module is for decorators related to testing.

Much of this code is inspired by the code in matplotlib.  Exact copies
are noted.
"""
from .noseclasses import (KnownFailureTest,
                          KnownFailureDidNotFailTest)

import nose
from nose.tools import make_decorator


def known_fail_if(cond):
    """
    Make sure a known failure fails.

    This function is a decorator factory.
    """
    # make the decorator function
    def dec(in_func):
        # make the wrapper function
        # if the condition is True
        if cond:
            def inner_wrap():
                # try the test anywoy
                try:
                    in_func()
                # when in fails, raises KnownFailureTest
                # which is registered with nose and it will be marked
                # as K in the results
                except Exception:
                    raise KnownFailureTest()
                # if it does not fail, raise KnownFailureDidNotFailTest which
                # is a normal exception.  This may seem counter-intuitive
                # but knowing when tests that _should_ fail don't can be useful
                else:
                    raise KnownFailureDidNotFailTest()
            # use `make_decorator` from nose to make sure that the meta-data on
            # the function is forwarded properly (name, teardown, setup, etc)
            return make_decorator(in_func)(inner_wrap)

        # if the condition is false, don't make a wrapper function
        # this is effectively a no-op
        else:
            return in_func

    # return the decorator function
    return dec


def skip_if(cond, msg=''):
    """
    A decorator to skip a test if condition is met
    """
    def dec(in_func):
        if cond:
            def wrapper():
                raise nose.SkipTest(msg)
            return make_decorator(in_func)(wrapper)
        else:
            return in_func
    return dec
