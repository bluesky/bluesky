import signal
import operator
from functools import reduce
from weakref import ref, WeakKeyDictionary
import types
from inspect import Parameter, Signature
import itertools
from collections import OrderedDict, Iterable
import sys
import numpy as np
from cycler import cycler 
import logging
logger = logging.getLogger(__name__)

__all__ = ['SignalHandler', 'CallbackRegistry', 'doc_type']



def doc_type(doc):
    """Determine if 'doc' is a 'start', 'stop', 'event' or 'descriptor'

    Returns
    -------
    {'start', 'stop', 'event', descriptor'}
    """
    # use an ordered dict to be a little faster with the assumption that
    # events are going to be the most common, then descriptors then
    # start/stops should come as pairs
    field_mapping = OrderedDict()
    field_mapping['event'] = ['seq_num', 'data']
    field_mapping['descriptor'] = ['data_keys', 'run_start']
    field_mapping['start'] = ['scan_id', 'beamline_id']
    field_mapping['stop'] = ['reason', 'exit_status']

    for doc_type, required_fields in field_mapping.items():
        could_be_this_one = True
        for field in required_fields:
            try:
                doc[field]
            except KeyError:
                could_be_this_one = False
        if could_be_this_one:
            logger.debug('document is a %s' % doc_type)
            return doc_type

    raise ValueError("Cannot determine the document type. Document I was "
                     "given:\n=====\n{}\n=====".format(doc))


class SignalHandler:
    def __init__(self, sig):
        self.sig = sig

    def __enter__(self):
        self.interrupted = False
        self.released = False
        self.original_handler = signal.getsignal(self.sig)

        def handler(signum, frame):
            self.release()
            self.interrupted = True

        signal.signal(self.sig, handler)
        return self

    def __exit__(self, type, value, tb):
        self.release()

    def release(self):
        if self.released:
            return False
        signal.signal(self.sig, self.original_handler)
        self.released = True
        return True


class CallbackRegistry:
    """
    See matplotlib.cbook.CallbackRegistry. This is a simplified since
    ``bluesky`` is python3.4+ only!
    """
    def __init__(self, halt_on_exception=True, allowed_sigs=None):
        self.halt_on_exception = halt_on_exception
        self.allowed_sigs = allowed_sigs
        self.callbacks = dict()
        self._cid = 0
        self._func_cid_map = {}

    def __getstate__(self):
        # We cannot currently pickle the callables in the registry, so
        # return an empty dictionary.
        return {}

    def __setstate__(self, state):
        # re-initialise an empty callback registry
        self.__init__()

    def connect(self, sig, func):
        """Register ``func`` to be called when ``sig`` is generated

        Parameters
        ----------
        sig
        func

        Returns
        -------
        cid : int
            The callback index. To be used with ``disconnect`` to deregister
            ``func`` so that it will no longer be called when ``sig`` is
            generated
        """
        if self.allowed_sigs is not None:
            if sig not in self.allowed_sigs:
                raise ValueError("Allowed signals are {0}".format(
                    self.allowed_sigs))
        self._func_cid_map.setdefault(sig, WeakKeyDictionary())
        # Note proxy not needed in python 3.
        # TODO rewrite this when support for python2.x gets dropped.
        proxy = _BoundMethodProxy(func)
        if proxy in self._func_cid_map[sig]:
            return self._func_cid_map[sig][proxy]

        proxy.add_destroy_callback(self._remove_proxy)
        self._cid += 1
        cid = self._cid
        self._func_cid_map[sig][proxy] = cid
        self.callbacks.setdefault(sig, dict())
        self.callbacks[sig][cid] = proxy
        return cid

    def _remove_proxy(self, proxy):
        # need the list because `del self._func_cid_map[sig]` mutates the dict
        for sig, proxies in list(self._func_cid_map.items()):
            try:
                del self.callbacks[sig][proxies[proxy]]
            except KeyError:
                pass

            if len(self.callbacks[sig]) == 0:
                del self.callbacks[sig]
                del self._func_cid_map[sig]

    def disconnect(self, cid):
        """Disconnect the callback registered with callback id *cid*

        Parameters
        ----------
        cid : int
            The callback index and return value from ``connect``
        """
        for eventname, callbackd in self.callbacks.items():
            try:
                del callbackd[cid]
            except KeyError:
                continue
            else:
                for sig, functions in self._func_cid_map.items():
                    for function, value in list(functions.items()):
                        if value == cid:
                            del functions[function]
                return

    def process(self, sig, *args, **kwargs):
        """Process ``sig``

        All of the functions registered to receive callbacks on ``sig``
        will be called with ``args`` and ``kwargs``

        Parameters
        ----------
        sig
        args
        kwargs
        """
        if self.allowed_sigs is not None:
            if sig not in self.allowed_sigs:
                raise ValueError("Allowed signals are {0}".format(
                    self.allowed_sigs))
        exceptions = []
        if sig in self.callbacks:
            for cid, func in self.callbacks[sig].items():
                try:
                    func(*args, **kwargs)
                except ReferenceError:
                    self._remove_proxy(func)
                except Exception as e:
                    if self.halt_on_exception:
                        raise
                    else:
                        exceptions.append((e, sys.exc_info()[2]))
        return exceptions


class _BoundMethodProxy:
    '''
    Our own proxy object which enables weak references to bound and unbound
    methods and arbitrary callables. Pulls information about the function,
    class, and instance out of a bound method. Stores a weak reference to the
    instance to support garbage collection.
    @organization: IBM Corporation
    @copyright: Copyright (c) 2005, 2006 IBM Corporation
    @license: The BSD License
    Minor bugfixes by Michael Droettboom
    '''
    def __init__(self, cb):
        self._hash = hash(cb)
        self._destroy_callbacks = []
        try:
            try:
                self.inst = ref(cb.__self__, self._destroy)
            except TypeError:
                self.inst = None
            self.func = cb.__func__
            self.klass = cb.__self__.__class__
        except AttributeError:
            self.inst = None
            self.func = cb
            self.klass = None

    def add_destroy_callback(self, callback):
        self._destroy_callbacks.append(_BoundMethodProxy(callback))

    def _destroy(self, wk):
        for callback in self._destroy_callbacks:
            try:
                callback(self)
            except ReferenceError:
                pass

    def __getstate__(self):
        d = self.__dict__.copy()
        # de-weak reference inst
        inst = d['inst']
        if inst is not None:
            d['inst'] = inst()
        return d

    def __setstate__(self, statedict):
        self.__dict__ = statedict
        inst = statedict['inst']
        # turn inst back into a weakref
        if inst is not None:
            self.inst = ref(inst)

    def __call__(self, *args, **kwargs):
        '''
        Proxy for a call to the weak referenced object. Take
        arbitrary params to pass to the callable.
        Raises `ReferenceError`: When the weak reference refers to
        a dead object
        '''
        if self.inst is not None and self.inst() is None:
            raise ReferenceError
        elif self.inst is not None:
            # build a new instance method with a strong reference to the
            # instance

            mtd = types.MethodType(self.func, self.inst())

        else:
            # not a bound method, just return the func
            mtd = self.func
        # invoke the callable and return the result
        return mtd(*args, **kwargs)

    def __eq__(self, other):
        '''
        Compare the held function and instance with that held by
        another proxy.
        '''
        try:
            if self.inst is None:
                return self.func == other.func and other.inst is None
            else:
                return self.func == other.func and self.inst() == other.inst()
        except Exception:
            return False

    def __ne__(self, other):
        '''
        Inverse of __eq__.
        '''
        return not self.__eq__(other)

    def __hash__(self):
        return self._hash


# The following three code blocks are from David Beazley's
# 'Python 3 Metaprogramming' https://www.youtube.com/watch?v=sPiWg5jSoZI

def make_signature(names):
    return Signature(Parameter(name, Parameter.POSITIONAL_OR_KEYWORD)
                     for name in names)


class StructMeta(type):
    def __new__(cls, name, bases, clsdict):
        clsobj = super().__new__(cls, name, bases, clsdict)
        sig = make_signature(clsobj._fields)
        setattr(clsobj, '__signature__', sig)
        return clsobj


class Struct(metaclass=StructMeta):
    "The _fields of any subclass become its attritubes and __init__ args."
    _fields = []
    def __init__(self, *args, **kwargs):
        bound = self.__signature__.bind(*args, **kwargs)
        for name, val in bound.arguments.items():
            setattr(self, name, val)

    def set(self, **kwargs):
        "Update attributes as keyword arguments."
        for attr, val in kwargs.items():
            setattr(self, attr, val)


class ExtendedList(list):
    "A list with some 'required' elements that can't be removed."
    # Elaborated version of http://stackoverflow.com/a/16380637/1221924
    def __init__(self, other=None):
        self.other = other or []

    def __len__(self):
        return list.__len__(self) + len(self.other)

    def __iter__(self):
        return itertools.chain(list.__iter__(self), iter(self.other))

    def __getitem__(self, index):
        l = list.__len__(self)

        if index > l:
            return self.other[index - l]
        else:
            return list.__getitem__(self, index)

    def __contains__(self, value):
        return super().__contains__(value) or (value in self.other)

    def remove(self, value):
        if (not super().__contains__(value)) and (value in self):
            raise ValueError('%s is mandatory and cannot be removed' % value)
        super().remove(value)


def normalize_subs_input(subs):
    "Accept a callable, a list, or a dict. Normalize to a dict."
    if subs is None:
        return {}
    if hasattr(subs, 'items'):
        return subs
    elif isinstance(subs, Iterable):
        return {'all': subs}
    elif callable(subs):
        return {'all': [subs]}
    else:
        raise ValueError("Subscriptions should be a callable, a list of "
                         "callables, or a dictionary mapping subscription "
                         "names to lists of callables.")


def snake_cyclers(cyclers, snake_booleans):
    """
    Combine cyclers with a 'snaking' back-and-forth order.

    Parameters
    ----------
    cyclers : cycler.Cycler
        or any iterable that yields dictionaries of lists
    snake_booleans : list
        a list of the same length as cyclers indicating whether each cycler
        should 'snake' (True) or not (False). Note that the first boolean
        does not make a difference because the first (slowest) dimension
        does not repeat.

    Returns
    -------
    result : cycler
    """
    if len(cyclers) != len(snake_booleans):
        raise ValueError("number of cyclers does not match number of booleans")
    lengths = []
    new_cyclers = []
    for c in cyclers:
        lengths.append(len(c))
    total_length = np.product(lengths)
    for i, (c, snake) in enumerate(zip(cyclers, snake_booleans)):
        num_tiles = np.product(lengths[:i])
        num_repeats = np.product(lengths[i+1:])
        for k, v in c._transpose().items():
            if snake:
                v = v + v[::-1]
            v2 = np.tile(np.repeat(v, num_repeats), num_tiles)
            expanded = v2[:total_length]
            new_cyclers.append(cycler(k, expanded))
    return reduce(operator.add, new_cyclers)
