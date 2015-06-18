import signal
from weakref import ref, WeakKeyDictionary
import types
from inspect import Parameter, Signature
import itertools
from collections import OrderedDict, defaultdict, deque
import sys

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


class ScanValidator:
    def __init__(self, scan, run_engine):
        run_engine_state = ['']
        self.run_engine = run_engine
        self.scan = scan
        self.message_names = list(self.run_engine._command_registry.keys())
        self.message_counts = defaultdict(int)
        self.message_order = []
        self.exit_status = 'Not yet validated'
        self.configured = set()

    def _process_message(self, message):
        # increment the number of coun
        self.message_counts[message.command] += 1
        if message.command == 'checkpoint':
            # search backwards through history and make sure that we
            # find a "save" before we find a "create", or don't find a save
            # at all.
            # the zeroth message is the message that we are trying to process
            for msg in self.message_order[1:]:
                if msg.command == 'save':
                    # all is well
                    break
                if msg.command == 'create':
                    self.exit_status = ("'checkpoint' received after 'create' "
                                        "and before 'save'")
                    self.report()
                    # flush stdout so that the scan validation report is not
                    # interleaved with the exception
                    sys.stdout.flush()
                    raise ValueError("A 'checkpoint' message cannot occur "
                                     "between a create and a save. This is a "
                                     "flawed scan. Printing out a report and "
                                     "ceasing to process the scan.")
        if message.command == 'save':
            # search backwards through history and make sure that 'create' is
            # encountered first. Otherwise, raise!
            all_is_well = False
            # the zeroth message is the message that we are trying to process
            for msg in self.message_order[1:]:
                if msg.command == 'create':
                    # all is well
                    all_is_well = True
                    break
                elif msg.command == 'save':
                    # all is definitely not well
                    self.exit_status = ("Two 'save's were received "
                                        "without a 'create' in between.")
                    self.report()
                    # flush stdout so that the scan validation report is not
                    # interleaved with the exception
                    sys.stdout.flush()
                    raise ValueError("Two 'save' messages cannot be processed "
                                     "without a 'create' message occuring "
                                     "between them. This is a flawed scan. "
                                     "Printing out a report and ceasing to "
                                     "process the scan.")

            if not all_is_well:
                self.exit_status = ("There is no 'create' message that "
                                    "precedes this 'save' message")
                self.report()
                # flush stdout so that the scan validation report is not
                # interleaved with the exception
                sys.stdout.flush()
                raise ValueError("A 'save' message cannot be processed "
                                 "without having a 'create' before it.This "
                                 "is a flawed scan. Printing out a report "
                                 "and ceasing to process the scan.")

        if message.command == 'configure':
            if message.obj in self.configured:
                # then we have tried to configure a detector twice without
                # deconfiguring it first
                self.exit_status = (
                    "'configure' cannot be received twice in a row. "
                    "'deconfigure' must be called before 'configure' can be "
                    "called again.")
                self.report()
                sys.stdout.flush()
                raise ValueError(
                    "A second 'configure' request for object %s was received "
                    "without processing a 'deconfigure' for this object." %
                    message.obj)
            # otherwise, add it to the set of configured detectors
            self.configured.add(message.obj)

        if message.command == 'deconfigure':
            if message.obj not in self.configured:
                # that is a problem!
                self.exit_status = ("'deconfigure' received without a "
                                    "corresponding 'configure' first.")
                self.report()
                sys.stdout.flush()
                raise ValueError(
                    "A 'deconfigure' request for object %s was received "
                    "without a 'configure' request first." % message.obj)
            # otherwise, remove it from the set of configured detectors
            self.configured.remove(message.obj)

    def validate(self):
        self.exit_status = 'Not yet validated'
        for msg in self.scan:
            if msg.command not in self.message_names:
                raise KeyError(
                    "The RunEngine you provided does not have a callback "
                    "registered for message = {}".format(msg))
            self.message_order.insert(0, msg)
            self._process_message(msg)
        self.exit_status = "Success"

    def report(self):
        print("Scan Validation Report")
        print("----------------------")
        print("Exit status of %s = %s." % (self.scan, self.exit_status))
        print()
        print("Here are the number of times that each message was received.")
        from prettytable import PrettyTable
        p = PrettyTable(field_names=['Message Name', 'Times Called'])
        p.padding_width = 1
        p.align['Message Name'] = 'l'
        p.align['Times Called'] = 'r'

        for k, v in self.message_counts.items():
            p.add_row([k, v])
        print(p)

        print()
        print("Messages received (newest messages first)")
        for idx, msg in enumerate(self.message_order):
            print("%s: %s" % (idx, msg))


if __name__ == "__main__":
    from bluesky.examples import *
    from bluesky.utils import ScanValidator
    from bluesky.tests.utils import setup_test_run_engine
    RE = setup_test_run_engine()

    sv = ScanValidator(bad_checkpoint_scan(), RE)
    sv.validate()
