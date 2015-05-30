import signal
from weakref import ref, WeakKeyDictionary
import types


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
    See matplotlib.cbook.CallbackRegistry. This is simplified, being py3 only.
    """
    def __init__(self):
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

    def connect(self, s, func):
        """
        register *func* to be called when a signal *s* is generated
        func will be called
        """
        self._func_cid_map.setdefault(s, WeakKeyDictionary())
        # Note proxy not needed in python 3.
        # TODO rewrite this when support for python2.x gets dropped.
        proxy = _BoundMethodProxy(func)
        if proxy in self._func_cid_map[s]:
            return self._func_cid_map[s][proxy]

        proxy.add_destroy_callback(self._remove_proxy)
        self._cid += 1
        cid = self._cid
        self._func_cid_map[s][proxy] = cid
        self.callbacks.setdefault(s, dict())
        self.callbacks[s][cid] = proxy
        return cid

    def _remove_proxy(self, proxy):
        for sig, proxies in self._func_cid_map.items():
            try:
                del self.callbacks[sig][proxies[proxy]]
            except KeyError:
                pass

            if len(self.callbacks[sig]) == 0:
                del self.callbacks[sig]
                del self._func_cid_map[sig]


    def disconnect(self, cid):
        """
        disconnect the callback registered with callback id *cid*
        """
        for eventname, callbackd in self.callbacks.items():
            try:
                del callbackd[cid]
            except KeyError:
                continue
            else:
                for sig, functions in self._func_cid_map.items():
                    for function, value in functions.items():
                        if value == cid:
                            del functions[function]
                return

    def process(self, s, *args, **kwargs):
        """
        process signal *s*.  All of the functions registered to receive
        callbacks on *s* will be called with *\*args* and *\*\*kwargs*
        """
        if s in self.callbacks:
            for cid, proxy in self.callbacks[s].items():
                try:
                    proxy(*args, **kwargs)
                except ReferenceError:
                    self._remove_proxy(proxy)

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
