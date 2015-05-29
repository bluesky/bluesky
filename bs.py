import time as ttime
from itertools import count
from collections import namedtuple, deque
import uuid
import signal
import threading
from queue import Queue, Empty
from weakref import ref, WeakKeyDictionary
import numpy as np

beamline_id='test'
owner='tester'
custom = {}
scan_id = 123


class Msg(namedtuple('Msg_base', ['message', 'obj', 'args', 'kwargs'])):
    __slots__ = ()

    def __repr__(self):
        return '{}: ({}), {}, {}'.format(
            self.message, self.obj, self.args, self.kwargs)


class Base:
    def __init__(self, name, fields):
        self._name = name
        self._fields = fields

    def describe(self):
        return {k: {'source': self._name, 'dtype': 'number'}
                for k in self._fields}

    def __repr__(self):
        return '{}: {}'.format(self._klass, self._name)


class Reader(Base):
    _klass = 'reader'

    def __init__(self, *args, **kwargs):
        super(Reader, self).__init__(*args, **kwargs)
        self._cnt = 0

    def read(self):
        data = dict()
        for k in self._fields:
            data[k] = self._cnt
            self._cnt += 1

        return data

    def trigger(self, *, blocking=True):
        if blocking:
            pass


class Mover(Base):
    _klass = 'mover'

    def __init__(self, *args, **kwargs):
        super(Mover, self).__init__(*args, **kwargs)
        self._data = {k: 0 for k in self._fields}
        self._staging = None

    def read(self):
        return dict(self._data)

    def set(self, new_values):
        if set(new_values) - set(self._data):
            raise ValueError('setting non-existent field')
        self._staging = new_values

    def trigger(self, *, blocking=True):
        if blocking:
            pass

        if self._staging:
            for k, v in self._staging.items():
                self._data[k] = v

        self._staging = None

    def settle(self):
        pass


class SynGaus(Mover):

    def __init__(self, name, motor_name, det_name, center, Imax, sigma=1):
        super(SynGaus, self).__init__(name, (motor_name, det_name))
        self._motor = motor_name
        self._det = det_name
        self.center = center
        self.Imax = Imax
        self.sigma = sigma

    def set(self, new_values):
        if self._det in new_values:
            raise ValueError("you can't set the detector value")
        super(SynGaus, self).set(new_values)
        m = self._staging[self._motor]
        v = self.Imax * np.exp(-(m - self.center)**2 / (2 * self.sigma**2))
        self._staging[self._det] = v


class FlyMagic(Base):
    def kickoff(self):
        pass

    def collect(self):
        pass


def MoveRead_gen(motor, detector):
    try:
        for j in range(10):
            yield Msg('set', motor, ({'x': j}, ), {})
            yield Msg('trigger', motor, (), {})
            yield Msg('trigger', detector, (), {})
            yield Msg('read', detector, (), {})
            yield Msg('read', motor, (), {})
    finally:
        print('Generator finished')


def SynGaus_gen(syngaus):
    try:
        for x in np.linspace(-5, 5, 100):
            yield Msg('set', syngaus, ({'x': x}, ), {})
            yield Msg('trigger', syngaus, (), {})
            yield Msg('wait', None, (.1, ), {})
            yield Msg('read', syngaus, (), {})

    finally:
        print('generator finished')


class RunEngine:
    def __init__(self):
        self.panic = False
        self._sigint_handler = None
        self._read_cache = deque()
        self._proc_registry = {
            'create': self._create,
            'save': self._save,
            'read': self._read,
            'null': self._null,
            'set': self._set,
            'trigger': self._trigger,
            'wait': self._wait
            }

        # queues for passing Documents from "scan thread" to main thread
        self.event_queue = Queue()
        self.descriptor_queue = Queue()
        self.start_queue = Queue()
        self.stop_queue = Queue()
        queues = {'event': self.event_queue,
                  'descriptor': self.descriptor_queue,
                  'start': self.start_queue,
                  'stop': self.stop_queue}

        # public dispatcher for callbacks processed on the main thread
        self.dispatcher = Dispatcher(queues)

        # private registry of callbacks processed on the "scan thread"
        self._scan_cb_registry = CallbackRegistry()
        for name, queue in queues.items():
            curried_push = lambda doc: self._push_to_queue(queue, doc)
            self._register_scan_callback(name, curried_push)

    def _register_scan_callback(self, name, func):
        """Register a callback to be processed by the scan thread.

        Unlike subscribe, functions registered here will be processed on the
        scan thread. They are guaranteed to be run (there is no Queue
        involved) and they block the scan's progress until they return.
        Use subscribe for any non-critical functions.
        """
        return self._scan_cb_registry.connect(name, func)

    def _push_to_queue(self, queue, doc):
        queue.put(doc)

    def run(self, g, additional_callbacks=None, use_threading=True):
        self._read_cache.clear()
        self._run_start_uid = uid()
        with SignalHandler(signal.SIGINT) as self._sigint_handler:
            func = lambda: self.run_engine(g, additional_callbacks=None)
            if use_threading:
                thread = threading.Thread(target=func)
                thread.start()
            else:
                func()

    def run_engine(self, gen, additional_callbacks=None):
        # This function is optionally run on its own thread.
        doc = dict(uid=self._run_start_uid,
                time=ttime.time(), beamline_id=beamline_id, owner=owner,
                scan_id=scan_id, **custom)
        self.emit_start(doc)
        response = None
        exit_status = None
        reason = ''
        while True:
            try:
                if self.panic:
                    exit_status = 'fail'
                    break
                if self._sigint_handler.interrupted:
                    exit_status = 'abort'
                    break
                msg = gen.send(response)
                response = self._proc_registry[msg.message](msg)

                print('{}\n   ret: {}'.format(msg, response))
            except StopIteration:
                exit_status = 'success'
                break
            except Exception as err:
                exit_status = 'fail'
                reason = str(err)
                raise err
            finally:
                doc = dict(run_start=self._run_start_uid,
                        time=ttime.time(),
                        exit_status=exit_status,
                        reason=reason)
                self.emit_stop(doc)

    def _create(self, msg):
        self._read_cache.clear()

    def _read(self, msg):
        ret = msg.obj.read(*msg.args, **msg.kwargs)
        self._read_cache.append(ret)
        return ret

    def _save(self, msg):
        raise NotImplementedError()

    def _null(self, msg):
        pass

    def _set(self, msg):
        return msg.obj.set(*msg.args, **msg.kwargs)

    def _trigger(self, msg):
        return msg.obj.trigger(*msg.args, **msg.kwargs)

    def _wait(self, msg):
        return ttime.sleep(*msg.args)

    def emit_event(self, event):
        self._scan_cb_registry.process('event', event)

    def emit_descriptor(self, descriptor):
        self._scan_cb_registry.process('descriptor', descriptor)

    def emit_start(self, start):
        self._scan_cb_registry.process('start', start)

    def emit_stop(self, stop):
        self._scan_cb_registry.process('stop', stop)


class Dispatcher(object):
    """Dispatch documents to user-defined consumers on the main thread."""

    def __init__(self, queues, timeout=0.05):
        self.queues = queues
        self.timeout = timeout
        self.cb_registry = CallbackRegistry()

    def process_queue(self, name):
        queue = self.queues[name]
        try:
            document = queue.get(timeout=self.timeout)
        except Empty:
            pass
        else:
            self.cb_registry.process(name, document)
        print("Processed {name} subscriptions".format(name=name))

    def subscribe(self, name, func):
        """
        Register a function to consume Event documents.

        The Run Engine can execute callback functions at the start and end
        of a scan, and after the insertion of new Event Descriptors
        and Events.

        Parameters
        ----------
        name: {'start', 'descriptor', 'event', 'stop'}
        func: callable
            expecting signature like ``f(mongoengine.Document)``
        """
        if name not in self.queues.keys():
            raise ValueError("Valid callbacks: {0}".format(self.queues.keys()))
        return self.cb_registry.connect(name, func)

    def unsubscribe(self, callback_id):
        """
        Unregister a callback function using its integer ID.

        Parameters
        ----------
        callback_id : int
            the ID issued by `subscribe`
        """
        self.cb_registry.disconnect(callback_id)


class SignalHandler():
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


class CallbackRegistry():
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

class _BoundMethodProxy(object):
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


def uid():
    return str(uuid.uuid4())


RE = RunEngine()
