import time as ttime
import sys
from itertools import count
from collections import namedtuple, deque, defaultdict
import uuid
import signal
import threading
from queue import Queue, Empty
from enum import Enum

from super_state_machine.machines import StateMachine
import numpy as np

from .utils import CallbackRegistry, SignalHandler


__all__ = ['Msg', 'Base', 'Reader', 'Mover', 'SynGauss', 'FlyMagic',
           'RunEngineStateMachine', 'RunEngine', 'Dispatcher',
           'PanicStateError', 'RunInterrupt']

# todo boo, hardcoded defaults
beamline_id = 'test'
owner = 'tester'
custom = {}
scan_id = 123


class Msg(namedtuple('Msg_base', ['command', 'obj', 'args', 'kwargs'])):
    __slots__ = ()

    def __new__(cls, command, obj=None, *args, **kwargs):
        return super(Msg, cls).__new__(cls, command, obj, args, kwargs)

    def __repr__(self):
        return '{}: ({}), {}, {}'.format(
            self.command, self.obj, self.args, self.kwargs)


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
            data[k] = {'value': self._cnt, 'timestamp': ttime.time()}
            self._cnt += 1

        return data

    def trigger(self):
        pass


class Mover(Base):
    _klass = 'mover'

    def __init__(self, *args, **kwargs):
        super(Mover, self).__init__(*args, **kwargs)
        self._data = {'pos': {'value': 0, 'timestamp': ttime.time()}}
        self.ready = True

    def read(self):
        return self._data

    def set(self, val, *, trigger=True, block_group=None):
        # If trigger is False, wait for a separate 'trigger' command to move.
        if not trigger:
            raise NotImplementedError
        # block_group is handled by the RunEngine
        self.ready = False
        ttime.sleep(0.1)  # simulate moving time
        self._data = {'pos': {'value': val, 'timestamp': ttime.time()}}
        self.ready = True

    def settle(self):
        pass


class SynGauss(Reader):
    """
    Evaluate a point on a Gaussian based on the value of a motor.

    Example
    -------
    motor = Mover('motor', ['pos'])
    det = SynGauss('sg', motor, 'pos', center=0, Imax=1, sigma=1)
    """
    _klass = 'reader'

    def __init__(self, name, motor, motor_field, center, Imax, sigma=1):
        super(SynGauss, self).__init__(name, 'I')
        self.ready = True
        self._motor = motor
        self._motor_field = motor_field
        self.center = center
        self.Imax = Imax
        self.sigma = sigma

    def trigger(self, *, block_group=True):
        self.ready = False
        m = self._motor._data[self._motor_field]['value']
        v = self.Imax * np.exp(-(m - self.center)**2 / (2 * self.sigma**2))
        self._data = {'intensity': {'value': v, 'timestamp': ttime.time()}}
        ttime.sleep(0.05)  # simulate exposure time
        self.ready = True

    def read(self):
        return self._data


class FlyMagic(Base):
    _klass = 'flyer'

    def __init__(self, name, motor, det, scan_points=15):
        super(FlyMagic, self).__init__(name, [motor, det])
        self._motor = motor
        self._det = det
        self._scan_points = scan_points
        self._time = None
        self._fly_count = 0

    def reset(self):
        self._fly_count = 0

    def kickoff(self):
        self._time = ttime.time()
        self._fly_count += 1

    def collect(self):
        if self._time is None:
            raise RuntimeError("Must kick off flyscan before you collect")

        dtheta = (np.pi / 10) * self._fly_count
        X = np.linspace(0, 2*np.pi, self._scan_points)
        Y = np.sin(X + dtheta)
        dt = (ttime.time() - self._time) / self._scan_points
        T = dt * np.arange(self._scan_points) + self._time

        for j, (t, x, y) in enumerate(zip(T, X, Y)):
            ev = {'time': t,
                  'data': {self._motor: x,
                           self._det: y},
                  'timestamps': {self._motor: t,
                                 self._det: t}
                  }

            yield ev
        self._time = None


class RunEngineStateMachine(StateMachine):
    """

    Attributes
    ----------
    idle
        State machine is in its idle state
    running
        State machine is in its running state
    aborting
        State machine has been asked to abort but is not yet ``idle``
    soft_pausing
        State machine has been asked to enter a soft pause but is not yet in
        its ``paused`` state
    hard_pausing
        State machine has been asked to enter a hard pause but is not yet in
        its ``paused`` state
    paused
        State machine is paused.
    panicked
        State machine has encountered a critical error and the run cannot be
        resumed.

    Example
    -------
    # Using the state machine See ``examples/state_machine.py`` or
    # ``examples/bluesky state machine.ipynb``
    >>> sm = RunEngineStateMachine()
    >>> sm.state
    'idle'

    # changing states
    # sm.set_[[state.value]]()
    >>> sm.set_running()
    >>> sm.state()
    'running'

    # or use a named transition
    >>> sm.stop()
    >>> sm.state()
    'idle'

    # or assign a new state directly to the state of the state machine
    >>> sm.state = 'panicked'
    >>> sm.state
    'panicked'
    >>> sm.all_is_well()
    >>> sm.state
    'idle'

    # or use the minimum amount of information necessary to change states
    >>> sm.state = 'pan'
    >>> sm.state
    'panicked'
    >>> sm.state = 'i'
    >>> sm.state
    'idle'

    """

    class States(Enum):
        """state.name = state.value"""
        IDLE = 'idle'
        RUNNING = 'running'
        ABORTING = 'aborting'
        SOFT_PAUSING = 'soft_pausing'
        HARD_PAUSING = 'hard_pausing'
        PAUSED = 'paused'
        PANICKED = 'panicked'

        @classmethod
        def states(cls):
            return [state.value for state in cls]

    class Meta:
        allow_empty = False
        initial_state = 'idle'
        transitions = {
            # Notice that 'transitions' and 'named_transitions' have
            # opposite to <--> from structure.
            # from_state : [valid_to_states]
            'idle': ['running', 'panicked'],
            'running': ['idle', 'soft_pausing', 'hard_pausing', 'panicked'],
            'aborting': ['idle', 'panicked'],
            'soft_pausing': ['paused', 'panicked'],
            'hard_pausing': ['paused', 'panicked'],
            'paused': ['aborting', 'running', 'panicked'],
            'panicked': ['idle'],
        }
        named_transitions = [
            # (transition_name, to_state : [valid_from_states])
            ('soft_pause', 'soft_pausing', ['running']),
            ('hard_pause', 'hard_pausing', ['running', 'soft_pausing']),
            ('pause', 'paused', ['soft_pausing', 'hard_pausing']),
            ('run', 'running', ['idle', 'paused']),
            ('stop', 'idle', ['running', 'aborting']),
            ('panic', 'panicked'),
            ('resume', 'running', ['paused']),
            ('abort', 'aborting', ['paused']),
            ('all_is_well', 'idle', ['panicked', 'idle']),
        ]
        named_checkers = [
            ('panicked', 'panicked'),
            ('paused', 'paused'),
            ('hard_pausing', 'hard_pausing'),
            ('soft_pausing', 'soft_pausing'),
            ('aborting', 'aborting'),
            ('running', 'running'),
            ('idle', 'idle'),
        ]


class RunEngine(object):
    def __init__(self):
        super(RunEngine, self).__init__()
        self._sm = RunEngineStateMachine()
        self.state = self._sm.state
        self._sigint_handler = None
        self._sigtstp_handler = None
        self._pause_requests = deque()  # holding {<name>: callable}
        self._objs_read = deque()  # objects read in one Event
        self._read_cache = deque()  # cache of obj.read() in one Event
        self._describe_cache = dict()  # cache of all obj.describe() output
        self._descriptor_uids = dict()  # cache of all Descriptor uids
        self._sequence_counters = dict()  # a seq_num counter per Descriptor
        self._block_groups = defaultdict(set)  # sets of objs to wait for
        self._temp_callback_ids = set()  # ids from CallbackRegistry
        self._msg_cache = None  # may be used to hold recently processed msgs
        self._command_registry = {
            'create': self._create,
            'save': self._save,
            'read': self._read,
            'null': self._null,
            'set': self._set,
            'trigger': self._trigger,
            'sleep': self._sleep,
            'wait': self._wait,
            'checkpoint': self._checkpoint,
            'pause': self._pause,
            'collect': self._collect,
            'kickoff': self._kickoff
            }

        # queues for passing Documents from "scan thread" to main thread
        queue_names = ['start', 'stop', 'event', 'descriptor']
        self._queues = {name: Queue() for name in queue_names}

        # public dispatcher for callbacks processed on the main thread
        self.dispatcher = Dispatcher(self._queues)
        self.subscribe = self.dispatcher.subscribe
        self.unsubscribe = self.dispatcher.unsubscribe

        # For why this function is necessary, see
        # http://stackoverflow.com/a/13355291/1221924
        def make_push_func(name):
            return lambda doc: self._push_to_queue(name, doc)

        # private registry of callbacks processed on the "scan thread"
        self._scan_cb_registry = CallbackRegistry()
        for name in self._queues.keys():
            self._register_scan_callback(name, make_push_func(name))

        self.verbose = True

    def clear(self):
        self._objs_read.clear()
        self._read_cache.clear()
        self._describe_cache.clear()
        self._descriptor_uids.clear()
        self._sequence_counters.clear()
        self._msg_cache = None
        # Unsubscribe for per-run callbacks.
        for cid in self._temp_callback_ids:
            self.unsubscribe(cid)
        self._temp_callback_ids.clear()

    def register_command(self, name, func):
        self._command_registry[name] = func

    def unregister_command(self, name):
        del self._command_registry[name]

    def panic(self):
        # Release GIL by sleeping, allowing other threads to set panic.
        ttime.sleep(0.01)
        self._sm.panic()

    def all_is_well(self):
        self._sm.all_is_well()

    def request_pause(self, hard=True, name=None, callback=None):
        """
        Command the Run Engine to pause.

        This function is called by 'pause' Messages. It can also be called
        by other threads. It cannot be called on the main thread during a run,
        but it is called by SIGINT (i.e., Ctrl+C).

        If there current run has no checkpoint commands, this will cause the
        run to abort.

        Parameters
        ----------
        hard : bool, optional
            If True, issue a 'hard pause' that stops at the next message.
            If False, issue a 'soft pause' that stops at the next checkpoint.
            True by default.
        name : str, optional
            Identify the source/reason for this pause. Required if there is a
            callback, below.
        callback : callable, optional
            "Permission to resume." Until this callable returns True, the Run
            Engine will not be allowed to resume. If None,
        """
        if hard:
            self._sm.hard_pause()
        else:
            if not self._sm.is_hard_pausing:
                self._sm.soft_pause()
        if callback is not None:
            if name is None:
                raise ValueError("Pause requests with a callback must include "
                                 "a name.")
            self._pause_requests.append({name: callback})

    def _register_scan_callback(self, name, func):
        """Register a callback to be processed by the scan thread.

        Functions registered here are guaranteed to be run (there is no Queue
        involved) and they block the scan's progress until they return.
        """
        return self._scan_cb_registry.connect(name, func)

    def _push_to_queue(self, name, doc):
        self._queues[name].put(doc)

    # todo remove the mutable from this function sig!!
    def run(self, gen, subscriptions={}, use_threading=True):
        """Run the scan defined by ``gen``

        Parameters
        ----------
        gen : generator
            A generator that yields ``Msg`` objects
        subscriptions : dict
            Temporary subscriptions to be added for this call to ``run()`` only
        use_threading : bool
            duh.
        """
        self._sm.run()
        self.clear()
        for name, func in subscriptions.items():
            self._temp_callback_ids.add(self.subscribe(name, func))
        self._run_start_uid = new_uid()
        if self._sm.is_panicked:
            raise PanicStateError("RunEngine is in a panic state. The run "
                                  "was aborted before it began. No records "
                                  "of this run were created.")
        with SignalHandler(signal.SIGINT) as self._sigint_handler:  # ^C
            def func():
                return self.run_engine(gen)
            if use_threading:
                self._thread = threading.Thread(target=func)
                self._thread.start()
                while self._thread.is_alive() and not self._sm.is_paused:
                    self.dispatcher.process_all_queues()
            else:
                func()
                self.dispatcher.process_all_queues()
            self.dispatcher.process_all_queues()  # catch any stragglers

    def resume(self):
        """Resume a run from the last checkpoint.

        Returns
        -------
        outstanding_pause_requests : list or None
            If any pause requests have not been released, a list of their names
            is immediately returned. Otherwise, after the run completes, this
            returns None.
        """
        outstanding_requests = []
        for name, func in self._pause_requests.items():
            if func():
                # We have permission to continue. Clear the request.
                del self._pause_requests[name]
            else:
                outstanding_requests.append(name)
        if outstanding_requests:
            return outstanding_requests
        self._sm.resume()
        self._resume()
        return None

    def _resume(self):
        # This could be as a result of self.resume() or self.abort().
        with SignalHandler(signal.SIGINT) as self._sigint_handler:  # ^C
            while self._thread.is_alive() and not self._sm.is_paused:
                self.dispatcher.process_all_queues()
        self.dispatcher.process_all_queues()  # catch any stragglers

    def abort(self):
        self._sm.abort()
        self._resume()  # to create RunStop Document

    def run_engine(self, gen):
        # This function is optionally run on its own thread.
        doc = dict(uid=self._run_start_uid,
                   time=ttime.time(), beamline_id=beamline_id, owner=owner,
                   scan_id=scan_id, **custom)
        self.debug("*** Emitted RunStart:\n%s" % doc)
        self.emit('start', doc)
        response = None
        exit_status = None
        reason = ''
        try:
            while True:
                # self.debug('MSG_CACHE', self._msg_cache)
                # Check for panic.
                if self._sm.is_panicked:
                    exit_status = 'fail'
                    raise PanicStateError("Something put the RunEngine into a "
                                          "panic state after the run began. "
                                          "Records were created, but the run "
                                          "was marked with "
                                          "exit_status='fail'.")

                # Check for pause requests from keyboard.
                if self._sigint_handler.interrupted:
                    self.debug("RunEngine detected a SIGINT (Ctrl+C)")
                    self.request_pause(hard=True, name='SIGINT')
                    self._sigint_handler.interrupted = False

                # If a hard pause was requested, sleep.
                if self._sm.is_hard_pausing:
                    self._sm.pause()
                    if self._msg_cache is None:
                        self._sm.abort()
                        exit_status = 'abort'
                        raise RunInterrupt("*** Hard pause requested. There "
                                           "are no checkpoints. Cannot resume;"
                                           " must abort. Run aborted.")
                    self.debug("*** Hard pause requested. Sleeping until "
                               "resume() is called. "
                               "Will rerun from last 'checkpoint' command.")
                    while True:
                        ttime.sleep(0.5)
                        if not self._sm.is_paused:
                            break
                    if self._sm.is_aborting:
                        exit_status = 'abort'
                        raise RunInterrupt("Run aborted.")
                    self._rerun_from_checkpoint()

                # If a soft pause was requested, acknowledge it, but wait
                # for a 'checkpoint' command to catch it (see self._checkpoint).
                if self._sm.is_soft_pausing:
                    self.debug("*** Soft pause requested. Continuing to "
                               "process messages until the next 'checkpoint' "
                               "command.")

                # Normal operation
                msg = gen.send(response)
                if self._msg_cache is not None:
                    # We have a checkpoint.
                    self._msg_cache.append(msg)
                response = self._command_registry[msg.command](msg)
                self.debug('RE.state: ' + self.state)

                self.debug('{}\n   ret: {}'.format(msg, response))
        except StopIteration:
            exit_status = 'success'
        except Exception as err:
            exit_status = 'fail'
            reason = str(err)
            raise err
        finally:
            doc = dict(run_start=self._run_start_uid,
                       time=ttime.time(),
                       exit_status=exit_status,
                       reason=reason)
            self.emit('stop', doc)
            self.debug("*** Emitted RunStop:\n%s" % doc)
            sys.stdout.flush()
            if self._sm.is_aborting or self._sm.is_running:
                self._sm.stop()

    def _create(self, msg):
        self._read_cache.clear()
        self._objs_read.clear()

    def _read(self, msg):
        obj = msg.obj
        self._objs_read.append(obj)
        if obj not in self._describe_cache:
            self._describe_cache[obj] = obj.describe()
        ret = obj.read(*msg.args, **msg.kwargs)
        self._read_cache.append(ret)
        return ret

    def _save(self, msg):
        # The Event Descriptor is uniquely defined by the set of objects
        # read in this Event grouping.
        objs_read = frozenset(self._objs_read)

        # Event Descriptor
        if objs_read not in self._descriptor_uids:
            # We don't not have an Event Descriptor for this set.
            data_keys = {}
            [data_keys.update(self._describe_cache[obj]) for obj in objs_read]
            _fill_missing_fields(data_keys)  # TODO Move this to ophyd/controls.
            descriptor_uid = new_uid()
            doc = dict(run_start=self._run_start_uid, time=ttime.time(),
                       data_keys=data_keys, uid=descriptor_uid)
            self.emit('descriptor', doc)
            self.debug("*** Emitted Event Descriptor:\n%s" % doc)
            self._descriptor_uids[objs_read] = descriptor_uid
            self._sequence_counters[objs_read] = count(1)
        else:
            descriptor_uid = self._descriptor_uids[objs_read]

        # Events
        seq_num = next(self._sequence_counters[objs_read])
        event_uid = new_uid()
        # Merge list of readings into single dict.
        readings = {k: v for d in self._read_cache for k, v in d.items()}
        for key in readings:
            readings[key]['value'] = _sanitize_np(readings[key]['value'])
        data, timestamps = _rearrange_into_parallel_dicts(readings)
        doc = dict(descriptor=descriptor_uid,
                   time=ttime.time(), data=data, timestamps=timestamps,
                   seq_num=seq_num, uid=event_uid)
        self.emit('event', doc)
        self.debug("*** Emitted Event:\n%s" % doc)

    def _kickoff(self, msg):
        return msg.obj.kickoff(*msg.args, **msg.kwargs)

    def _collect(self, msg):
        obj = msg.obj
        if obj not in self._describe_cache:
            self._describe_cache[obj] = obj.describe()

        obj_read = frozenset((obj,))
        if obj_read not in self._descriptor_uids:
            # We don't not have an Event Descriptor for this set.
            data_keys = obj.describe()
            descriptor_uid = new_uid()
            doc = dict(run_start=self._run_start_uid, time=ttime.time(),
                       data_keys=data_keys, uid=descriptor_uid)
            self.emit('descriptor', doc)
            self.debug("Emitted Event Descriptor:\n%s" % doc)
            self._descriptor_uids[obj_read] = descriptor_uid
            self._sequence_counters[obj_read] = count(1)
        else:
            descriptor_uid = self._descriptor_uids[obj_read]

        for ev in obj.collect():
            seq_num = next(self._sequence_counters[obj_read])
            event_uid = new_uid()
            reading = ev['data']
            for key in ev['data']:
                reading[key] = _sanitize_np(reading[key])
            ev['data'] = reading
            ev['descriptor'] = descriptor_uid
            ev['seq_num'] = seq_num
            ev['uid'] = event_uid
            self.emit('event', ev)
            self.debug("Emitted Event:\n%s" % ev)

    def _null(self, msg):
        pass

    def _set(self, msg):
        if 'block_group' in msg.kwargs:
            group = msg.kwargs['block_group']
            self._block_groups[group].add(msg.obj)
        return msg.obj.set(*msg.args, **msg.kwargs)

    def _trigger(self, msg):
        if 'block_group' in msg.kwargs:
            group = msg.kwargs['block_group']
            self._block_groups[group].add(msg.obj)
        return msg.obj.trigger(*msg.args, **msg.kwargs)

    def _wait(self, msg):
        # Block progress until every object that was trigged
        # triggered with the keyword argument `block=group` is done.
        group = msg.kwargs.get('group', msg.args[0])
        objs = self._block_groups[group]
        while True:
            if all([obj.ready for obj in objs]):
                break
        del self._block_groups[group]
        return objs

    def _sleep(self, msg):
        return ttime.sleep(*msg.args)

    def _pause(self, msg):
        self.request_pause(*msg.args, **msg.kwargs)

    def _checkpoint(self, msg):
        self._msg_cache = deque()
        if self._sm.is_soft_pausing:
            self._sm.pause()  # soft_pausing -> paused
            self.debug("*** Checkpoint reached. Sleeping until resume() is "
                       "called. Will resume from checkpoint.")
            while True:
                ttime.sleep(0.5)
                if not self._sm.is_paused:
                    break

    def _rerun_from_checkpoint(self):
        self.debug("*** Rerunning from checkpoint...")
        for msg in self._msg_cache:
            response = self._command_registry[msg.command](msg)
            self.debug('{}\n   ret: {} (On rerun, responses are not sent.)'
                       ''.format(msg, response))

    def emit(self, name, doc):
        self._scan_cb_registry.process(name, doc)

    def debug(self, msg):
        if self.verbose:
            print(msg)


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

    def process_all_queues(self):
        for name in self.queues.keys():
            self.process_queue(name)

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


def new_uid():
    return str(uuid.uuid4())


def _sanitize_np(val):
    "Convert any numpy objects into built-in Python types."
    if isinstance(val, np.generic):
        if np.isscalar(val):
            return val.item()
        return val.tolist()
    return val


def _rearrange_into_parallel_dicts(readings):
    data = {}
    timestamps = {}
    for key, payload in readings.items():
        data[key] = payload['value']
        timestamps[key] = payload['timestamp']
    return data, timestamps


def _fill_missing_fields(data_keys):
    """This is a stop-gap until all describe() methods are complete."""
    result = {}
    for key, value in data_keys.items():
        result[key] = {}
        # required keys
        result[key]['source'] = value.get('source')
        result[key]['dtype'] = value.get('dtype', 'number')  # just guessing
        # optional keys
        if 'shape' in value:
            result[key]['shape'] = value['shape']
        if 'external' in value:
            result[key]['external'] = value['external']


class PanicStateError(Exception):
    pass


class RunInterrupt(KeyboardInterrupt):
    pass
