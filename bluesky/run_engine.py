import time as ttime
import sys
from itertools import count
from collections import namedtuple, deque, defaultdict, Iterable
import uuid
import signal
import threading
from queue import Queue, Empty
from enum import Enum

from super_state_machine.machines import StateMachine
from super_state_machine.extras import PropertyMachine
import numpy as np

from .utils import CallbackRegistry, SignalHandler


__all__ = ['Msg', 'Base', 'Reader', 'Mover', 'SynGauss', 'FlyMagic',
           'RunEngineStateMachine', 'RunEngine', 'Dispatcher',
           'RunInterrupt', 'PanicError', 'IllegalMessageSequence']

# todo boo, hardcoded defaults
beamline_id = 'test'
owner = 'tester'
scan_id = 123


class LossyLiFoQueue(Queue):
    '''Variant of Queue that is a 'lossy' first-in-last-out

    The queue size is strictly bounded at `maxsize - 1` by
    discarding old entries.
    '''
    def _init(self, maxsize):
        if maxsize < 2:
            raise ValueError("maxsize must be 2 or greater "
                             "for LossyLiFoQueue")
        self.queue = deque(maxlen=maxsize - 1)

    def _get(self):
        return self.queue.pop()


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
    is_idle
        State machine is in its idle state
    is_running
        State machine is in its running state
    is_aborting
        State machine has been asked to abort but is not yet ``idle``
    is_soft_pausing
        State machine has been asked to enter a soft pause but is not yet in
        its ``paused`` state
    is_hard_pausing
        State machine has been asked to enter a hard pause but is not yet in
        its ``paused`` state
    is_paused
        State machine is paused.
    """

    class States(Enum):
        """state.name = state.value"""
        IDLE = 'idle'
        RUNNING = 'running'
        ABORTING = 'aborting'
        SOFT_PAUSING = 'soft_pausing'
        HARD_PAUSING = 'hard_pausing'
        PAUSED = 'paused'

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
            'idle': ['running'],
            'running': ['idle', 'soft_pausing', 'hard_pausing'],
            'aborting': ['idle'],
            'soft_pausing': ['paused', 'soft_pausing', 'hard_pausing'],
            'hard_pausing': ['paused', 'hard_pausing'],
            'paused': ['aborting', 'running'],
        }
        named_transitions = [
            # (transition_name, to_state : [valid_from_states])
            ('soft_pause', 'soft_pausing', ['running', 'soft_pausing']),
            ('hard_pause', 'hard_pausing', ['running', 'soft_pausing', 'hard_pausing']),
            ('pause', 'paused', ['soft_pausing', 'hard_pausing']),
            ('run', 'running', ['idle', 'paused']),
            ('stop', 'idle', ['running', 'aborting']),
            ('resume', 'running', ['paused']),
            ('abort', 'aborting', ['paused']),
        ]
        named_checkers = [
            ('can_hard_pause', 'hard_pausing'),
            ('can_soft_pause', 'soft_pausing'),
        ]


class RunEngine:

    state = PropertyMachine(RunEngineStateMachine)
    _REQUIRED_FIELDS = ['beamline_id', 'owner']

    def __init__(self, memory=None):
        """
        The Run Engine execute messages and emits Documents.

        Parameters
        ----------
        memory : dict-like
            The default is a standard Python dictionary, but fancier objects
            can be used to store long-term history and persist it between
            sessions. The standard configuration instantiates a Run Engine with
            history.History, a simple interface to a sqlite file. Any object
            supporting `__getitem__`, `__setitem__`, and `clear` will work.

        Attributes
        ----------
        state
            {'idle', 'running', 'soft_pausing, 'hard_pausing', 'paused',
             'aborting'}
        memory
            direct access to the dict-like persistent storage described above
        persistent_fields
            list of metadata fields that will be remembered and reused between
            subsequence runs

        Methods
        -------
        request_pause
            Pause the Run Engine at the next checkpoint.
        resume
            Start from the last checkpoint after a pause.
        abort
            Move from 'paused' to 'idle', stopping the run permanently.
        panic
            Force the Run Engine to stop and/or disallow resume.
        all_is_well
            Un-panic
        register_command
            Teach the Run Engine a new Message command.
        unregister_command
            Undo register_command.
        """
        super(RunEngine, self).__init__()
        if memory is None:
            memory = {}
        self.memory = memory
        self._extra_fields = set(['project', 'group', 'sample'])
        self._panic = False
        self._bundling = False
        self._sigint_handler = None
        self._sigtstp_handler = None
        self._objs_read = deque()  # objects read in one Event
        self._read_cache = deque()  # cache of obj.read() in one Event
        self._describe_cache = dict()  # cache of all obj.describe() output
        self._descriptor_uids = dict()  # cache of all Descriptor uids
        self._sequence_counters = dict()  # a seq_num counter per Descriptor
        self._pause_requests = dict()  # holding {<name>: callable}
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
        self._queues = {name: LossyLiFoQueue(maxsize=2)
                        for name in queue_names}

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

    def _clear(self):
        self._bundling = False
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
        """
        Register a new Message command.

        Parameters
        ----------
        name : str
        func : callable
            This can be a function or a method. The signature is `f(msg)`.
        """
        self._command_registry[name] = func

    def unregister_command(self, name):
        """
        Unregister a Message command.

        Parameters
        ----------
        name : str
        """
        del self._command_registry[name]

    @property
    def persistent_fields(self):
        return list(set(self._extra_fields) | set(self._REQUIRED_FIELDS))

    @persistent_fields.setter
    def persistent_fields(self, val):
        self._extra_fields = val

    def panic(self):
        """
        Do not permit the RunEngine to run until all_is_well() is called.

        If the RunEngine is currently running, it will abort before processing
        any more Messages and enter the 'idle' state.

        If the RunEngine is currently paused, it will stay in the 'paused'
        state, and it will disallow resume() until all_is_well() is called.
        """
        # Release GIL by sleeping, allowing other threads to set panic.
        ttime.sleep(0.01)
        self._panic = True

    def all_is_well(self):
        """
        Un-panic.

        If the panic occurred during a pause, the run can be resumed.
        """
        self._panic = False

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
        # No matter what, this will be processed if we try to resume.
        if callback is not None:
            if name is None:
                raise ValueError("Pause requests with a callback must include "
                                 "a name.")
            self._pause_requests[name] = callback
        # Now to the right pause state if we can.
        if hard and self.state.can_hard_pause:
            self.state.hard_pause()
        else:
            if self.state.can_soft_pause:
                self.state.soft_pause()

    def _register_scan_callback(self, name, func):
        """Register a callback to be processed by the scan thread.

        Functions registered here are guaranteed to be run (there is no Queue
        involved) and they block the scan's progress until they return.
        """
        return self._scan_cb_registry.connect(name, func)

    def _push_to_queue(self, name, doc):
        self._queues[name].put(doc)

    def __call__(self, gen, subs=None, use_threading=True, **metadata):
        """Run the scan defined by ``gen``

        Any keyword arguments other than those listed below will be
        interpreted as metadata and recorded with the run.

        Parameters
        ----------
        gen : generator
            a generator or that yields ``Msg`` objects (or an iterable that
            returns such a generator)
        subs: dict, optional
            Temporary subscriptions (a.k.a. callbacks) to be used on this run.
            - Valid dict keys are: {'start', 'stop', 'event', 'descriptor'}
            - Dict values must be a function with the signature `f(dict)` or a
              list of such functions
        use_threading : bool, optional
            True by default. False makes debugging easier, but removes some
            features like pause/resume and main-thread subscriptions.

        Examples
        --------
        # Simplest example:
        >>> RE(my_scan)
        # Examples using subscriptions (a.k.a. callbacks):
        >>> def print_data(doc):
        ...     print("Measured: %s" % doc['data'])
        ...
        >>> def celebrate(doc):
        ...     # Do nothing with the input.
        ...     print("The run is finished!!!")
        ...
        >>> RE(my_generator, subs={'event': print_data, 'stop': celebrate})
        """
        if subs is None:
            subs = {}
        self._clear()
        for name, funcs in subs.items():
            if not isinstance(funcs, Iterable):
                # Take funcs to be a single function.
                funcs = [funcs]
            for func in funcs:
                if not callable(func):
                    raise ValueError("subs values must be functions or lists "
                                     "of functions. The offending entry is\n "
                                     "{0}".format(func))
                self._temp_callback_ids.add(self.subscribe(name, func))
        self._run_start_uid = new_uid()
        if self._panic:
            raise PanicError("RunEngine is panicked. The run "
                             "was aborted before it began. No records "
                             "of this run were created.")

        # Advance and stash scan_id
        try:
            scan_id = self.memory['scan_id'] + 1
        except KeyError:
            scan_id = 1
        self.memory['scan_id'] = scan_id
        metadata['scan_id'] = scan_id

        for field in self.persistent_fields:
            if field in metadata:
                # Stash and use new value.
                self.memory[field] = metadata[field]
            else:
                # Use old value.
                try:
                    metadata[field] = self.memory[field]
                except KeyError:
                    if field not in self._REQUIRED_FIELDS:
                        continue
                    raise KeyError("There is no entry for '{0}'. "
                                   "It is required for this run. In future "
                                   "runs, the most recent entry can be reused "
                                   "unless a new value is specified.".format(
                                        field))

        self.state.run()
        with SignalHandler(signal.SIGINT) as self._sigint_handler:  # ^C
            def func():
                return self._run(gen, metadata)
            if use_threading:
                self._thread = threading.Thread(target=func,
                                                name='scan_thread')
                self._thread.start()
                while self._thread.is_alive() and not self.state.is_paused:
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
        if self._panic:
            raise PanicError("Run Engine is panicked. If are you sure all is "
                             "well, call the all_is_well() method.")
        outstanding_requests = []
        # We listify so we can modify the dict inside the loop.
        for name, func in list(self._pause_requests.items()):
            if func():
                # We have permission to continue. Clear the request.
                del self._pause_requests[name]
            else:
                outstanding_requests.append(name)
        if outstanding_requests:
            return outstanding_requests
        self.state.resume()
        self._resume()
        return None

    def _resume(self):
        # This could be as a result of self.resume() or self.abort().
        with SignalHandler(signal.SIGINT) as self._sigint_handler:  # ^C
            while self._thread.is_alive() and not self.state.is_paused:
                self.dispatcher.process_all_queues()
        self.dispatcher.process_all_queues()  # catch any stragglers

    def abort(self):
        """
        Abort a paused scan.
        """
        self.state.abort()
        self._resume()  # to create RunStop Document

    def _run(self, gen, metadata):
        gen = iter(gen)  # no-op on generators; needed for classes
        doc = dict(uid=self._run_start_uid, time=ttime.time(), **metadata)
        self.debug("*** Emitted RunStart:\n%s" % doc)
        self.emit('start', doc)
        response = None
        exit_status = None
        reason = ''
        try:
            self._check_for_trouble()
            while True:
                # Send last response; get new message but don't process it yet.
                msg = gen.send(response)
                if self._msg_cache is not None:
                    # We have a checkpoint.
                    self._msg_cache.append(msg)
                self._check_for_trouble()
                # There is no trouble. Now process the message.
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
            if self.state.is_aborting or self.state.is_running:
                self.state.stop()
            elif self.state.is_soft_pausing or self.state.is_hard_pausing:
                # Apparently an exception was raised mid-pause.
                self.debug("The RunEngine encountered an error while "
                           "attempting to pause. Aborting and going to idle.")
                self.state.pause()
                self.state.abort()
                self.state.stop()

    def _check_for_trouble(self):
        # Check for panic.
        if self._panic:
            exit_status = 'fail'
            raise PanicError("Something told the Run Engine to "
                                "panic after the run began. "
                                "Records were created, but the run "
                                "was marked with "
                                "exit_status='fail'.")

        # Check for pause requests from keyboard.
        if self._sigint_handler.interrupted:
            self.debug("RunEngine detected a SIGINT (Ctrl+C)")
            self.request_pause(hard=True, name='SIGINT')
            self._sigint_handler.interrupted = False

        # If a hard pause was requested, sleep.
        resumable = self._msg_cache is not None
        if self.state.is_hard_pausing:
            self.state.pause()
            if not resumable:
                self.state.abort()
                exit_status = 'abort'
                raise RunInterrupt("*** Hard pause requested. There "
                                    "are no checkpoints. Cannot resume;"
                                    " must abort. Run aborted.")
            self.debug("*** Hard pause requested. Sleeping until "
                        "resume() is called. "
                        "Will rerun from last 'checkpoint' command.")
            while True:
                ttime.sleep(0.5)
                if not self.state.is_paused:
                    break
            if self.state.is_aborting:
                exit_status = 'abort'
                raise RunInterrupt("Run aborted.")
            self._rerun_from_checkpoint()

        # If a soft pause was requested, acknowledge it, but wait
        # for a 'checkpoint' command to catch it (see self._checkpoint).
        if self.state.is_soft_pausing:
            if not resumable:
                self.state.pause()
                self.state.abort()
                exit_status = 'abort'
                raise RunInterrupt("*** Soft pause requested. There "
                                    "are no checkpoints. Cannot resume;"
                                    " must abort. Run aborted.")
            self.debug("*** Soft pause requested. Continuing to "
                        "process messages until the next 'checkpoint' "
                        "command.")

    def _create(self, msg):
        self._read_cache.clear()
        self._objs_read.clear()
        self._bundling = True

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
        self._bundling = False

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
        if self._bundling:
            raise IllegalMessageSequence("Cannot 'checkpoint' after 'create' "
                                         "and before 'save'. Aborting!")
        self._msg_cache = deque()
        if self.state.is_soft_pausing:
            self.state.pause()  # soft_pausing -> paused
            self.debug("*** Checkpoint reached. Sleeping until resume() is "
                       "called. Will resume from checkpoint.")
            while True:
                ttime.sleep(0.5)
                if not self.state.is_paused:
                    break

    def _rerun_from_checkpoint(self):
        self.debug("*** Rerunning from checkpoint...")
        for msg in self._msg_cache:
            response = self._command_registry[msg.command](msg)
            self.debug('{}\n   ret: {} (On rerun, responses are not sent.)'
                       ''.format(msg, response))

    def emit(self, name, doc):
        "Process blocking, scan-thread callbacks."
        self._scan_cb_registry.process(name, doc)

    def debug(self, msg):
        "Print if the verbose attribute is True."
        if self.verbose:
            print(msg)


class Dispatcher:
    """Dispatch documents to user-defined consumers on the main thread."""

    def __init__(self, queues):
        self.queues = queues
        self.cb_registry = CallbackRegistry()

    def process_queue(self, name):
        """
        Process the last item in the queue.
        """
        queue = self.queues[name]
        try:
            document = queue.get_nowait()
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


class PanicError(Exception):
    pass


class RunInterrupt(KeyboardInterrupt):
    pass


class IllegalMessageSequence(Exception):
    pass
