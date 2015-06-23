import os
import asyncio
import itertools
import types
import time as ttime
import sys
from itertools import count
from collections import namedtuple, deque, defaultdict, Iterable
import uuid
import signal
from enum import Enum


import json
import jsonschema
from super_state_machine.machines import StateMachine
from super_state_machine.extras import PropertyMachine
import numpy as np
from pkg_resources import resource_filename as rs_fn

from .utils import CallbackRegistry, SignalHandler, ExtendedList


__all__ = ['Msg', 'RunEngineStateMachine', 'RunEngine', 'Dispatcher',
           'RunInterrupt', 'PanicError', 'IllegalMessageSequence']


def expiring_function(func, *args, **kwargs):
    """
    If timeout has not occurred, call func(*args, **kwargs).

    This is meant to used with the event loop's run_in_exector
    method. Outside that context, it doesn't make any sense.
    """
    def dummy(start_time, timeout):
        if loop.time() > start_time + timeout:
            return
        func(*args, **kwargs)
        return

    return dummy


class DocumentNames(Enum):
    stop = 'stop'
    start = 'start'
    descriptor = 'descriptor'
    event = 'event'

SCHEMA_PATH = 'schema'
SCHEMA_NAMES = {DocumentNames.start: 'run_start.json',
                DocumentNames.stop: 'run_stop.json',
                DocumentNames.event: 'event.json',
                DocumentNames.descriptor: 'event_descriptor.json'}
fn = '{}/{{}}'.format(SCHEMA_PATH)
schemas = {}
for name, filename in SCHEMA_NAMES.items():
    with open(rs_fn('bluesky', fn.format(filename))) as fin:
        schemas[name] = json.load(fin)


loop = asyncio.get_event_loop()
loop.set_debug(True)


class Msg(namedtuple('Msg_base', ['command', 'obj', 'args', 'kwargs'])):
    __slots__ = ()

    def __new__(cls, command, obj=None, *args, **kwargs):
        return super(Msg, cls).__new__(cls, command, obj, args, kwargs)

    def __repr__(self):
        return '{}: ({}), {}, {}'.format(
            self.command, self.obj, self.args, self.kwargs)


class RunEngineStateMachine(StateMachine):
    """

    Attributes
    ----------
    is_idle
        State machine is in its idle state
    is_running
        State machine is in its running state
    is_paused
        State machine is paused.
    """

    class States(Enum):
        """state.name = state.value"""
        IDLE = 'idle'
        RUNNING = 'running'
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
            'running': ['idle', 'paused'],
            'paused': ['idle', 'running'],
        }
        named_checkers = [
            ('can_pause', 'paused'),
        ]


class RunEngine:

    _loop = loop  # just a convenient way to inspect the global event loop
    state = PropertyMachine(RunEngineStateMachine)
    _REQUIRED_FIELDS = ['beamline_id', 'owner', 'group', 'config']

    def __init__(self, md=None, logbook=None):
        """
        The Run Engine execute messages and emits Documents.

        Parameters
        ----------
        md : dict-like, optional
            The default is a standard Python dictionary, but fancier objects
            can be used to store long-term history and persist it between
            sessions. The standard configuration instantiates a Run Engine with
            history.History, a simple interface to a sqlite file. Any object
            supporting `__getitem__`, `__setitem__`, and `clear` will work.

        logbook : callable, optional
            logbook(msg, properties=dict)


        Attributes
        ----------
        state
            {'idle', 'running', 'paused'}
        md
            direct access to the dict-like persistent storage described above
        persistent_fields
            list of metadata fields that will be remembered and reused between
            subsequence runs
        event_timeout
            number of seconds before Events yet unprocessed by callbacks are
            skipped
        logbook
            callable accepting a message and an optional dict

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
        super().__init__()
        if md is None:
            md = {}
        self.md = md
        self.logbook = logbook
        self._metadata_per_call = {}  # for all runs generated by one __call__
        self._metadata_per_run = {}  # for one run, incorporating Msg metadata
        self.persistent_fields = ExtendedList(self._REQUIRED_FIELDS)
        self.persistent_fields.extend(['project', 'group', 'sample'])
        self._panic = False
        self._bundling = False  # if we are in the middle of bundling readings
        self._run_is_open = False  # if we have emitted a RunStart, no RunStop
        self._deferred_pause_requested = False  # pause at next 'checkpoint'
        self._sigint_handler = None
        self._sigtstp_handler = None
        self._exception = None  # stored and then raised in the _run loop
        self._objs_read = deque()  # objects read in one Event
        self._read_cache = deque()  # cache of obj.read() in one Event
        self._configured = list()  # objects configured, not yet deconfigured
        self._run_start_uids = list()  # run start uids generated by __call__
        self._describe_cache = dict()  # cache of all obj.describe() output
        self._descriptor_uids = dict()  # cache of all Descriptor uids
        self._sequence_counters = dict()  # a seq_num counter per Descriptor
        self._pause_requests = dict()  # holding {<name>: callable}
        self._block_groups = defaultdict(set)  # sets of objs to wait for
        self._temp_callback_ids = set()  # ids from CallbackRegistry
        self._msg_cache = None  # may be used to hold recently processed msgs
        self._genstack = deque()  # stack of generators to work off of

        self._exit_status = 'fail'  # pessimistic default
        self._reason = ''
        self._task = None
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
            'kickoff': self._kickoff,
            'logbook': self._logbook,
            'configure': self._configure,
            'deconfigure': self._deconfigure,
            'subscribe': self._subscribe,
            'open_run': self._open_run,
            'close_run': self._close_run,
            'wait_for': self._wait_for,
        }

        # public dispatcher for callbacks processed on the main thread
        self.dispatcher = Dispatcher()
        self.event_timeout = 0.1
        self.subscribe = self.dispatcher.subscribe
        self.unsubscribe = self.dispatcher.unsubscribe

        # private registry of blocking callbacks
        self._scan_cb_registry = CallbackRegistry(allowed_sigs=DocumentNames)

        self.verbose = False

        loop.call_soon(self._check_for_trouble)
        loop.call_soon(self._check_for_signals)

    def _clear_run_cache(self):
        self._metadata_per_run.clear()
        self._bundling = False
        self._run_is_open = False
        self._objs_read.clear()
        self._read_cache.clear()
        self._configured.clear()
        self._describe_cache.clear()
        self._descriptor_uids.clear()
        self._sequence_counters.clear()
        self._block_groups.clear()

    def _clear_call_cache(self):
        self._metadata_per_call.clear()
        self._deferred_pause_requested = False
        self._msg_cache = None
        self._genstack = deque()
        self._exception = None
        self._run_start_uids.clear()
        self._exit_status = 'fail'
        self._reason = ''
        self._task = None

        # Unsubscribe for per-run callbacks.
        for cid in self._temp_callback_ids:
            self.unsubscribe(cid)
        self._temp_callback_ids.clear()

    def reset(self):
        self._clear_run_cache()
        self._clear_call_cache()
        self.dispatcher.unsubscribe_all()

    @property
    def resumable(self):
        return self._msg_cache is not None

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

    def panic(self):
        """
        Do not permit the RunEngine to run until all_is_well() is called.

        If the RunEngine is currently running, it will abort before processing
        any more Messages and enter the 'idle' state.

        If the RunEngine is currently paused, it will stay in the 'paused'
        state, and it will disallow resume() until all_is_well() is called.
        """
        self._panic = True
        self._task.cancel()

    def all_is_well(self):
        """
        Un-panic.

        If the panic occurred during a pause, the run can be resumed.
        """
        self._panic = False
        # The cycle where _check_for_trouble schedules a future call to itself
        # is broken when it raises a PanicError.
        loop.call_later(0.1, self._check_for_trouble)

    def request_pause(self, defer=False, name=None, callback=None):
        """
        Command the Run Engine to pause.

        This function is called by 'pause' Messages. It can also be called
        by other threads. It cannot be called on the main thread during a run,
        but it is called by SIGINT (i.e., Ctrl+C).

        If there current run has no checkpoint commands, this will cause the
        run to abort.

        Parameters
        ----------
        defer : bool, optional
            If False, pause immediately before processing any new messages.
            If True, pause at the next checkpoint.
            False by default.
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
        if not defer:
            if self.state.can_pause:
                print("Pausing...")
                self.state = 'paused'
                if self.resumable:
                    loop.stop()
                else:
                    print("No checkpoint; cannot pause. Aborting...")
                    self._exception = FailedPause()
                    self._task.cancel()
            else:
                print("Cannot pause from {0} state. "
                      "Ignoring request.".format(self.state))
        else:
            if self.state.is_running:
                self._deferred_pause_requested = True
                print("Deferred pause acknowledged. Continuing to checkpoint.")
            else:
                print("Cannot pause from {0} state. "
                      "Ignoring request.".format(self.state))

    def _register_scan_callback(self, name, func):
        """Register a callback to be processed by the scan thread.

        Functions registered here are guaranteed to be run (there is no Queue
        involved) and they block the scan's progress until they return.
        """
        return self._scan_cb_registry.connect(name, func)

    def __call__(self, plan, subs=None, **metadata):
        """Run the scan defined by ``plan``

        Any keyword arguments other than those listed below will be
        interpreted as metadata and recorded with the run.

        Parameters
        ----------
        plan : generator
            a generator or that yields ``Msg`` objects (or an iterable that
            returns such a generator)
        subs: dict, optional
            Temporary subscriptions (a.k.a. callbacks) to be used on this run.
            - Valid dict keys are: {'start', 'stop', 'event', 'descriptor'}
            - Dict values must be a function with the signature `f(dict)` or a
              list of such functions

        Returns
        -------
        uids : list
            list of Header uids (a.k.a RunStart uids) of run(s)

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
        # First thing's first: if we are in the wrong state, raise.
        if not self.state.is_idle:
            raise RuntimeError("The RunEngine is in a %s state" % self.state)
        if self._panic:
            raise PanicError("RunEngine is panicked. The run "
                             "was aborted before it began. No records "
                             "of this run were created.")

        # Register temporary subscriptions. Save tokens to unsubscribe later.
        if subs is None:
            subs = {}
        try:
            scan_subs = plan.subs
        except AttributeError:
            scan_subs = {}
        self._clear_call_cache()
        for name, funcs in itertools.chain(subs.items(), scan_subs.items()):
            if not isinstance(funcs, Iterable):
                # Take funcs to be a single function.
                funcs = [funcs]
            for func in funcs:
                if not callable(func):
                    raise ValueError("subs values must be functions or lists "
                                     "of functions. The offending entry is\n "
                                     "{0}".format(func))
                self._temp_callback_ids.add(self.subscribe(name, func))

        self._metadata_per_call = metadata

        self.state = 'running'
        gen = iter(plan)  # no-op on generators; needed for classes
        if not isinstance(gen, types.GeneratorType):
            # If plan does not support .send, we must wrap it in a generator.
            gen = (msg for msg in gen)
        self._genstack.append(gen)
        with SignalHandler(signal.SIGINT) as self._sigint_handler:  # ^C
            self._task = loop.create_task(self._run())
            loop.run_forever()
            if self._task.done() and not self._task.cancelled():
                exc = self._task.exception()
                if exc is not None:
                    raise exc

        return self._run_start_uids

    def resume(self):
        """Resume a run from the last checkpoint.

        Returns
        -------
        requests_or_uids : list
            If any pause requests have not been released, a list of their names
            is immediately returned. Otherwise, after the run completes, this
            returns the list of uids this plan knows about.
        """
        # The state machine does not capture the whole picture.
        if not self.state.is_paused:
            raise RuntimeError("The RunEngine is the {0} state. You can only "
                               "resume for the paused state."
                               "".format(self.state))
        if self._panic:
            raise PanicError("Run Engine is panicked. If are you sure all is "
                             "well, call the all_is_well() method.")

        # Check that all pause requests have been released.
        outstanding_requests = []
        for name, func in list(self._pause_requests.items()):
            if func():
                # We have permission to continue. Clear the request.
                del self._pause_requests[name]
            else:
                outstanding_requests.append(name)
        if outstanding_requests:
            return outstanding_requests

        self._genstack.append((msg for msg in list(self._msg_cache)))
        self._msg_cache = deque()
        self._resume_event_loop()
        return self._run_start_uids

    def _resume_event_loop(self):
        # may be called by 'resume' or 'abort'
        self.state = 'running'
        with SignalHandler(signal.SIGINT) as self._sigint_handler:  # ^C
            if self._task.done():
                return
            loop.run_forever()
            if self._task.done() and not self._task.cancelled():
                exc = self._task.exception()
                if exc is not None:
                    raise exc

    def request_suspend(self, fut):
        """
        Request that the run suspend it self until
        """
        if not self.resumable:
            print("No checkpoint; cannot suspend. Aborting...")
            self._exception = FailedPause()
        else:
            print("Suspending....To get prompt hit Ctrl-C to pause the scan")
            wait_msg = Msg('wait_for', [fut, ])
            new_msg_lst = [wait_msg, ] + list(self._msg_cache)
            self._msg_cache = deque()
            self._genstack.append((msg for msg in new_msg_lst))

    def abort(self, reason=''):
        """
        Stop a paused scan and mark it as aborted.
        """
        # The state machine does not capture the whole picture.
        if not self.state.is_paused:
            raise RuntimeError("The RunEngine is the {0} state. You can only "
                               "resume for the paused state.".format(
                                   self.state))
        print("Aborting....")
        self._reason = reason
        self._exception = RequestAbort()
        self._task.cancel()
        self._resume_event_loop()

    def stop(self):
        print("Stopping...")
        self._exception = RequestStop()
        self._resume_event_loop()

    @asyncio.coroutine
    def _run(self):
        response = None
        self._reason = ''
        try:
            while True:
                if self._exception is not None:
                    raise self._exception
                # Send last response; get new message but don't process it yet.
                try:
                    try:
                        msg = self._genstack[-1].send(response)
                    except TypeError:
                        # Deal with new generators that need to
                        # be primed
                        msg = self._genstack[-1].send(None)
                except StopIteration:
                    self._genstack.pop()
                    if len(self._genstack):
                        continue
                    else:
                        raise
                if self._msg_cache is not None:
                    # We have a checkpoint.
                    self._msg_cache.append(msg)
                # There is no trouble. Now process the message.
                coro = self._command_registry[msg.command]
                self.debug("About to process: {0}, {1}".format(coro, msg))
                yield from asyncio.sleep(0.001)  # TODO Do we need this?
                response = yield from coro(msg)
                self.debug('RE.state: ' + self.state)
                self.debug('msg: {}\n   response: {}'.format(msg, response))
        except (StopIteration, RequestStop):
            self._exit_status = 'success'
            yield from asyncio.sleep(0.001)  # TODO Do we need this?
        except (FailedPause, RequestAbort, asyncio.CancelledError):
            self._exit_status = 'abort'
            yield from asyncio.sleep(0.001)  # TODO Do we need this?
            if isinstance(self._exception, PanicError):
                self._exit_status = 'fail'
                raise self._exception
        except Exception as err:
            self._exit_status = 'fail'  # Exception raises during 'running'
            self._reason = str(err)
            raise err
        finally:
            self.state = 'idle'
            # in case we were interrupted between 'configure' and 'deconfigure'
            for obj in self._configured:
                obj.deconfigure()
                self._configured.remove(obj)
            sys.stdout.flush()
            # Emit RunStop if necessary.
            if self._run_is_open:
                yield from self._close_run(Msg('close_run'))
                self._run_is_open = False

            for task in asyncio.Task.all_tasks(loop):
                task.cancel()
            loop.stop()

    def _check_for_trouble(self):
        if self.state.is_running:
            # Check for panic.
            if self._panic:
                self._exit_status = 'fail'
                exc = PanicError("Something told the Run Engine to "
                                 "panic after the run began. "
                                 "Records were created, but the run "
                                 "was marked with "
                                 "exit_status='fail'.")
                self._exception = exc  # will stop _run coroutine

        loop.call_later(0.1, self._check_for_trouble)

    def _check_for_signals(self):
        # Check for pause requests from keyboard.
        if self.state.is_running:
            if self._sigint_handler.interrupted:
                self.debug("RunEngine detected a SIGINT (Ctrl+C)")
                loop.call_soon(self.request_pause, False, 'SIGINT')
                self._sigint_handler.interrupted = False

        loop.call_later(0.1, self._check_for_signals)

    def increment_scan_id(self):
        scan_id = self.md.get('scan_id', 0) + 1
        self.md['scan_id'] = scan_id
        return scan_id

    @asyncio.coroutine
    def _wait_for(self, msg):
        futs = msg.obj
        yield from asyncio.wait(futs)

    @asyncio.coroutine
    def _open_run(self, msg):
        if self._run_is_open:
            raise IllegalMessageSequence("A 'close_run' messgae was not "
                                         "received before the 'open_run' "
                                         "message")
        self._clear_run_cache()
        self._run_is_open = True
        self._run_start_uid = new_uid()
        self._run_start_uids.append(self._run_start_uid)

        # Metadata can come from history, __call__, or the open_run Msg.
        self._metadata_per_run = {k: v for k, v in self.md.items()
                                  if k in self.persistent_fields}
        self._metadata_per_run.update(self._metadata_per_call)
        self._metadata_per_run.update(msg.kwargs)

        for field in self._REQUIRED_FIELDS:
            if field not in self._metadata_per_run:
                raise KeyError("The field '{0}' was not specified as is "
                               "required.".format(field))

        # If any persistent fields have been overridden, update md.
        for field in self.persistent_fields:
            if field not in self._metadata_per_run:
                continue
            is_updated = (field not in self.md or
                          self._metadata_per_run[field] != self.md[field])
            if is_updated:
                self.md[field] = self._metadata_per_run[field]

        # At this point required fields must be present.
        if 'scan_id' not in self._metadata_per_run:
            self._metadata_per_run['scan_id'] = self.increment_scan_id()
        doc = dict(uid=self._run_start_uid, time=ttime.time(),
                   **self._metadata_per_run)
        yield from self.emit(DocumentNames.start, doc)
        self.debug("*** Emitted RunStart:\n%s" % doc)

    @asyncio.coroutine
    def _close_run(self, msg):
        doc = dict(run_start=self._run_start_uid,
                   time=ttime.time(), uid=new_uid(),
                   exit_status=self._exit_status,
                   reason=self._reason)
        yield from self.emit(DocumentNames.stop, doc)
        self.debug("*** Emitted RunStop:\n%s" % doc)
        self._run_is_open = False

    @asyncio.coroutine
    def _create(self, msg):
        self._read_cache.clear()
        self._objs_read.clear()
        self._bundling = True

    @asyncio.coroutine
    def _read(self, msg):
        obj = msg.obj
        self._objs_read.append(obj)
        if obj not in self._describe_cache:
            self._describe_cache[obj] = obj.describe()
        ret = obj.read(*msg.args, **msg.kwargs)
        self._read_cache.append(ret)
        return ret

    @asyncio.coroutine
    def _save(self, msg):
        # The Event Descriptor is uniquely defined by the set of objects
        # read in this Event grouping.
        objs_read = frozenset(self._objs_read)

        # Event Descriptor
        if objs_read not in self._descriptor_uids:
            # We don't not have an Event Descriptor for this set.
            data_keys = {}
            [data_keys.update(self._describe_cache[obj]) for obj in objs_read]
            _fill_missing_fields(data_keys)  # TODO Move this to ophyd/controls
            descriptor_uid = new_uid()
            doc = dict(run_start=self._run_start_uid, time=ttime.time(),
                       data_keys=data_keys, uid=descriptor_uid)
            yield from self.emit(DocumentNames.descriptor, doc)
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
        yield from self.emit(DocumentNames.event, doc)
        self.debug("*** Emitted Event:\n%s" % doc)

    @asyncio.coroutine
    def _kickoff(self, msg):
        block_group = msg.kwargs.pop('block_group', None)
        ret = msg.obj.kickoff(*msg.args, **msg.kwargs)

        if block_group:
            p_event = asyncio.Event()

            def done_callback():
                loop.call_soon_threadsafe(p_event.set)

            ret.finished_cb = done_callback
            self._block_groups[block_group].add(p_event.wait())

        return ret

    @asyncio.coroutine
    def _collect(self, msg):
        obj = msg.obj
        data_keys_list = obj.describe()
        for data_keys in data_keys_list:
            objs_read = frozenset(data_keys)
            if objs_read not in self._descriptor_uids:
                # We don't not have an Event Descriptor for this set.
                descriptor_uid = new_uid()
                doc = dict(run_start=self._run_start_uid, time=ttime.time(),
                           data_keys=data_keys, uid=descriptor_uid)
                self.emit(DocumentNames.descriptor, doc)
                self.debug("Emitted Event Descriptor:\n%s" % doc)
                self._descriptor_uids[objs_read] = descriptor_uid
                self._sequence_counters[objs_read] = count(1)

        for ev in obj.collect():
            objs_read = frozenset(ev['data'])
            seq_num = next(self._sequence_counters[objs_read])
            descriptor_uid = self._descriptor_uids[objs_read]
            event_uid = new_uid()

            reading = ev['data']
            for key in ev['data']:
                reading[key] = _sanitize_np(reading[key])
            ev['data'] = reading
            ev['descriptor'] = descriptor_uid
            ev['seq_num'] = seq_num
            ev['uid'] = event_uid
            self.emit(DocumentNames.event, ev)
            self.debug("Emitted Event:\n%s" % ev)

    @asyncio.coroutine
    def _null(self, msg):
        pass

    @asyncio.coroutine
    def _set(self, msg):
        block_group = msg.kwargs.pop('block_group', None)
        ret = msg.obj.set(*msg.args, **msg.kwargs)
        if block_group:
            p_event = asyncio.Event()

            def done_callback():
                loop.call_soon_threadsafe(p_event.set)

            ret.finished_cb = done_callback
            self._block_groups[block_group].add(p_event.wait())

        return ret

    @asyncio.coroutine
    def _trigger(self, msg):
        block_group = msg.kwargs.pop('block_group', None)
        ret = msg.obj.trigger(*msg.args, **msg.kwargs)

        if block_group:
            p_event = asyncio.Event()

            def done_callback():
                loop.call_soon_threadsafe(p_event.set)

            ret.finished_cb = done_callback
            self._block_groups[block_group].add(p_event.wait())

        return ret

    @asyncio.coroutine
    def _wait(self, msg):
        # Block progress until every object that was trigged
        # triggered with the keyword argument `block=group` is done.
        group = msg.kwargs.get('group', msg.args[0])
        objs = list(self._block_groups.pop(group, []))
        if objs:
            yield from self._wait_for(Msg('wait_for', objs))

    @asyncio.coroutine
    def _sleep(self, msg):
        yield from asyncio.sleep(*msg.args)

    @asyncio.coroutine
    def _pause(self, msg):
        self.request_pause(*msg.args, **msg.kwargs)

    @asyncio.coroutine
    def _checkpoint(self, msg):
        if self._bundling:
            raise IllegalMessageSequence("Cannot 'checkpoint' after 'create' "
                                         "and before 'save'. Aborting!")
        self._msg_cache = deque()
        if self._deferred_pause_requested:
            self.state = 'paused'
            loop.stop()

    @asyncio.coroutine
    def _logbook(self, msg):
        if self.logbook:
            d = msg.kwargs
            input_message, = msg.args
            msg = []
            msg.append('Header uid: {uid}')
            msg.append('')
            msg.append('Scan Plan')
            msg.append('---------')
            msg.append(input_message)
            msg.append('')
            msg.append('Metadata')
            msg.append('--------')
            msg.append(_run_engine_log_template(self._metadata_per_run))
            log_message = '\n'.join(msg)

            d['uid'] = self._run_start_uid
            d.update(self.md)
            return self.logbook(log_message, d)

    @asyncio.coroutine
    def _configure(self, msg):
        # If an object has no 'configure' method, assume it does not need
        # configuring.
        _, obj, args, kwargs = msg
        if not hasattr(obj, 'configure'):
            return None
        result = obj.configure(*args, **kwargs)
        self._configured.append(obj)
        return result

    @asyncio.coroutine
    def _deconfigure(self, msg):
        # If an object has no 'deconfigure' method, assume it does not need
        # deconfiguring.
        _, obj, args, kwargs = msg
        if not hasattr(obj, 'deconfigure'):
            return None
        # Deconfigure is not allowed to have args or kwargs.
        # TODO Address this in Message validation.
        result = obj.deconfigure()
        self._configured.remove(obj)
        return result

    @asyncio.coroutine
    def _subscribe(self, msg):
        """
        Add a subscription after the run has started.

        This, like subscriptions passed to __call__, apply to one run only.
        """
        _, obj, args, kwargs = msg
        token = self.subscribe(*args, **kwargs)
        self._temp_callback_ids.add(token)
        return token

    @asyncio.coroutine
    def _process_event(self, doc):
        "Wrap a curried dispatcher method in a coroutine."
        return self.dispatcher.process('event', doc)

    @asyncio.coroutine
    def _impatient_event_processor(self, doc):
        "Schedule a Event to be processed with a timeout."
        yield from asyncio.wait_for(self._process_event,
                                    self._event_timeout)

    @asyncio.coroutine
    def emit(self, name, doc):
        "Process blocking callbacks and schedule non-blocking callbacks."
        jsonschema.validate(doc, schemas[name])
        self._scan_cb_registry.process(name, doc)
        if name != DocumentNames.event:
            loop.call_soon(self.dispatcher.process, name, doc)
        else:
            start_time = loop.time()
            dummy = expiring_function(self.dispatcher.process, name, doc)
            loop.run_in_executor(None, dummy, start_time, self.event_timeout)

    def debug(self, msg):
        "Print if the verbose attribute is True."
        if self.verbose:
            print(msg)


class Dispatcher:
    """Dispatch documents to user-defined consumers on the main thread."""

    def __init__(self):
        self.cb_registry = CallbackRegistry(allowed_sigs=DocumentNames,
                                            halt_on_exception=False)
        self._counter = count()
        self._token_mapping = dict()

    def process(self, name, doc):
        self.cb_registry.process(name, doc)

    def subscribe(self, name, func):
        """
        Register a function to consume Event documents.

        The Run Engine can execute callback functions at the start and end
        of a scan, and after the insertion of new Event Descriptors
        and Events.

        Parameters
        ----------
        name: {'start', 'descriptor', 'event', 'stop', 'all'}
        func: callable
            expecting signature like ``f(mongoengine.Document)``

        Returns
        -------
        token : int
            an integer token that can be used to unsubscribe
        """
        if name == 'all':
            private_tokens = []
            for key in DocumentNames:
                private_tokens.append(self.cb_registry.connect(key, func))
            public_token = next(self._counter)
            self._token_mapping[public_token] = private_tokens
            return public_token

        if name not in DocumentNames:
            name = DocumentNames[name]
        private_token = self.cb_registry.connect(name, func)
        public_token = next(self._counter)
        self._token_mapping[public_token] = [private_token]
        return public_token

    def unsubscribe(self, token):
        """
        Unregister a callback function using its integer ID.

        Parameters
        ----------
        token : int
            the integer token issued by `subscribe`
        """
        for private_token in self._token_mapping[token]:
            self.cb_registry.disconnect(private_token)

    def unsubscribe_all(self):
        """Unregister ALL callbacks from the dispatcher
        """
        for public_token in self._token_mapping.keys():
            self.unsubscribe(public_token)


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
        result[key]['shape'] = value.get('shape', None)
        if 'external' in value:
            result[key]['external'] = value['external']
    return result


def _run_engine_log_template(metadata):
    template = []
    for key in metadata:
        template.append("{key}: {{{key}}}".format(key=key))
    return '\n'.join(template)


class RequestAbort(Exception):
    pass


class RequestStop(Exception):
    pass


class PanicError(Exception):
    pass


class RunInterrupt(KeyboardInterrupt):
    pass


class IllegalMessageSequence(Exception):
    pass


class FailedPause(Exception):
    pass
