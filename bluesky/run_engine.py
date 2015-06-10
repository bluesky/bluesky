import os
import time as ttime
import sys
from itertools import count
from collections import namedtuple, deque, defaultdict, Iterable
import uuid
import signal
import threading
from queue import Queue, Empty
from enum import Enum
import traceback

import json
import jsonschema
from super_state_machine.machines import StateMachine
from super_state_machine.extras import PropertyMachine
import numpy as np
from pkg_resources import resource_filename as rs_fn

from .utils import CallbackRegistry, SignalHandler, ExtendedList


__all__ = ['Msg', 'RunEngineStateMachine', 'RunEngine', 'Dispatcher',
           'RunInterrupt', 'PanicError', 'IllegalMessageSequence']

SCHEMA_PATH = 'schema'
SCHEMA_NAMES = {'start': 'run_start.json',
                'stop': 'run_stop.json',
                'event': 'event.json',
                'descriptor': 'event_descriptor.json'}
fn = '{}/{{}}'.format(SCHEMA_PATH)
schemas = {}
for name, filename in SCHEMA_NAMES.items():
    with open(rs_fn('bluesky', fn.format(filename))) as fin:
        schemas['name'] = json.load(fin)


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
            {'idle', 'running', 'soft_pausing, 'hard_pausing', 'paused',
             'aborting'}
        md
            direct access to the dict-like persistent storage described above
        persistent_fields
            list of metadata fields that will be remembered and reused between
            subsequence runs
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
        super(RunEngine, self).__init__()
        if md is None:
            md = {}
        self.md = md
        self.logbook = logbook
        self.persistent_fields = ExtendedList(self._REQUIRED_FIELDS)
        self.persistent_fields.extend(['project', 'group', 'sample'])
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
        self._exit_status = None  # {'success', 'fail', 'abort'}
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
            'logbook': self._logbook
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

        self.verbose = False

    def _clear(self):
        self._bundling = False
        self._objs_read.clear()
        self._read_cache.clear()
        self._describe_cache.clear()
        self._descriptor_uids.clear()
        self._sequence_counters.clear()
        self._msg_cache = None
        self._exit_status = None
        # clear the main thread queues
        for queue in self._queues.values():
            try:
                while True:
                    self.debug(
                        "This was left on the queue after the last run ended:"
                        "====\n{}\n====".format(queue.get_nowait()))
            except Empty:
                # queue is empty
                pass

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
            scan_id = self.md['scan_id'] + 1
        except KeyError:
            scan_id = 1
        self.md['scan_id'] = scan_id
        metadata['scan_id'] = scan_id

        for field in self.persistent_fields:
            if field in metadata:
                # Stash and use new value.
                self.md[field] = metadata[field]
            else:
                # Use old value.
                try:
                    metadata[field] = self.md[field]
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
        print("Aborting....")
        self._resume()  # to create RunStop Document

    def _run(self, gen, metadata):
        gen = iter(gen)  # no-op on generators; needed for classes
        doc = dict(uid=self._run_start_uid, time=ttime.time(), **metadata)
        self.debug("*** Emitted RunStart:\n%s" % doc)
        self.emit('start', doc)
        response = None
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
                self.debug('msg: {}\n   response: {}'.format(msg, response))
        except StopIteration:
            self._exit_status = 'success'
        except Exception as err:
            self._exit_status = 'fail'
            reason = str(err)
            raise err
        finally:
            doc = dict(run_start=self._run_start_uid,
                       time=ttime.time(), uid=new_uid(),
                       exit_status=self._exit_status,
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
            self._exit_status = 'fail'
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
            print("Pausing...")
            if not resumable:
                self.state.abort()
                print("No checkpoint; cannot pause. Aborting...")
                self._exit_status = 'abort'
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
                self._exit_status = 'abort'
                raise RunInterrupt("Run aborted.")
            self._rerun_from_checkpoint()

        # If a soft pause was requested, acknowledge it, but wait
        # for a 'checkpoint' command to catch it (see self._checkpoint).
        if self.state.is_soft_pausing:
            if not resumable:
                self.state.pause()
                print("No checkpoint; cannot pause. Aborting...")
                self.state.abort()
                self._exit_status = 'abort'
                raise RunInterrupt("*** Soft pause requested. There "
                                   "are no checkpoints. Cannot resume;"
                                   " must abort. Run aborted.")
            print("Soft pause acknowledged. Continuing to next checkpoint.")
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
            _fill_missing_fields(data_keys)  # TODO Move this to ophyd/controls
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
        block_group = msg.kwargs.pop('block_group', None)
        if block_group:
            self._block_groups[block_group].add(msg.obj)

        return msg.obj.kickoff(*msg.args, **msg.kwargs)

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
                self.emit('descriptor', doc)
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
            self.emit('event', ev)
            self.debug("Emitted Event:\n%s" % ev)

    def _null(self, msg):
        pass

    def _set(self, msg):
        block_group = msg.kwargs.pop('block_group', None)
        if block_group:
            self._block_groups[block_group].add(msg.obj)
        return msg.obj.set(*msg.args, **msg.kwargs)

    def _trigger(self, msg):
        block_group = msg.kwargs.pop('block_group', None)
        if block_group:
            self._block_groups[block_group].add(msg.obj)

        return msg.obj.trigger(*msg.args, **msg.kwargs)

    def _wait(self, msg):
        # Block progress until every object that was trigged
        # triggered with the keyword argument `block=group` is done.
        group = msg.kwargs.get('group', msg.args[0])
        objs = self._block_groups[group]
        while True:
            if all([obj.ready for obj in objs]):
                break
            ttime.sleep(1e-4)
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
            print("Checkpoint reached. Pausing...")
            self.debug("*** Checkpoint reached. Sleeping until resume() is "
                       "called. Will resume from checkpoint.")
            while True:
                ttime.sleep(0.5)
                if not self.state.is_paused:
                    break
            if self.state.is_aborting:
                self._exit_status = 'abort'
                raise RunInterrupt("Run aborted.")

    def _rerun_from_checkpoint(self):
        print("Rerunning from last checkpoint...")
        self.debug("*** Rerunning from checkpoint...")
        for msg in self._msg_cache:
            response = self._command_registry[msg.command](msg)
            self.debug('{}\n   ret: {} (On rerun, responses are not sent.)'
                       ''.format(msg, response))

    def _logbook(self, msg):
        if self.logbook:
            d = msg.kwargs
            log_message, = msg.args
            d['uid'] = self._run_start_uid

            self.logbook(log_message, d)

    def emit(self, name, doc):
        "Process blocking, scan-thread callbacks."
        jsonschema.validate(doc, schemas[name])
        self._scan_cb_registry.process(name, doc)

    def debug(self, msg):
        "Print if the verbose attribute is True."
        if self.verbose:
            print(msg)


class Dispatcher:
    """Dispatch documents to user-defined consumers on the main thread."""

    def __init__(self, queues):
        self.queues = queues
        self.cb_registry = CallbackRegistry(halt_on_exception=False)
        self._counter = count()
        self._token_mapping = dict()

    def process_queue(self, name):
        """
        Process the last item in the queue.
        """
        queue = self.queues[name]
        try:
            document = queue.get_nowait()
        except Empty:
            # no documents available on the queue
            pass
        else:
            # process the callbacks for "name" and grab any exceptions that
            # come out the registry as "ninety_nine_probs"
            ninety_nine_probs = self.cb_registry.process(name, document)
            # spam the screen with the exceptions that are being raised
            if ninety_nine_probs:
                print("The following exceptions were raised during processing "
                      "of the [[{}]] queue".format(name))
                for idx, (error, tb) in enumerate(ninety_nine_probs):
                    err = "Error %s" % idx
                    print("\n%s\n%s\n%s\n" % (err, '-'*len(err), error))
                    print("\tTraceback\n"
                          "\t---------")
                    # todo format this traceback properly
                    tb_list = traceback.extract_tb(tb)
                    for item in traceback._format_list_iter(tb_list):
                        print('\t%s' % item)

    def process_all_queues(self):
        # keys are hard-coded to enforce order
        for name in ['start', 'descriptor', 'event', 'stop']:
            self.process_queue(name)

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
        queue_keys = self.queues.keys()
        if name in queue_keys:
            private_token = self.cb_registry.connect(name, func)
            public_token = next(self._counter)
            self._token_mapping[public_token] = [private_token]
        elif name == 'all':
            private_tokens = []
            for key in queue_keys:
                private_tokens.append(self.cb_registry.connect(key, func))
            public_token = next(self._counter)
            self._token_mapping[public_token] = private_tokens
        else:
            valid_names = queue_keys + ['all']
            raise ValueError("Valid names: {0}".format(valid_names))
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


class PanicError(Exception):
    pass


class RunInterrupt(KeyboardInterrupt):
    pass


class IllegalMessageSequence(Exception):
    pass
