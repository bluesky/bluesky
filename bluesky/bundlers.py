from collections import deque
import inspect
import time as ttime
from typing import Any, Deque, Dict, FrozenSet, List, Tuple, Set, Optional, Union
from event_model import DocumentNames, compose_run, ComposeDescriptorBundle, pack_event_page, DataKey
from .log import doc_logger
from .protocols import (
    T, Callback, Configurable, PartialEvent, Flyable,
    HasName, Readable, Reading, Subscribable, check_supports,  EventCollectable, EventPageCollectable, Collectable
)
from .utils import (
    new_uid,
    IllegalMessageSequence,
    _rearrange_into_parallel_dicts,
    short_uid,
    Msg,
    maybe_await,
    maybe_collect_asset_docs,
    maybe_update_hints,
)

ObjDict = Dict[Any, Dict[str, T]]


class RunBundler:
    def __init__(self, md, record_interruptions, emit, emit_sync, log, *, strict_pre_declare=False):
        # if create can YOLO implicitly create a stream
        self._strict_pre_declare = strict_pre_declare
        # state stolen from the RE
        self.bundling = False  # if we are in the middle of bundling readings
        self._bundle_name = None  # name given to event descriptor
        self._run_start_uid = None  # The (future) runstart uid
        self._objs_read: Deque[HasName] = deque()  # objects read in one Event
        self._read_cache: Deque[Dict[str, Reading]] = deque()  # cache of obj.read() in one Event
        self._asset_docs_cache = deque()  # cache of obj.collect_asset_docs()
        self._describe_cache: ObjDict[DataKey] = dict()  # cache of all obj.describe() output
        self._describe_collect_cache: ObjDict[Dict[str, DataKey]] = dict()  # cache of all obj.describe() output

        self._config_desc_cache: ObjDict[DataKey] = dict()  # " obj.describe_configuration()
        self._config_values_cache: ObjDict[Any] = dict()  # " obj.read_configuration() values
        self._config_ts_cache: ObjDict[Any] = dict()  # " obj.read_configuration() timestamps
        # cache of {name: (doc, compose_event, compose_event_page)}
        self._descriptors: Dict[Any, ComposeDescriptorBundle] = dict()
        self._descriptor_objs: Dict[str, Set[HasName]] = dict()
        # cache of {obj: {objs_frozen_set: (doc, compose_event, compose_event_page)}
        self._local_descriptors: Dict[Any, Dict[FrozenSet[str], ComposeDescriptorBundle]] = dict()
        # a seq_num counter per stream
        self._sequence_counters: Dict[Any, int] = dict()
        self._sequence_counters_copy: Dict[Any, int] = dict()  # for if we redo data-points
        self._monitor_params: Dict[Subscribable, Tuple[Callback, Dict]] = dict()  # cache of {obj: (cb, kwargs)}
        self.run_is_open = False
        self._uncollected = set()  # objects after kickoff(), before collect()
        # we expect the RE to take care of the composition
        self._md = md
        # this is state on the RE, mirror it here rather than refer to
        # the parent
        self.record_interruptions = record_interruptions
        # this is RE.emit, but lifted to this context
        self.emit = emit
        self.emit_sync = emit_sync
        self.log = log

    async def open_run(self, msg):
        self.run_is_open = True
        self._run_start_uid = new_uid()
        self._interruptions_desc_uid = None  # uid for a special Event Desc.
        self._interruptions_counter = 0  # seq_num, special Event stream

        run = compose_run(uid=self._run_start_uid, event_counters=self._sequence_counters, metadata=self._md)
        doc = run.start_doc
        self._compose_descriptor = run.compose_descriptor
        self._compose_resource = run.compose_resource
        self._compose_stop = run.compose_stop
        self._compose_stream_resource = run.compose_stream_resource

        await self.emit(DocumentNames.start, doc)
        doc_logger.debug("[start] document is emitted (run_uid=%r)", self._run_start_uid,
                         extra={'doc_name': 'start',
                                'run_uid': self._run_start_uid})
        await self.reset_checkpoint_state_coro()

        # Emit an Event Descriptor for recording any interruptions as Events.
        if self.record_interruptions:
            # To store the interruptions uid outside of event-model
            self._interruptions_desc_uid = new_uid()
            dk = {"dtype": "string", "shape": [], "source": "RunEngine"}
            self._interruptions_desc, self._interruptions_compose_event, *_ = self._compose_descriptor(
                uid=self._interruptions_desc_uid,
                name="interruptions",
                data_keys={"interruption": dk},
            )
            await self.emit(DocumentNames.descriptor, self._interruptions_desc)

        return self._run_start_uid

    async def close_run(self, msg):
        """Instruct the RunEngine to write the RunStop document

        Expected message object is::

            Msg('close_run', None, exit_status=None, reason=None)

        if *exit_stats* and *reason* are not provided, use the values
        stashed on the RE.
        """
        if not self.run_is_open:
            raise IllegalMessageSequence(
                "A 'close_run' message was received but there is no run "
                "open. If this occurred after a pause/resume, add "
                "a 'checkpoint' message after the 'close_run' message."
            )
        self.log.debug("Stopping run %r", self._run_start_uid)
        # Clear any uncleared monitoring callbacks.
        for obj, (cb, kwargs) in list(self._monitor_params.items()):
            obj.clear_sub(cb)
            del self._monitor_params[obj]
        reason = msg.kwargs.get("reason", None)
        if reason is None:
            reason = ""
        exit_status = msg.kwargs.get("exit_status", "success") or "success"

        doc = self._compose_stop(
            exit_status=exit_status,
            reason=reason,
        )
        await self.emit(DocumentNames.stop, doc)
        doc_logger.debug("[stop] document is emitted (run_uid=%r)", self._run_start_uid,
                         extra={'doc_name': 'stop',
                                'run_uid': self._run_start_uid})
        await self.reset_checkpoint_state_coro()
        self.run_is_open = False
        return doc["run_start"]

    async def _prepare_stream(
        self,
        desc_key: str,
        objs_dks: Dict[Any, Dict[str, DataKey]],
    ):
        # We do not have an Event Descriptor for this set
        # so one must be created.
        data_keys = {}
        config = {}
        object_keys = {}
        hints = {}

        for obj, dks in objs_dks.items():
            maybe_update_hints(hints, obj)
            # dks is an OrderedDict. Record that order as a list.
            object_keys[obj.name] = list(dks)
            data_keys.update(dks)
            config[obj.name] = {
                "data": self._config_values_cache[obj],
                "timestamps": self._config_ts_cache[obj],
                "data_keys": self._config_desc_cache[obj]
            }

        self._descriptors[desc_key] = self._compose_descriptor(
            desc_key,
            data_keys,
            configuration=config,
            hints=hints,
            object_keys=object_keys,
        )
        await self.emit(DocumentNames.descriptor, self._descriptors[desc_key].descriptor_doc)
        doc_logger.debug(
            "[descriptor] document emitted with name %r containing "
            "data keys %r (run_uid=%r)",
            desc_key,
            data_keys.keys(),
            self._run_start_uid,
            extra={
                'doc_name': 'descriptor',
                'run_uid': self._run_start_uid,
                'data_keys': data_keys.keys()}
        )
        self._descriptor_objs[desc_key] = objs_dks
        if desc_key not in self._sequence_counters:
            self._sequence_counters[desc_key] = 1
            self._sequence_counters_copy[desc_key] = 1

        return (
            self._descriptors[desc_key].descriptor_doc,
            self._descriptors[desc_key].compose_event,
            list(objs_dks)
        )

    async def _ensure_cached(self, obj, collect=False):
        if (not collect and obj not in self._describe_cache):
            await self._cache_describe(obj)
        elif (collect and obj not in self._describe_collect_cache):
            await self._cache_describe_collect(obj)
        if obj not in self._config_desc_cache:
            await self._cache_describe_config(obj)
            await self._cache_read_config(obj)

    async def declare_stream(self, msg):
        """Generate and emit an EventDescriptor."""
        command, no_obj, objs, kwargs, _ = msg
        stream_name = kwargs['name']
        collect = kwargs.get('collect', False)
        assert no_obj is None
        objs = frozenset(objs)
        objs_dks = {}
        for obj in objs:
            await self._ensure_cached(obj, collect=collect)
            if collect:
                dks = self._describe_collect_cache[obj]
                formatted_data_keys = self._maybe_format_datakeys_with_stream_name(
                    dks, message_stream_name=stream_name
                )
                assert len(formatted_data_keys) == 1 \
                    and formatted_data_keys[0][0] == stream_name, \
                    (
                        "Expecting describe_collect to return a Dict[str, DataKey] "
                        f"for the passed in {stream_name}"
                    )
            else:
                dks = self._describe_cache[obj]

            objs_dks[obj] = dks

        return (await self._prepare_stream(stream_name, objs_dks))

    async def create(self, msg):
        """
        Start bundling future obj.read() calls for an Event document.

        Expected message object is::

            Msg('create', None, name='primary')
            Msg('create', name='primary')

        Note that the `name` kwarg will be the 'name' field of the resulting
        descriptor. So descriptor['name'] = msg.kwargs['name'].

        Also note that changing the 'name' of the Event will create a new
        Descriptor document.
        """
        if self.bundling:
            raise IllegalMessageSequence(
                "A second 'create' message is not "
                "allowed until the current event "
                "bundle is closed with a 'save' or "
                "'drop' message."
            )
        self._read_cache.clear()
        self._asset_docs_cache.clear()
        self._objs_read.clear()
        self.bundling = True
        command, obj, args, kwargs, _ = msg
        try:
            self._bundle_name = kwargs["name"]
        except KeyError:
            try:
                self._bundle_name, = args
            except ValueError:
                raise ValueError(
                    "Msg('create') now requires a stream name, given as "
                    "Msg('create', name) or Msg('create', name=name)"
                ) from None
        if self._strict_pre_declare:
            if self._bundle_name not in self._descriptors:
                raise IllegalMessageSequence(
                    "In strict mode you must pre-declare streams."
                )

    async def read(self, msg, reading):
        """
        Add a reading to the open event bundle.

        Expected message object is::

            Msg('read', obj)
        """
        if self.bundling:
            obj = msg.obj
            # if the object is not in the _describe_cache, cache it
            # Note: there is a race condition between the code here
            # and in monitor() and collect(), so if you do them concurrently
            # on the same device you make obj.describe() calls multiple times.
            # As this is harmless and not an expected use case, we don't guard
            # against it. Reading multiple devices concurrently works fine.
            await self._ensure_cached(obj)

            # check that current read collides with nothing else in
            # current event
            cur_keys = set(self._describe_cache[obj].keys())
            for read_obj in self._objs_read:
                # that is, field names
                known_keys = self._describe_cache[read_obj].keys()
                if set(known_keys) & cur_keys:
                    raise ValueError(
                        f"Data keys (field names) from {obj!r} "
                        f"collide with those from {read_obj!r}. "
                        f"The colliding keys are {set(known_keys) & cur_keys}"
                    )

            # add this object to the cache of things we have read
            self._objs_read.append(obj)

            # Stash the results, which will be emitted the next time _save is
            # called --- or never emitted if _drop is called instead.
            self._read_cache.append(reading)
            # Ask the object for any resource or datum documents is has cached
            # and cache them as well. Likewise, these will be emitted if and
            # when _save is called.
            self._asset_docs_cache.extend(
                [x async for x in maybe_collect_asset_docs(msg, obj, *msg.args, **msg.kwargs)]
            )

        return reading

    async def _cache_describe(self, obj):
        "Read the object's describe and cache it."
        obj = check_supports(obj, Readable)
        self._describe_cache[obj] = await maybe_await(obj.describe())

    async def _cache_describe_config(self, obj):
        "Read the object's describe_configuration and cache it."

        if isinstance(obj, Configurable):
            conf_keys = await maybe_await(obj.describe_configuration())
        else:
            conf_keys = {}

        self._config_desc_cache[obj] = conf_keys

    async def _cache_read_config(self, obj):
        "Read the object's configuration and cache it."
        if isinstance(obj, Configurable):
            conf = await maybe_await(obj.read_configuration())
        else:
            conf = {}
        config_values = {}
        config_ts = {}
        for key, val in conf.items():
            config_values[key] = val["value"]
            config_ts[key] = val["timestamp"]
        self._config_values_cache[obj] = config_values
        self._config_ts_cache[obj] = config_ts

    async def monitor(self, msg):
        """
        Monitor a signal. Emit event documents asynchronously.

        A descriptor document is emitted immediately. Then, a closure is
        defined that emits Event documents associated with that descriptor
        from a separate thread. This process is not related to the main
        bundling process (create/read/save).

        Expected message object is::

            Msg('monitor', obj, **kwargs)
            Msg('monitor', obj, name='event-stream-name', **kwargs)

        where kwargs are passed through to ``obj.subscribe()``
        """
        obj = check_supports(msg.obj, Subscribable)
        if msg.args:
            raise ValueError(
                "The 'monitor' Msg does not accept positional " "arguments."
            )
        kwargs = dict(msg.kwargs)
        name = kwargs.pop("name", short_uid("monitor"))
        if obj in self._monitor_params:
            raise IllegalMessageSequence(
                "A 'monitor' message was sent for {}"
                "which is already monitored".format(obj)
            )

        await self._ensure_cached(obj)

        _, compose_event, _ = await self._prepare_stream(name, {obj: self._describe_cache[obj]})

        def emit_event(readings: Dict[str, Reading] = None, *args, **kwargs):
            if readings is not None:
                # We were passed something we can use, but check no args or kwargs
                assert not args and not kwargs, \
                    "If subscribe callback called with readings, " \
                    "args and kwargs are not supported."
            else:
                # Ignore the inputs. Use this call as a signal to call read on the
                # object, a crude way to be sure we get all the info we need.
                readable_obj = check_supports(obj, Readable)
                readings = readable_obj.read()
                assert not inspect.isawaitable(readings), \
                    f"{readable_obj} has async read() method and the callback " \
                    "passed to subscribe() was not called with Dict[str, Reading]"
            data, timestamps = _rearrange_into_parallel_dicts(readings)
            doc = compose_event(
                data=data,
                timestamps=timestamps,
            )
            self.emit_sync(DocumentNames.event, doc)

        self._monitor_params[obj] = emit_event, kwargs
        # TODO: deprecate **kwargs when Ophyd.v2 is available
        obj.subscribe(emit_event, **kwargs)

    def record_interruption(self, content):
        """
        Emit an event in the 'interruptions' event stream.

        If we are not inside a run or if self.record_interruptions is False,
        nothing is done.
        """
        if self._interruptions_desc_uid is not None:
            # We are inside a run and self.record_interruptions is True.
            doc = self._interruptions_compose_event(
                data={"interruption": content},
                timestamps={"interruption": ttime.time()},
            )
            self._interruptions_counter += 1
            self.emit_sync(DocumentNames.event, doc)

    def rewind(self):
        self._sequence_counters.clear()
        self._sequence_counters.update(self._sequence_counters_copy)
        # This is needed to 'cancel' an open bundling (e.g. create) if
        # the pause happens after a 'checkpoint', after a 'create', but
        # before the paired 'save'.
        self.bundling = False

    async def unmonitor(self, msg):
        """
        Stop monitoring; i.e., remove the callback emitting event documents.

        Expected message object is::

            Msg('unmonitor', obj)
        """
        obj = check_supports(msg.obj, Subscribable)
        if obj not in self._monitor_params:
            raise IllegalMessageSequence(
                f"Cannot 'unmonitor' {obj}; it is not " "being monitored."
            )
        cb, kwargs = self._monitor_params[obj]
        obj.clear_sub(cb)
        del self._monitor_params[obj]
        await self.reset_checkpoint_state_coro()

    async def save(self, msg):
        """Save the event that is currently being bundled

        Create and emit an Event document containing the data read from devices
        in self._objs_read. Emit any Resource and Datum documents cached by
        those devices before emitting the Event document. If this is the first
        Event of its stream then create and emit the Event Descriptor document
        before emitting Resource, Datum, and Event documents.

        Expected message object is::

            Msg('save')
        """
        if not self.bundling:
            raise IllegalMessageSequence(
                "A 'create' message must be sent, to "
                "open an event bundle, before that "
                "bundle can be saved with 'save'."
            )

        # Short-circuit if nothing has been read. (Do not create empty Events.)
        if not self._objs_read:
            self.bundling = False
            self._bundle_name = None
            return
        # The Event Descriptor is uniquely defined by the set of objects
        # read in this Event grouping.
        objs_read = frozenset(self._objs_read)

        # Event Descriptor key
        desc_key = self._bundle_name

        # This is a separate check because it can be reset on resume.
        self.bundling = False
        self._bundle_name = None

        descriptor_doc, compose_event, _, = self._descriptors.get(
            desc_key, (None, None, None)
        )
        d_objs = self._descriptor_objs.get(desc_key, None)

        objs_dks = {}

        # we do not have the descriptor cached, make it
        if descriptor_doc is None or d_objs is None:
            for obj in objs_read:
                await self._ensure_cached(obj, collect=isinstance(obj, Collectable))
                objs_dks[obj] = self._describe_cache[obj]

            descriptor_doc, compose_event, d_objs = await self._prepare_stream(desc_key, objs_dks)
            # do have the descriptor cached
        elif frozenset(d_objs) != objs_read:
            raise RuntimeError(
                "Mismatched objects read, expected {!s}, "
                "got \n\n\n{!s}".format(frozenset(d_objs), objs_read)
            )

        # Resource and Datum documents
        for resource_or_datum_name, resource_or_datum_doc in self._asset_docs_cache:
            # Add a 'run_start' field to resource documents on their way out
            # since this field could not have been set correctly before this point.
            self._pack_external_assets(resource_or_datum_name, resource_or_datum_doc, message_stream_name=desc_key)

            doc_logger.debug(
                "[%s] document emitted %r",
                resource_or_datum_name,
                resource_or_datum_doc,
                extra={
                    "doc_name": resource_or_datum_name,
                    "run_uid": self._run_start_uid,
                    "doc": resource_or_datum_doc
                }
            )

            await self.emit(
                DocumentNames(resource_or_datum_name),
                resource_or_datum_doc
            )

        # Merge list of readings into single dict.
        readings = {k: v for d in self._read_cache for k, v in d.items()}
        data, timestamps = _rearrange_into_parallel_dicts(readings)
        # Mark all externally-stored data as not filled so that consumers
        # know that the corresponding data are identifiers, not dereferenced
        # data.
        filled = {
            k: False
            for k, v in self._descriptors[desc_key].descriptor_doc["data_keys"].items()
            if "external" not in v or v["external"] != "STREAM:"
        }
        event_doc = compose_event(
            data=data,
            timestamps=timestamps,
            filled=filled,
        )
        await self.emit(DocumentNames.event, event_doc)
        doc_logger.debug(
            "[event] document emitted with data keys %r (run_uid=%r)",
            data.keys(),
            self._run_start_uid,
            extra={
                'doc_name': 'event',
                'run_uid': self._run_start_uid,
                'data_keys': data.keys()}
        )

    def clear_monitors(self):
        for obj, (cb, kwargs) in list(self._monitor_params.items()):
            try:
                obj.clear_sub(cb)
            except Exception:
                self.log.exception("Failed to stop monitoring %r.", obj)
            else:
                del self._monitor_params[obj]

    def reset_checkpoint_state(self):

        # Keep a safe separate copy of the sequence counters to use if we
        # rewind and retake some data points.

        for key, counter in list(self._sequence_counters.items()):
            self._sequence_counters_copy[key] = counter

    async def reset_checkpoint_state_coro(self):
        self.reset_checkpoint_state()

    async def suspend_monitors(self):
        for obj, (cb, kwargs) in self._monitor_params.items():
            obj.clear_sub(cb)

    async def restore_monitors(self):
        for obj, (cb, kwargs) in self._monitor_params.items():
            obj.subscribe(cb, **kwargs)

    async def clear_checkpoint(self, msg):
        self._sequence_counters_copy.clear()

    async def drop(self, msg):
        """Drop the event that is currently being bundled

        Expected message object is::

            Msg('drop')
        """
        if not self.bundling:
            raise IllegalMessageSequence(
                "A 'create' message must be sent, to "
                "open an event bundle, before that "
                "bundle can be dropped with 'drop'."
            )

        self.bundling = False
        self._bundle_name = None
        self.log.debug("Dropped open event bundle")

    async def kickoff(self, msg):
        """Start a flyscan object.

        Expected message object is:

        If `flyer_object` has a `kickoff` function that takes no arguments::

            Msg('kickoff', flyer_object)
            Msg('kickoff', flyer_object, group=<name>)

        If *flyer_object* has a ``kickoff`` function that takes
        ``(start, stop, steps)`` as its function arguments::

            Msg('kickoff', flyer_object, start, stop, step)
            Msg('kickoff', flyer_object, start, stop, step, group=<name>)

        """
        self._uncollected.add(msg.obj)

    async def complete(self, msg):
        """
        Tell a flyer, 'stop collecting, whenever you are ready'.

        The flyer returns a status object. Some flyers respond to this
        command by stopping collection and returning a finished status
        object immediately. Other flyers finish their given course and
        finish whenever they finish, irrespective of when this command is
        issued.

        Expected message object is::

            Msg('complete', flyer, group=<GROUP>)

        where <GROUP> is a hashable identifier.
        """
        ...

    def _maybe_format_datakeys_with_stream_name(
        self,
        describe_collect_dict: Union[Dict[str, DataKey], Dict[str, Dict[str, DataKey]]],
        message_stream_name: Optional[str] = None,
    ) -> List[Tuple[str, Dict[str, DataKey]]]:
        """
        Check if the dictionary returned by describe collect is a dict
            `{str: DataKey}` or a `{str: {str: DataKey}}`.
        If a `message_stream_name` is passed then return a singleton list of the form of
            `{message_stream_name: describe_collect_dict}.items()`.
        If the `message_stream_name` is None then return the `describe_collect_dict.items()`.
        """
        def has_str_source(d: dict):
            return isinstance(d, dict) and isinstance(d.get("source", None), str)
        if describe_collect_dict:
            first_value = list(describe_collect_dict.values())[0]
            if has_str_source(first_value):
                # We have Dict[str, DataKey], so return just this
                # If stream name not given then default to "primary"
                return [(message_stream_name or "primary", describe_collect_dict)]
            elif all(has_str_source(v) for v in first_value.values()):
                # We have Dict[str, Dict[str, DataKey]] so return its items
                if message_stream_name and list(describe_collect_dict) != [message_stream_name]:
                    # The collect contained a name and describe_collect returned a Dict[str, Dict[str, DataKey]],
                    # this is only acceptable if the only key in the parent dict is message_stream_name

                    raise RuntimeError(
                        f"Expected a single stream {message_stream_name!r}, got {describe_collect_dict}"
                    )
                return list(describe_collect_dict.items())
            else:
                raise RuntimeError(
                    f"Invalid describe_collect return: {describe_collect_dict} when collect "
                    f"was called on {message_stream_name}"
                )
        else:
            # Empty dict, could be either but we don't care
            return []

    async def _cache_describe_collect(self, obj: Collectable):
        "Read the object's describe and cache it."
        obj = check_supports(obj, Collectable)
        self._describe_collect_cache[obj] = await maybe_await(obj.describe_collect())

    async def _describe_collect(self, collect_obj: Flyable, message_stream_name: Optional[str] = None):
        "Read the object's describe_collect and cache it."

        await self._ensure_cached(collect_obj, collect=True)

        describe_collect = self._describe_collect_cache[collect_obj]

        describe_collect_items = list(self._maybe_format_datakeys_with_stream_name(
                describe_collect, message_stream_name=message_stream_name
        ))

        local_descriptors: Dict[Any, Dict[FrozenSet[str], ComposeDescriptorBundle]] = {}

        # collect_obj.describe_collect() returns a dictionary like this:
        #     {name_for_desc1: data_keys_for_desc1,
        #      name_for_desc2: data_keys_for_desc2, ...}
        for stream_name, stream_data_keys in describe_collect_items:
            if stream_name not in self._descriptors:
                # We do not have an Event Descriptor for this set.
                # if we have not yet read the configuration, do so
                await self._prepare_stream(stream_name, {collect_obj: stream_data_keys})
            else:
                objs_read = self._descriptor_objs[stream_name]
                if stream_data_keys != objs_read[collect_obj]:
                    raise RuntimeError(
                        "Mismatched objects read, "
                        "expected {!s}, "
                        "got {!s}".format(stream_data_keys, objs_read)
                    )

            local_descriptors[frozenset(stream_data_keys)] = self._descriptors[stream_name]

        self._local_descriptors[collect_obj] = local_descriptors

    async def _pack_external_assets(self, name, doc, message_stream_name=None):
        """Packs some external asset document `doc` with relevant information from the run."""
        if name in ["resource", "stream_resource"]:
            doc["run_start"] = self._run_start_uid
        elif name == "stream_datum":
            doc["descriptor"] = self._descriptors[message_stream_name]["uid"]
        elif name in ["datum"]:
            ...
        else:
            raise RuntimeError(
                f"Tried to emit an external asset {name}, acceptable external assets are "
                "`resource`, `stream_resource`, `datum`, or `stream_datum`"
            )

    async def _collect_events(
        self, collect_obj, local_descriptors, return_payload: bool, stream: bool, message_stream_name: str
    ):
        event_list: List[PartialEvent] = []
        payload = []

        if message_stream_name:
            compose_event = self._descriptors[message_stream_name].compose_event

        for ev in collect_obj.collect():
            if return_payload:
                payload.append(ev)

            if not message_stream_name:
                objs_read = frozenset(ev["data"])
                compose_event = local_descriptors[objs_read].compose_event

            ev = compose_event(data=ev["data"], timestamps=ev["timestamps"])

            if stream:
                doc_logger.debug(
                    "[event] document is emitted with data keys %r (run_uid=%r)",
                    ev['data'].keys(), self._run_start_uid,
                    ev["uid"],
                    extra={
                        'doc_name': 'event',
                        'run_uid': self._run_start_uid,
                        'data_keys': ev['data'].keys()
                    }
                )
                await self.emit(DocumentNames.event, ev)
            else:
                event_list.append(ev)

        if not stream:
            await self.emit(DocumentNames.event_page, pack_event_page(*event_list))
            doc_logger.debug(
                "[event_page] document is emitted for descriptors (run_uid=%r)",
                self._run_start_uid,
                extra={
                    'doc_name': 'event_page',
                    'run_uid': self._run_start_uid
                }
            )
        return payload

    async def _collect_event_pages(
            self, collect_obj, local_descriptors, return_payload: bool, message_stream_name: str
            ):
        payload = []

        if message_stream_name:
            compose_event_page = self._descriptors[message_stream_name].compose_event_page

        for ev_page in collect_obj.collect_pages():
            if return_payload:
                payload.append(ev_page)

            if not message_stream_name:
                objs_read = frozenset(ev_page["data"])
                compose_event_page = local_descriptors[objs_read].compose_event_page

            ev_page = compose_event_page(data=ev_page["data"], timestamps=ev_page["timestamps"])
            doc_logger.debug(
                "[event_page] document is emitted with data keys %r (run_uid=%r)",
                ev_page['data'].keys(), self._run_start_uid,
                ev_page["uid"],
                extra={
                    'doc_name': 'event_page',
                    'run_uid': self._run_start_uid,
                    'data_keys': ev_page['data'].keys()
                }
            )

            await self.emit(DocumentNames.event_page, ev_page)
        return payload

    async def collect(self, msg):
        """
        Collect data cached by a flyer and emit documents.

        Expect message object is

            Msg('collect', collect_obj)
            Msg('collect', flyer_object, stream=True, return_payload=False, name='stream_name')
        """

        collect_obj = check_supports(msg.obj, Collectable)
        assert not (isinstance(collect_obj, EventCollectable) and isinstance(collect_obj, EventPageCollectable)),\
            "collect() was called for a device which is both EventCollectable and EventPageCollectable. "\
            "If you want to have an EventCollectable device format only some events as event_pages "\
            "then use `stream=False`."

        if not self.run_is_open:
            # sanity check -- 'kickoff' should catch this and make this
            # code path impossible
            raise IllegalMessageSequence(
                "A 'collect' message was sent but no run is open."
            )
        self._uncollected.discard(collect_obj)

        # If message_stream_name is given then only one descriptor is generated on `describe_collect`
        message_stream_name = msg.kwargs.get("name", None)

        # If stream is True, run 'event' subscription per document.
        # If stream is False, run 'event_page' subscription once.
        stream = msg.kwargs.get("stream", False)

        # If True, accumulate all the Events in memory and return them at the
        # end, providing the plan access to the Events. If False, do not
        # accumulate, and return None.
        return_payload = msg.kwargs.get('return_payload', True)

        # Obtain `local_descriptors` and describe collect depending on if `message_stream_name` was passed in
        if message_stream_name:
            if message_stream_name not in self._descriptors:
                await self._describe_collect(collect_obj, message_stream_name=message_stream_name)
        else:
            if collect_obj not in self._local_descriptors:
                await self._describe_collect(collect_obj)

        local_descriptors = self._local_descriptors[collect_obj]

        # Pack a Resource or Datum document with relevant information and emit
        async for name, doc in maybe_collect_asset_docs(msg, collect_obj):
            self._pack_external_assets(name, doc, message_stream_name=message_stream_name)
            await self.emit(DocumentNames(name), doc)

        if isinstance(collect_obj, EventCollectable):
            payload = await self._collect_events(
                collect_obj, local_descriptors, return_payload, stream, message_stream_name
            )
        elif isinstance(collect_obj, EventPageCollectable):
            if stream:
                raise IllegalMessageSequence(
                    "A 'collect' with `stream=True` was sent for an EventPageCollectable device, "
                    "stream is not used for EventPages"
                )
            payload = await self._collect_event_pages(
                collect_obj, local_descriptors, return_payload, message_stream_name
            )
        else:
            return_payload = False

        if return_payload:
            return payload

    async def backstop_collect(self):
        for obj in list(self._uncollected):
            try:
                await self.collect(Msg("collect", obj))
            except Exception:
                self.log.exception("Failed to collect %r.", obj)

    async def configure(self, msg):
        """Configure an object

        Expected message object is ::

            Msg('configure', object, *args, **kwargs)

        which results in this call ::

            object.configure(*args, **kwargs)
        """
        obj = msg.obj
        # Invalidate any event descriptors that include this object.
        # New event descriptors, with this new configuration, will
        # be created for any future event documents.
        for name in list(self._descriptors):
            obj_set = self._descriptor_objs[name]
            if obj in obj_set:
                del self._descriptors[name]
                await self._prepare_stream(name, obj_set)
                continue

        await self._cache_read_config(obj)
