import asyncio
import inspect
import time as ttime
from collections import defaultdict, deque
from collections.abc import Callable, Iterable
from itertools import combinations
from logging import LoggerAdapter
from typing import Any, Literal, TypeAlias, TypeGuard, cast

from event_model import (
    ComposeDescriptorBundle,
    DataKey,
    Datum,
    DocumentNames,
    EventModelValueError,
    Resource,
    StreamDatum,
    StreamRange,
    StreamResource,
    compose_run,
    pack_event_page,
)
from event_model.documents.event import Event

from .log import doc_logger
from .protocols import (
    Asset,
    Callback,
    Collectable,
    Configurable,
    EventCollectable,
    EventPageCollectable,
    Flyable,
    HasName,
    Readable,
    Reading,
    StreamAsset,
    Subscribable,
    SyncOrAsync,
    T,
    WritesStreamAssets,
    check_supports,
)
from .utils import (
    IllegalMessageSequence,
    Msg,
    _rearrange_into_parallel_dicts,
    iterate_maybe_async,
    maybe_await,
    maybe_collect_asset_docs,
    maybe_update_hints,
    new_uid,
    short_uid,
)

ObjDict = dict[Any, dict[str, T]]
ExternalAssetDoc: TypeAlias = Datum | Resource | StreamDatum | StreamResource


def _describe_collect_dict_is_valid(
    describe_collect_dict: Any | dict[str, Any],
) -> TypeGuard[dict[str, DataKey]]:
    """
    Check if the describe_collect dictionary contains valid DataKeys.
    """
    if not isinstance(describe_collect_dict, dict):
        return False
    return all(
        isinstance(value, dict) and isinstance(value.get("source", None), str)
        for value in describe_collect_dict.values()
    )


class RunBundler:
    def __init__(
        self,
        md: dict | None,
        record_interruptions: bool,
        emit: Callable,
        emit_sync: Callable,
        log: LoggerAdapter,
        *,
        strict_pre_declare: bool,
    ):
        # if create can YOLO implicitly create a stream
        self._strict_pre_declare = strict_pre_declare
        # state stolen from the RE
        self.bundling = False  # if we are in the middle of bundling readings
        self._bundle_name = None  # name given to event descriptor
        self._run_start_uid = None  # The (future) runstart uid
        self._objs_read: deque[HasName] = deque()  # objects read in one Event
        self._read_cache: deque[dict[str, Reading]] = deque()  # cache of obj.read() in one Event
        self._asset_docs_cache: deque[Asset | StreamAsset] = deque()  # cache of obj.collect_asset_docs()
        self._describe_cache: ObjDict[DataKey] = dict()  # cache of all obj.describe() output  # noqa: C408
        self._describe_collect_cache: dict[Any, dict[str, DataKey] | dict[str, dict[str, DataKey]]] = dict()  # noqa: C408  # cache of all obj.describe() output

        self._config_desc_cache: ObjDict[DataKey] = dict()  # " obj.describe_configuration()  # noqa: C408
        self._config_values_cache: ObjDict[Any] = dict()  # " obj.read_configuration() values  # noqa: C408
        self._config_ts_cache: ObjDict[Any] = dict()  # " obj.read_configuration() timestamps  # noqa: C408
        # cache of {name: (doc, compose_event, compose_event_page)}
        self._descriptors: dict[Any, ComposeDescriptorBundle] = dict()  # noqa: C408
        self._descriptor_objs: dict[str, dict[HasName, dict[str, DataKey]]] = dict()  # noqa: C408
        # cache of {obj: {objs_frozen_set: (doc, compose_event, compose_event_page)}
        self._local_descriptors: dict[Any, dict[frozenset[str], ComposeDescriptorBundle]] = dict()  # noqa: C408
        # a seq_num counter per stream
        self._sequence_counters: dict[Any, int] = dict()  # noqa: C408
        self._sequence_counters_copy: dict[Any, int] = dict()  # for if we redo data-points  # noqa: C408
        self._monitor_params: dict[Subscribable, tuple[Callback, dict]] = dict()  # noqa: C408  # cache of {obj: (cb, kwargs)}
        # a cache of stream_resource uid to the data_keys that stream_resource collects for
        self._stream_resource_data_keys: dict[str, Iterable[str]] = dict()  # noqa: C408
        self.run_is_open = False
        self._uncollected: set[HasName] = set()  # objects after kickoff(), before collect()
        # we expect the RE to take care of the composition
        self._md = md
        # this is state on the RE, mirror it here rather than refer to
        # the parent
        self.record_interruptions = record_interruptions
        # this is RE.emit, but lifted to this context
        self.emit = emit
        self.emit_sync = emit_sync
        self.log = log
        # Map of set of collect objects to list of stream names that they can be collected into
        self._declared_stream_names: dict[frozenset, list[str]] = {}

    async def open_run(self, msg: Msg):
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
        doc_logger.debug(
            "[start] document is emitted (run_uid=%r)",
            self._run_start_uid,
            extra={"doc_name": "start", "run_uid": self._run_start_uid},
        )
        await self.reset_checkpoint_state_coro()

        # Emit an Event Descriptor for recording any interruptions as Events.
        if self.record_interruptions:
            # To store the interruptions uid outside of event-model
            self._interruptions_desc_uid = new_uid()
            dk = {"dtype": "string", "shape": [], "source": "RunEngine"}
            descriptor_bundle = self._compose_descriptor(
                uid=self._interruptions_desc_uid,
                name="interruptions",
                data_keys={"interruption": dk},
            )
            self._interruptions_desc = descriptor_bundle.descriptor_doc
            self._interruptions_compose_event = descriptor_bundle.compose_event
            await self.emit(DocumentNames.descriptor, self._interruptions_desc)

        return self._run_start_uid

    async def close_run(self, msg: Msg):
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
        for obj, (cb, kwargs) in list(self._monitor_params.items()):  # noqa: B007
            obj.clear_sub(cb)
            del self._monitor_params[obj]
        reason = msg.kwargs.get("reason", None)
        if reason is None:
            reason = ""

        exit_status: Literal["success", "abort", "fail"] = msg.kwargs.get("exit_status", "success") or "success"
        doc = self._compose_stop(
            exit_status=exit_status,
            reason=reason,
        )
        await self.emit(DocumentNames.stop, doc)
        doc_logger.debug(
            "[stop] document is emitted (run_uid=%r)",
            self._run_start_uid,
            extra={"doc_name": "stop", "run_uid": self._run_start_uid},
        )
        await self.reset_checkpoint_state_coro()
        self.run_is_open = False
        return doc["run_start"]

    async def _prepare_stream(
        self,
        desc_key: str,
        objs_dks: dict[HasName, dict[str, DataKey]],
    ):
        # We do not have an Event Descriptor for this set
        # so one must be created.
        data_keys = {}
        config = {}
        object_keys = {}
        hints: dict[str, Any] = {}

        for obj, dks in objs_dks.items():
            maybe_update_hints(hints, obj)
            # dks is an OrderedDict. Record that order as a list.
            object_keys[obj.name] = list(dks)
            for key in dks.keys():
                dks[key]["object_name"] = obj.name
            data_keys.update(dks)
            config[obj.name] = {
                "data": self._config_values_cache[obj],
                "timestamps": self._config_ts_cache[obj],
                "data_keys": self._config_desc_cache[obj],
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
            "[descriptor] document emitted with name %r containing data keys %r (run_uid=%r)",
            desc_key,
            data_keys.keys(),
            self._run_start_uid,
            extra={"doc_name": "descriptor", "run_uid": self._run_start_uid, "data_keys": data_keys.keys()},
        )
        self._descriptor_objs[desc_key] = objs_dks
        if desc_key not in self._sequence_counters:
            self._sequence_counters[desc_key] = 1
            self._sequence_counters_copy[desc_key] = 1

        return (
            self._descriptors[desc_key].descriptor_doc,
            self._descriptors[desc_key].compose_event,
            list(objs_dks),
        )

    async def _ensure_cached(self, obj, collect=False):
        coros = []
        if not collect and obj not in self._describe_cache:
            coros.append(self._cache_describe(obj))
        elif collect and obj not in self._describe_collect_cache:
            coros.append(self._cache_describe_collect(obj))
        if obj not in self._config_desc_cache:
            coros.append(self._cache_describe_config(obj))
            coros.append(self._cache_read_config(obj))
        await asyncio.gather(*coros)

    async def declare_stream(self, msg):
        """Generate and emit an EventDescriptor."""
        command, no_obj, objs, kwargs, _ = msg
        stream_name = kwargs.get("name")
        assert stream_name is not None, "A stream name that is not None is required for pre-declare"

        collect = kwargs.get("collect", False)
        assert no_obj is None
        objs = frozenset(objs)
        objs_dks = {}  # {collect_object: stream_data_keys}

        await asyncio.gather(*[self._ensure_cached(obj, collect=collect) for obj in objs])
        for obj in objs:
            if collect:
                data_keys = self._describe_collect_cache[obj]
                streams_and_data_keys = self._format_datakeys_with_stream_name(
                    data_keys, message_stream_name=stream_name
                )

                # ensure that there is only one stream and it is the stream we have provided.
                assert len(streams_and_data_keys) == 1 and streams_and_data_keys[0][0] == stream_name, (
                    "`declare_stream` contained `collect=True` but  `describe_collect` did "
                    f"not return a single Dict[str, DataKey] for the passed in {stream_name}"
                )
            else:
                data_keys = self._describe_cache[obj]

            objs_dks[obj] = data_keys

        existing_stream_names = self._declared_stream_names.setdefault(objs, [])
        existing_stream_names.append(stream_name)

        return await self._prepare_stream(stream_name, objs_dks)

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
                (self._bundle_name,) = args
            except ValueError:
                raise ValueError(
                    "Msg('create') now requires a stream name, given as "
                    "Msg('create', name) or Msg('create', name=name)"
                ) from None
        if self._strict_pre_declare:
            if self._bundle_name not in self._descriptors:
                raise IllegalMessageSequence("In strict mode you must pre-declare streams.")

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
            asset_docs_collected = [x async for x in maybe_collect_asset_docs(msg, obj, *msg.args, **msg.kwargs)]
            self._asset_docs_cache.extend(asset_docs_collected)

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
            raise ValueError("The 'monitor' Msg does not accept positional arguments.")
        kwargs = dict(msg.kwargs)
        name = kwargs.pop("name", short_uid("monitor"))
        if obj in self._monitor_params:
            raise IllegalMessageSequence(f"A 'monitor' message was sent for {obj} which is already monitored")

        await self._ensure_cached(obj)

        stream_bundle = await self._prepare_stream(name, {obj: self._describe_cache[obj]})
        compose_event = stream_bundle[1]

        def emit_event(readings: dict[str, Reading] | None = None, *args, **kwargs):
            if readings is not None:
                # We were passed something we can use, but check no args or kwargs
                assert not args and not kwargs, (
                    "If subscribe callback called with readings, args and kwargs are not supported."
                )
            else:
                # Ignore the inputs. Use this call as a signal to call read on the
                # object, a crude way to be sure we get all the info we need.
                readable_obj = check_supports(obj, Readable)  # type: ignore
                readings = readable_obj.read()  # type: ignore
                assert not inspect.isawaitable(readings), (
                    f"{readable_obj} has async read() method and the callback "
                    "passed to subscribe() was not called with Dict[str, Reading]"
                )
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
        # make sure we do not forget about streams we roll back to the
        # very beginning of
        for desc_key in self._descriptor_objs:
            if desc_key not in self._sequence_counters:
                self._sequence_counters[desc_key] = 1
                self._sequence_counters_copy[desc_key] = 1

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
            raise IllegalMessageSequence(f"Cannot 'unmonitor' {obj}; it is not being monitored.")
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

        (
            descriptor_doc,
            compose_event,
            _,
        ) = self._descriptors.get(desc_key, (None, None, None))
        d_objs = self._descriptor_objs.get(desc_key, None)

        objs_dks = {}

        # we do not have the descriptor cached, make it
        if descriptor_doc is None or d_objs is None:
            # use the dequeue not the set to preserve order
            for obj in self._objs_read:
                await self._ensure_cached(obj, collect=isinstance(obj, Collectable))
                objs_dks[obj] = self._describe_cache[obj]

            descriptor_doc, compose_event, d_objs = await self._prepare_stream(desc_key, objs_dks)

        # do have the descriptor cached
        elif frozenset(d_objs) != objs_read:
            raise RuntimeError(f"Mismatched objects read, expected {frozenset(d_objs)!s}, got {objs_read!s}")

        # Resource and Datum documents
        indices_generated = await self._pack_external_assets(self._asset_docs_cache, message_stream_name=desc_key)
        if indices_generated > 1:
            raise RuntimeError(
                "Received multiple indices in a `stream_datum` document for one event, "
                ' during a `read()` `save()`. `stream_datum` should have indices {"start": n, "stop": n+1} '
                "in a `read()` `save()`."
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
            if "external" in v and v["external"] != "STREAM:"
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
            extra={"doc_name": "event", "run_uid": self._run_start_uid, "data_keys": data.keys()},
        )

    def clear_monitors(self):
        for obj, (cb, kwargs) in list(self._monitor_params.items()):  # noqa: B007
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
        for obj, (cb, kwargs) in self._monitor_params.items():  # noqa: B007
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

    def _format_datakeys_with_stream_name(
        self,
        describe_collect_dict: dict[str, DataKey] | dict[str, dict[str, DataKey]],
        message_stream_name: str | None = None,
    ) -> list[tuple[str, dict[str, DataKey]]]:
        """
        Check if the dictionary returned by describe collect is a dict
            `{str: DataKey}` or a `{str: {str: DataKey}}`.
        If a `message_stream_name` is passed then return a singleton list of the form of
            `{message_stream_name: describe_collect_dict}.items()`.
        If the `message_stream_name` is None then return the `describe_collect_dict.items()`.
        """

        def _contains_message_stream_name(
            describe_collect_dict: Any | dict[str, Any],
        ) -> TypeGuard[dict[str, dict[str, DataKey]]]:
            return isinstance(describe_collect_dict, dict) and all(
                _describe_collect_dict_is_valid(v) for v in describe_collect_dict.values()
            )

        if describe_collect_dict:
            if _describe_collect_dict_is_valid(describe_collect_dict):
                flat_describe_collect_dict = describe_collect_dict
                return [(message_stream_name or "primary", flat_describe_collect_dict)]
            # Validate that all of the values nested values are DataKeys
            elif _contains_message_stream_name(describe_collect_dict):
                # We have Dict[str, Dict[str, DataKey]] so return its items
                nested_describe_collect_dict = describe_collect_dict
                if message_stream_name and list(nested_describe_collect_dict) != [message_stream_name]:
                    # The collect contained a name and describe_collect returned a Dict[str, Dict[str, DataKey]],
                    # this is only acceptable if the only key in the parent dict is message_stream_name

                    raise RuntimeError(
                        f"Expected a single stream {message_stream_name!r}, got {describe_collect_dict}"
                    )
                return list(nested_describe_collect_dict.items())
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
        c: dict[str, DataKey] | dict[str, dict[str, DataKey]] = await maybe_await(obj.describe_collect())
        self._describe_collect_cache[obj] = c

    async def _describe_collect(self, collect_object: Flyable):
        """Read an object's describe_collect and cache it.

        Read describe collect for a collect_object and ensure it is cached in the
        _describe_collect_cache. This is required for scans of a single collect object
        where the data structure is doubly nested. In this case calling
        describe_collect on the object returns a data structure like so:

            {
                "stream1" : {"stream1-pv1":{data_keys}, "stream1-pv2" :{data_keys}},
                "stream2" : {"stream2-pv1":{data_keys}, "stream2-pv2" :{data_keys}}
            }

        Single nested data keys should be rejected since they are new style, and are
        collected under one stream. They should be pre-declared with declare-stream,
        prior to collecting. The describe_collect on this object returns a data
        structure like so:
            {
                "stream1-pv1": {data_keys},
                "stream1-pv1": {data_keys}
            }

        """
        await self._ensure_cached(collect_object, collect=True)

        describe_collect = self._describe_collect_cache[collect_object]
        describe_collect_items = self._format_datakeys_with_stream_name(describe_collect)

        local_descriptors: dict[frozenset[str], ComposeDescriptorBundle] = {}

        # Check that singly nested stuff should have been pre-declared
        def is_data_key(obj: Any) -> bool:
            return isinstance(obj, dict) and {"dtype", "shape", "source"}.issubset(frozenset(obj.keys()))

        assert all(not is_data_key(value) for value in describe_collect.values()), (
            "Single nested data keys should be pre-declared"
        )

        # Make sure you can't use identical data keys in multiple streams
        # Data structure is assumed to be dict[stream_name, dictionary of key -> data_key]
        duplicates: dict[str, dict[str, DataKey]] = defaultdict(dict)
        for (
            (stream1, data_keys1),
            (stream2, data_keys2),
        ) in combinations(describe_collect_items, 2):
            for common_key in data_keys1.keys() & data_keys2.keys():
                duplicates[stream1][common_key] = data_keys1[common_key]
                duplicates[stream2][common_key] = data_keys2[common_key]
        if len(duplicates) > 0:
            raise RuntimeError(
                "Can't use identical data keys in multiple streams:",
                f"Data keys: {list(duplicates.values())}",
                f"streams: {duplicates.keys()}",
            )

        for stream_name, stream_data_keys in describe_collect_items:
            if stream_name not in self._descriptor_objs or (
                collect_object not in self._descriptor_objs[stream_name]
            ):
                await self._prepare_stream(stream_name, {collect_object: stream_data_keys})
            else:
                objs_read = self._descriptor_objs[stream_name]
                if stream_data_keys != objs_read[collect_object]:
                    raise RuntimeError(
                        f"Mismatched objects read, expected {stream_data_keys!s}, got {objs_read!s}"
                    )
            local_descriptors[frozenset(stream_data_keys)] = self._descriptors[stream_name]

        self._local_descriptors[collect_object] = local_descriptors

    async def _pack_seq_nums_into_stream_datum(
        self, doc: StreamDatum, message_stream_name: str, stream_datum_previous_indices_difference: int
    ) -> int:
        if doc["seq_nums"] != StreamRange(start=0, stop=0):
            raise EventModelValueError(
                f"Received `seq_nums` {doc['seq_nums']} in stream {doc['stream_resource']} "
                "during `collect()` or `describe_collect()`. `seq_nums` should be None or "
                "`StreamRange(start=0, stop=0)` on `ComposeStreamDatum` when used with the "
                "run engine."
            )

        indices_difference = doc["indices"]["stop"] - doc["indices"]["start"]

        if (
            stream_datum_previous_indices_difference
            and stream_datum_previous_indices_difference != indices_difference
        ):
            raise EventModelValueError(
                f"Received `indices` {doc['indices']} during `collect()` these are of a different "
                f"width `{indices_difference}` than other detectors in the same collect() or save()."
            )

        current_seq_counter = self._sequence_counters[message_stream_name]
        doc["seq_nums"] = StreamRange(start=current_seq_counter, stop=current_seq_counter + indices_difference)

        if doc["stream_resource"] not in self._stream_resource_data_keys:
            raise RuntimeError(
                f"Receieved a `steam_datum` referring to an unknown stream resource {doc['stream_resource']}"
            )

        return indices_difference

    # message strem name here?
    async def _pack_external_assets(
        self, asset_docs: Iterable[Asset | StreamAsset], message_stream_name: str | None
    ):
        """Packs some external asset documents with relevant information from the run."""

        stream_datum_previous_indices_difference = 0
        data_keys_received = set()
        descriptor_doc = None
        external_data_keys = None

        if message_stream_name:
            descriptor_doc = self._descriptors[message_stream_name].descriptor_doc
            external_data_keys = self.get_external_data_keys(descriptor_doc["data_keys"])

        for name, doc in asset_docs:
            if name == DocumentNames.resource.value:
                resource_doc = cast(Resource, doc)
                if self._run_start_uid is None:
                    raise RuntimeError("Run start uid was never set before Resource document was emitted.")
                resource_doc["run_start"] = self._run_start_uid
            elif name == DocumentNames.stream_resource.value:
                stream_resource_doc = cast(StreamResource, doc)
                if self._run_start_uid is None:
                    raise RuntimeError("Run start uid was never set before StreamResource document was emitted.")
                stream_resource_doc["run_start"] = self._run_start_uid

                if stream_resource_doc["uid"] in self._stream_resource_data_keys:
                    raise RuntimeError(f"Received `stream_resource` with uid {stream_resource_doc['uid']} twice.")

                self._stream_resource_data_keys[stream_resource_doc["uid"]] = stream_resource_doc["data_key"]

                if not external_data_keys or stream_resource_doc["data_key"] not in external_data_keys:
                    raise RuntimeError(
                        f"Receieved a `stream_resource` with data_key {stream_resource_doc['data_key']} "
                        f"that is not in the descriptor 'STREAM:' data_keys {external_data_keys}"
                    )

            elif name == DocumentNames.stream_datum.value:
                stream_datum_doc = cast(StreamDatum, doc)
                if stream_datum_doc["descriptor"]:
                    raise RuntimeError(
                        f"Received a `stream_datum` {stream_datum_doc['uid']} with a `descriptor` uid already "
                        f"filled in, with the value {stream_datum_doc['descriptor']} this should be an empty "
                        f"string."
                    )
                if not descriptor_doc:
                    raise RuntimeError(f"`descriptor` not made for stream {message_stream_name}.")
                data_keys_received.add(self._stream_resource_data_keys[stream_datum_doc["stream_resource"]])

                stream_datum_doc["descriptor"] = descriptor_doc["uid"]
                if message_stream_name is None:
                    raise RuntimeError(
                        "`message_stream_name` was never set before StreamDatum document was emitted."
                    )
                stream_datum_previous_indices_difference = await self._pack_seq_nums_into_stream_datum(
                    stream_datum_doc,
                    message_stream_name,
                    stream_datum_previous_indices_difference,
                )
            elif name == DocumentNames.datum.value:
                ...
            else:
                raise RuntimeError(
                    f"Tried to emit an external asset {name}, acceptable external assets are "
                    "`resource`, `stream_resource`, `datum`, or `stream_datum`"
                )

            await self.emit(DocumentNames(name), doc)

            doc_logger.debug(
                "[%s] document emitted %r",
                name,
                doc,
                extra={"doc_name": name, "run_uid": self._run_start_uid, "doc": doc},
            )

        # Check we have a stream_datum for each external data_key in the descriptor
        if descriptor_doc and data_keys_received and set(external_data_keys) != data_keys_received:  # type: ignore
            raise RuntimeError(
                f"Received `stream_datum` for each of the data_keys {data_keys_received}, "  # type: ignore
                f"expected `stream_datum` for each of the data_keys {set(external_data_keys)}."  # type: ignore
            )

        return stream_datum_previous_indices_difference

    def get_external_data_keys(self, data_keys: dict[str, DataKey]) -> list[str]:
        """Get the external data keys from the descriptor data_keys dictionary"""
        return [x for x in data_keys if ("external" in data_keys[x] and data_keys[x]["external"] == "STREAM:")]

    async def _collect_events(
        self,
        collect_obj: EventCollectable,
        local_descriptors,
        return_payload: bool,
        message_stream_name: str | None,
    ):
        payload = []
        pages: dict[frozenset[str], list[Event]] = defaultdict(list)

        if message_stream_name:
            compose_event = self._descriptors[message_stream_name].compose_event
            data_keys = self._descriptors[message_stream_name].descriptor_doc["data_keys"]
            objs_read = frozenset(data_keys.keys())

        async for partial_event in iterate_maybe_async(collect_obj.collect()):
            if return_payload:
                payload.append(partial_event)

            if not message_stream_name:
                objs_read = frozenset(partial_event["data"])
                compose_event = local_descriptors[objs_read].compose_event
                data_keys = local_descriptors[objs_read].descriptor_doc["data_keys"]
                assert frozenset(data_keys.keys()) == objs_read

            if [x for x in self.get_external_data_keys(data_keys) if x in partial_event["data"]]:
                raise RuntimeError("Received an event containing data for external data keys.")

            # is there a way to generalise the keys?
            if "filled" in partial_event.keys():
                event = compose_event(
                    data=partial_event["data"],
                    timestamps=partial_event["timestamps"],
                    filled=partial_event["filled"],
                )
            else:
                event = compose_event(data=partial_event["data"], timestamps=partial_event["timestamps"])

            pages[objs_read].append(event)

        for event_list in pages.values():
            await self.emit(DocumentNames.event_page, pack_event_page(*event_list))
            doc_logger.debug(
                "[event_page] document is emitted for descriptors (run_uid=%r)",
                self._run_start_uid,
                extra={"doc_name": "event_page", "run_uid": self._run_start_uid},
            )
        return payload

    async def _collect_event_pages(
        self,
        collect_obj: EventPageCollectable,
        local_descriptors,
        return_payload: bool,
        message_stream_name: str | None,
    ):
        payload = []

        if message_stream_name:
            compose_event_page = self._descriptors[message_stream_name].compose_event_page
            data_keys = self._descriptors[message_stream_name].descriptor_doc["data_keys"]

        async for ev_page in iterate_maybe_async(collect_obj.collect_pages()):
            if return_payload:
                payload.append(ev_page)

            if not message_stream_name:
                objs_read = frozenset(ev_page["data"])
                compose_event_page = local_descriptors[objs_read].compose_event_page
                data_keys = local_descriptors[objs_read].descriptor_doc["data_keys"]

            if [x for x in self.get_external_data_keys(data_keys) if x in ev_page["data"]]:
                raise RuntimeError("Received an event_page containing data for external data keys.")

            ev_page = compose_event_page(data=ev_page["data"], timestamps=ev_page["timestamps"])
            doc_logger.debug(
                "[event_page] document is emitted with data keys %r (run_uid=%r)",
                ev_page["data"].keys(),
                ev_page["uid"],
                extra={
                    "doc_name": "event_page",
                    "run_uid": self._run_start_uid,
                    "data_keys": ev_page["data"].keys(),
                },
            )

            await self.emit(DocumentNames.event_page, ev_page)
        return payload

    async def collect(self, msg: Msg):
        """
        Collect data cached by a flyer and emit documents.

        Expect message object is
            Msg('collect',  collect_obj,  collect_obj_2, ..., stream=True,
                return_payload=True, name='stream_name')

        Where there must be at least one collect object. If multiple are used
        they must obey the WritesStreamAssets protocol.
        """
        stream_name: str | None = None

        if not self.run_is_open:
            # sanity check -- 'kickoff' should catch this and make this
            # code path impossible
            raise IllegalMessageSequence("A 'collect' message was sent but no run is open.")

        # If stream is True, run 'event' subscription per document.
        # If stream is False, run 'event_page' subscription once.
        # Stream is True is no longer supported
        stream = msg.kwargs.get("stream", False)
        if stream is True:
            raise RuntimeError(
                "Collect now emits EventPages (stream=False), "
                "so emitting Events (stream=True) is no longer supported"
            )

        # If True, accumulate all the Events in memory and return them at the
        # end, providing the plan access to the Events. If False, do not
        # accumulate, and return None.
        return_payload = msg.kwargs.get("return_payload", True)

        # Get a list of the collectable objects from the message obj and args
        collect_objects = [check_supports(obj, Collectable) for obj in (msg.obj,) + msg.args]

        # Get references to get_index methods if we have more than one collect object
        # raise error if collect_objects don't obey WritesStreamAssests protocol
        indices: list[Callable[[], SyncOrAsync[int]]] = []
        if len(collect_objects) > 1:
            indices = [check_supports(obj, WritesStreamAssets).get_index for obj in collect_objects]

        # Warn for page collectable support
        for obj in collect_objects:
            if isinstance(obj, EventCollectable) and isinstance(obj, EventPageCollectable):
                doc_logger.warn(
                    "collect() was called for a device %r which is both EventCollectable "
                    "and EventPageCollectable. Using device.collect_pages().",
                    obj.name,
                )
            self._uncollected.discard(obj)

        # Get the provided message stream name for singly nested scans
        message_stream_name: str | None = msg.kwargs.get("name", None)

        # Retrive the stream names from pre-declared streams
        declared_stream_names = self._declared_stream_names.get(frozenset(collect_objects), [])

        # If a stream name was provided in the message, check the stream has been declared
        # If one was not provided, but a single stream has been declared, then use that stream.
        if message_stream_name:
            assert message_stream_name in declared_stream_names, (
                "If a message stream name is provided declare stream needs to be called first."
            )
            stream_name = message_stream_name
        elif declared_stream_names:
            assert len(frozenset(declared_stream_names)) == 1  # Allow duplicate declarations
            stream_name = declared_stream_names[0]

        # If there is not a stream then we should be using an old-style doubly nested
        # and we need to describe_collect and prepare the nested streams.
        if not stream_name:
            if frozenset(collect_objects) not in self._local_descriptors or (
                collect_objects[0] not in self._local_descriptors
            ):
                if len(collect_objects) > 1:
                    raise IllegalMessageSequence(
                        "If collecting multiple objects you must predeclare a stream for all "
                        "the objects first and provide the stream name"
                    )
                else:
                    await self._describe_collect(collect_objects[0])

        # Get the indicies from the collect objects
        coros = [maybe_await(get_index()) for get_index in indices]
        if coros:
            # There is more than one collect object, so collect up to a minimum index
            min_index = min(await asyncio.gather(*coros))
        else:
            # There is only one collect object, so don't pass an index down
            min_index = None

        collected_asset_docs = [
            x
            for obj in collect_objects
            async for x in maybe_collect_asset_docs(
                msg,
                obj,
                index=min_index,
            )
        ]

        indices_difference = await self._pack_external_assets(
            collected_asset_docs, message_stream_name=stream_name
        )

        # Make event pages for an object which is EventCollectable or EventPageCollectable
        # objects that are EventCollectable will now group the Events and Emit an Event Page
        if len(collect_objects) == 1 and not isinstance(collect_objects[0], WritesStreamAssets):
            local_descriptors: dict[frozenset[str], ComposeDescriptorBundle] = {}
            collect_obj = collect_objects[0]

            # If the single collect object is singly nested, gather descriptors
            if collect_obj not in self._local_descriptors:
                if stream_name is None:
                    raise RuntimeError(
                        "Expected stream_name to be set before calling collect() on a singly nested device."
                    )
                objs = self._descriptor_objs[stream_name]
                data_keys = objs[collect_obj]
                local_descriptors[frozenset(data_keys)] = self._descriptors[stream_name]
                self._local_descriptors[collect_obj] = local_descriptors

            local_descriptors = self._local_descriptors[collect_obj]

            if isinstance(collect_obj, EventPageCollectable):
                payload = await self._collect_event_pages(
                    collect_obj, local_descriptors, return_payload, stream_name
                )
                # TODO: check that event pages have same length as indices_difference
            elif isinstance(collect_obj, EventCollectable):
                payload = await self._collect_events(collect_obj, local_descriptors, return_payload, stream_name)
                # TODO: check that events have same length as indices_difference
            else:
                return_payload = False
                if not stream_name:
                    raise RuntimeError(
                        "A `collect` message on a device that isn't EventCollectable or EventPageCollectable "
                        "requires a `name=stream_name` argument"
                    )
                # Since there are no events or event_pages incrementing the sequence counter, we do it ourselves.
                self._sequence_counters[stream_name] += indices_difference

            if return_payload:
                return payload

        else:
            # Since there are no events or event_pages incrementing the sequence counter, we do it ourselves.
            self._sequence_counters[stream_name] += indices_difference

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
        await self._cache_read_config(obj)
        # Invalidate any event descriptors that include this object.
        # New event descriptors, with this new configuration, will
        # be created for any future event documents.
        for name in list(self._descriptors):
            obj_set = self._descriptor_objs[name]
            if obj in obj_set:
                del self._descriptors[name]
                await self._prepare_stream(name, obj_set)
                continue
