from collections import deque
from itertools import count, tee
import time as ttime
from event_model import DocumentNames
from .utils import (
    IllegalBundlingSequence,
    IllegalMessageSequence,
    IllegalMonitorSequence,
    Msg,
    _rearrange_into_parallel_dicts,
    new_uid,
    short_uid,
)


class RunBundler:
    def __init__(self, md, record_interruptions, emit, emit_sync, log, *, loop):
        # state stolen from the RE
        self.bundling = False  # if we are in the middle of bundling readings
        self._bundle_name = None  # name given to event descriptor
        self._run_start_uid = None  # The (future) runstart uid
        self._objs_read = deque()  # objects read in one Event
        self._read_cache = deque()  # cache of obj.read() in one Event
        self._asset_docs_cache = deque()  # cache of obj.collect_asset_docs()
        self._describe_cache = dict()  # cache of all obj.describe() output
        self._config_desc_cache = dict()  # " obj.describe_configuration()
        self._config_values_cache = dict()  # " obj.read_configuration() values
        self._config_ts_cache = dict()  # " obj.read_configuration() timestamps
        self._descriptors = dict()  # cache of {name: (objs_frozen_set, doc)}
        self._sequence_counters = dict()  # a seq_num counter per stream
        self._teed_sequence_counters = dict()  # for if we redo data-points
        self._monitor_params = dict()  # cache of {obj: (cb, kwargs)}
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

        self.loop = loop

    async def open_run(self, msg):
        self.run_is_open = True
        self._run_start_uid = new_uid()
        self._interruptions_desc_uid = None  # uid for a special Event Desc.
        self._interruptions_counter = count(1)  # seq_num, special Event stream

        doc = dict(uid=self._run_start_uid, time=ttime.time(), **self._md)
        await self.emit(DocumentNames.start, doc)
        self.log.debug("Emitted RunStart (uid=%r)", doc["uid"])
        await self.reset_checkpoint_state_coro()

        # Emit an Event Descriptor for recording any interruptions as Events.
        if self.record_interruptions:
            self._interruptions_desc_uid = new_uid()
            dk = {"dtype": "string", "shape": [], "source": "RunEngine"}
            interruptions_desc = dict(
                time=ttime.time(),
                uid=self._interruptions_desc_uid,
                name="interruptions",
                data_keys={"interruption": dk},
                run_start=self._run_start_uid,
            )
            await self.emit(DocumentNames.descriptor, interruptions_desc)

        return self._run_start_uid

    async def close_run(self, msg):

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
        # Count the number of Events in each stream.
        num_events = {}
        for bundle_name, counter in self._sequence_counters.items():
            if bundle_name is None:
                # rare but possible via Msg('create', name='primary')
                continue
            num_events[bundle_name] = next(counter) - 1
        reason = msg.kwargs.get("reason", None)
        if reason is None:
            reason = ""
        exit_status = msg.kwargs.get("exit_status", "success") or "success"

        doc = dict(
            run_start=self._run_start_uid,
            time=ttime.time(),
            uid=new_uid(),
            exit_status=exit_status,
            reason=reason,
            num_events=num_events,
        )
        await self.emit(DocumentNames.stop, doc)
        self.log.debug("Emitted RunStop (uid=%r)", doc["uid"])
        await self.reset_checkpoint_state_coro()
        self.run_is_open = False
        return doc["run_start"]

    async def create(self, msg):
        if self.bundling:
            raise IllegalBundlingSequence(msg, "A second 'create' message is "
                                          "not allowed until the current event"
                                          " bundle is closed with a 'save' or "
                                          "'drop' message.")

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

    async def read(self, msg, reading):
        if self.bundling:
            obj = msg.obj
            # if the object is not in the _describe_cache, cache it
            if obj not in self._describe_cache:
                # Validate that there is no data key name collision.
                data_keys = obj.describe()
                self._describe_cache[obj] = data_keys
                self._config_desc_cache[obj] = obj.describe_configuration()
                self._cache_config(obj)

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
            if hasattr(obj, "collect_asset_docs"):
                self._asset_docs_cache.extend(
                    obj.collect_asset_docs(*msg.args, **msg.kwargs)
                )

        return reading

    def _cache_config(self, obj):
        "Read the object's configuration and cache it."
        config_values = {}
        config_ts = {}
        for key, val in obj.read_configuration().items():
            config_values[key] = val["value"]
            config_ts[key] = val["timestamp"]
        self._config_values_cache[obj] = config_values
        self._config_ts_cache[obj] = config_ts

    async def monitor(self, msg):
        obj = msg.obj
        if msg.args:
            raise ValueError(
                "The 'monitor' Msg does not accept positional " "arguments."
            )
        kwargs = dict(msg.kwargs)
        name = kwargs.pop("name", short_uid("monitor"))
        if obj in self._monitor_params:
            raise IllegalMonitorSequence(msg, "A 'monitor' message was sent "
                                         "for {} which is already "
                                         "monitored".format(obj))

        descriptor_uid = new_uid()
        data_keys = obj.describe()
        config = {obj.name: {"data": {}, "timestamps": {}}}
        config[obj.name]["data_keys"] = obj.describe_configuration()
        for key, val in obj.read_configuration().items():
            config[obj.name]["data"][key] = val["value"]
            config[obj.name]["timestamps"][key] = val["timestamp"]
        object_keys = {obj.name: list(data_keys)}
        hints = {}
        if hasattr(obj, "hints"):
            hints.update({obj.name: obj.hints})
        desc_doc = dict(
            run_start=self._run_start_uid,
            time=ttime.time(),
            data_keys=data_keys,
            uid=descriptor_uid,
            configuration=config,
            hints=hints,
            name=name,
            object_keys=object_keys,
        )
        self.log.debug(
            "Emitted Event Descriptor with name %r containing " "data keys %r (uid=%r)",
            name,
            data_keys.keys(),
            descriptor_uid,
        )
        seq_num_counter = count(1)

        def emit_event(*args, **kwargs):
            # Ignore the inputs. Use this call as a signal to call read on the
            # object, a crude way to be sure we get all the info we need.
            data, timestamps = _rearrange_into_parallel_dicts(obj.read())
            doc = dict(
                descriptor=descriptor_uid,
                time=ttime.time(),
                data=data,
                timestamps=timestamps,
                seq_num=next(seq_num_counter),
                uid=new_uid(),
            )
            self.emit_sync(DocumentNames.event, doc)

        self._monitor_params[obj] = emit_event, kwargs
        await self.emit(DocumentNames.descriptor, desc_doc)
        obj.subscribe(emit_event, **kwargs)

    def record_interruption(self, content):
        """
        Emit an event in the 'interruptions' event stream.

        If we are not inside a run or if self.record_interruptions is False,
        nothing is done.
        """
        if self._interruptions_desc_uid is not None:
            # We are inside a run and self.record_interruptions is True.
            doc = dict(
                descriptor=self._interruptions_desc_uid,
                time=ttime.time(),
                uid=new_uid(),
                seq_num=next(self._interruptions_counter),
                data={"interruption": content},
                timestamps={"interruption": ttime.time()},
            )
            self.emit_sync(DocumentNames.event, doc)

    def rewind(self):
        self._sequence_counters.clear()
        self._sequence_counters.update(self._teed_sequence_counters)
        # This is needed to 'cancel' an open bundling (e.g. create) if
        # the pause happens after a 'checkpoint', after a 'create', but
        # before the paired 'save'.
        self.bundling = False

    async def unmonitor(self, msg):

        obj = msg.obj
        if obj not in self._monitor_params:
            raise IllegalMonitorSequence(msg,
                                         f"Cannot 'unmonitor' {obj}; it is "
                                         "not being monitored.")

        cb, kwargs = self._monitor_params[obj]
        obj.clear_sub(cb)
        del self._monitor_params[obj]
        await self.reset_checkpoint_state_coro()

    async def save(self, msg):
        if not self.bundling:
            raise IllegalBundlingSequence(msg, "A 'create' message must be "
                                          "sent, to open an event bundle, "
                                          "before that bundle can be saved "
                                          "with 'save'.")

        # Short-circuit if nothing has been read. (Do not create empty Events.)
        if not self._objs_read:
            self.bundling = False
            self._bundle_name = None
            return
        # The Event Descriptor is uniquely defined by the set of objects
        # read in this Event grouping.
        objs_read = frozenset(self._objs_read)

        # Event Descriptor documents
        desc_key = self._bundle_name

        # This is a separate check because it can be reset on resume.
        seq_num_key = desc_key
        if seq_num_key not in self._sequence_counters:
            counter = count(1)
            counter_copy1, counter_copy2 = tee(counter)
            self._sequence_counters[seq_num_key] = counter_copy1
            self._teed_sequence_counters[seq_num_key] = counter_copy2
        self.bundling = False
        self._bundle_name = None

        d_objs, doc = self._descriptors.get(desc_key, (None, None))
        if d_objs is not None and d_objs != objs_read:
            raise RuntimeError(
                "Mismatched objects read, expected {!s}, "
                "got {!s}".format(d_objs, objs_read)
            )
        if doc is None:
            # We don't not have an Event Descriptor for this set.
            data_keys = {}
            config = {}
            object_keys = {}
            hints = {}
            for obj in objs_read:
                dks = self._describe_cache[obj]
                name = obj.name
                # dks is an OrderedDict. Record that order as a list.
                object_keys[obj.name] = list(dks)
                for field, dk in dks.items():
                    dk["object_name"] = name
                data_keys.update(dks)
                config[name] = {}
                config[name]["data"] = self._config_values_cache[obj]
                config[name]["timestamps"] = self._config_ts_cache[obj]
                config[name]["data_keys"] = self._config_desc_cache[obj]
                if hasattr(obj, "hints"):
                    hints[name] = obj.hints
            descriptor_uid = new_uid()
            doc = dict(
                run_start=self._run_start_uid,
                time=ttime.time(),
                data_keys=data_keys,
                uid=descriptor_uid,
                configuration=config,
                name=desc_key,
                hints=hints,
                object_keys=object_keys,
            )
            await self.emit(DocumentNames.descriptor, doc)
            self.log.debug(
                "Emitted Event Descriptor with name %r containing "
                "data keys %r (uid=%r)",
                desc_key,
                data_keys.keys(),
                descriptor_uid,
            )
            self._descriptors[desc_key] = (objs_read, doc)

        descriptor_uid = doc["uid"]

        # Resource and Datum documents
        for name, doc in self._asset_docs_cache:
            # Add a 'run_start' field to the resource document on its way out.
            if name == "resource":
                doc["run_start"] = self._run_start_uid
            await self.emit(DocumentNames(name), doc)

        # Event documents
        seq_num = next(self._sequence_counters[seq_num_key])
        event_uid = new_uid()
        # Merge list of readings into single dict.
        readings = {k: v for d in self._read_cache for k, v in d.items()}
        for key in readings:
            readings[key]["value"] = readings[key]["value"]
        data, timestamps = _rearrange_into_parallel_dicts(readings)
        # Mark all externally-stored data as not filled so that consumers
        # know that the corresponding data are identifies, not dereferenced
        # data.
        filled = {
            k: False
            for k, v in self._descriptors[desc_key][1]["data_keys"].items()
            if "external" in v
        }
        doc = dict(
            descriptor=descriptor_uid,
            time=ttime.time(),
            data=data,
            timestamps=timestamps,
            seq_num=seq_num,
            uid=event_uid,
            filled=filled,
        )
        await self.emit(DocumentNames.event, doc)
        self.log.debug(
            "Emitted Event with data keys %r (uid=%r)", data.keys(), event_uid
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
            counter_copy1, counter_copy2 = tee(counter)
            self._sequence_counters[key] = counter_copy1
            self._teed_sequence_counters[key] = counter_copy2

    async def reset_checkpoint_state_coro(self):
        self.reset_checkpoint_state()

    async def suspend_monitors(self):
        for obj, (cb, kwargs) in self._monitor_params.items():
            obj.clear_sub(cb)

    async def restore_monitors(self):
        for obj, (cb, kwargs) in self._monitor_params.items():
            obj.subscribe(cb, **kwargs)

    async def clear_checkpoint(self, msg):
        self._teed_sequence_counters.clear()

    async def drop(self, msg):
        if not self.bundling:
            raise IllegalBundlingSequence(msg, "A 'create' message must be "
                                          "sent, to open an event bundle, "
                                          "before that bundle can be dropped "
                                          "with 'drop'.")

        self.bundling = False
        self._bundle_name = None
        self.log.debug("Dropped open event bundle")

    async def kickoff(self, msg):
        self._uncollected.add(msg.obj)

    async def complete(self, msg):
        ...

    async def collect(self, msg):
        """
        Collect data cached by a flyer and emit descriptor and event documents.

        Expect message object is

            Msg('collect', obj)
            Msg('collect', obj, stream=True)
        """
        obj = msg.obj

        if not self.run_is_open:
            # sanity check -- 'kickoff' should catch this and make this
            # code path impossible
            raise IllegalMessageSequence(
                "A 'collect' message was sent but no " "run is open."
            )
        self._uncollected.discard(obj)

        if hasattr(obj, "collect_asset_docs"):
            # Resource and Datum documents
            for name, doc in obj.collect_asset_docs():
                # Add a 'run_start' field to the resource document on its way out.
                if name == "resource":
                    doc["run_start"] = self._run_start_uid
                await self.emit(DocumentNames(name), doc)

        named_data_keys = obj.describe_collect()
        # e.g., {name_for_desc1: data_keys_for_desc1,
        #        name for_desc2: data_keys_for_desc2, ...}
        bulk_data = {}
        local_descriptors = {}  # hashed on obj_read, not (name, objs_read)
        for stream_name, data_keys in named_data_keys.items():
            desc_key = stream_name
            d_objs = frozenset(data_keys)
            if desc_key not in self._descriptors:
                objs_read = d_objs
                # We don't not have an Event Descriptor for this set.
                descriptor_uid = new_uid()
                object_keys = {obj.name: list(data_keys)}
                hints = {}
                if hasattr(obj, "hints"):
                    hints.update({obj.name: obj.hints})
                doc = dict(
                    run_start=self._run_start_uid,
                    time=ttime.time(),
                    data_keys=data_keys,
                    uid=descriptor_uid,
                    name=stream_name,
                    hints=hints,
                    object_keys=object_keys,
                )
                await self.emit(DocumentNames.descriptor, doc)
                self.log.debug(
                    "Emitted Event Descriptor with name %r "
                    "containing data keys %r (uid=%r)",
                    stream_name,
                    data_keys.keys(),
                    descriptor_uid,
                )
                self._descriptors[desc_key] = (objs_read, doc)
                self._sequence_counters[desc_key] = count(1)
            else:
                objs_read, doc = self._descriptors[desc_key]
            if d_objs != objs_read:
                raise RuntimeError(
                    "Mismatched objects read, "
                    "expected {!s}, "
                    "got {!s}".format(d_objs, objs_read)
                )

            descriptor_uid = doc["uid"]
            local_descriptors[objs_read] = (stream_name, descriptor_uid)

            bulk_data[descriptor_uid] = []

        # If stream is True, run 'event' subscription per document.
        # If stream is False, run 'bulk_events' subscription once.
        stream = msg.kwargs.get("stream", False)
        for ev in obj.collect():
            objs_read = frozenset(ev["data"])
            stream_name, descriptor_uid = local_descriptors[objs_read]
            seq_num = next(self._sequence_counters[stream_name])

            event_uid = new_uid()

            reading = ev["data"]
            for key in ev["data"]:
                reading[key] = reading[key]
            ev["data"] = reading
            ev["descriptor"] = descriptor_uid
            ev["seq_num"] = seq_num
            ev["uid"] = event_uid

            if stream:
                self.log.debug(
                    "Emitted Event with data keys %r (uid=%r)",
                    ev["data"].keys(),
                    ev["uid"],
                )
                await self.emit(DocumentNames.event, ev)
            else:
                bulk_data[descriptor_uid].append(ev)

        if not stream:
            await self.emit(DocumentNames.bulk_events, bulk_data)
            self.log.debug(
                "Emitted bulk events for descriptors with uids " "%r", bulk_data.keys()
            )

    async def backstop_collect(self):
        for obj in list(self._uncollected):
            try:
                await self.collect(Msg("collect", obj))
            except Exception:
                self.log.exception("Failed to collect %r.", obj)

    async def configure(self, msg):
        obj = msg.obj
        # Invalidate any event descriptors that include this object.
        # New event descriptors, with this new configuration, will
        # be created for any future event documents.
        for name in list(self._descriptors):
            obj_set, _ = self._descriptors[name]
            if obj in obj_set:
                del self._descriptors[name]
        self._cache_config(obj)
