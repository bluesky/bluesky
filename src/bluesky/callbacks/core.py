"""
Useful callbacks for the Run Engine
"""

import collections
import copy
import logging
import os
import textwrap
import time as ttime
import warnings
from collections import OrderedDict, deque, namedtuple
from datetime import datetime
from functools import partial as _partial
from functools import wraps as _wraps
from itertools import count
from typing import Dict, List, Optional

from event_model import DocumentRouter
from event_model.documents import Event, EventDescriptor, StreamDatum, StreamResource

from ..consolidators import ConsolidatorBase, consolidator_factory
from ..utils import ensure_uid

MIMETYPE_LOOKUP = {
    "hdf5": "application/x-hdf5",
    "AD_HDF5_SWMR_STREAM": "application/x-hdf5",
    "AD_HDF5_SWMR_SLICE": "application/x-hdf5",
    "AD_TIFF": "multipart/related;type=image/tiff",
    "AD_HDF5_GERM": "application/x-hdf5",
}

logger = logging.getLogger(__name__)


def make_callback_safe(func=None, *, logger=None):
    """
    If the wrapped func raises any exceptions, log them but continue.

    This is intended to ensure that any failures in non-critical callbacks do
    not interrupt data acquisition. It should *not* be applied to any
    critical callbacks, such as ones that perform data-saving, but is well
    suited to callbacks that perform non-critical streaming visualization or
    data processing.

    To debug the issue causing a failure, it can be convenient to turn this
    off and let the failures raise. To do this, set the environment variable
    ``BLUESKY_DEBUG_CALLBACKS=1``.

    Parameters
    ----------
    func: callable
    logger: logging.Logger, optional

    Examples
    --------

    Decorate a callback to make sure it will not interrupt data acquisition if
    it fails.

    >>> @make_callback_safe
    ... def callback(name, doc):
    ...     ...
    """
    if func is None:
        return _partial(make_callback_safe, logger=logger)

    @_wraps(func)
    def inner(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            debug_mode = os.environ.get("BLUESKY_DEBUG_CALLBACKS", False)
            if logger is not None:
                if debug_mode:
                    msg = f"Exception in {func}"
                else:
                    msg = (
                        f"An exception raised in the callback {func} "
                        "is being suppressed to not interrupt plan "
                        "execution.  To investigate try setting the "
                        "BLUESKY_DEBUG_CALLBACKS env to '1'"
                    )
                logger.exception(msg)
            if debug_mode:
                raise

    return inner


def make_class_safe(cls=None, *, to_wrap=None, logger=None):
    """
    If the wrapped func raises any exceptions, log them but continue.

    This is intended to ensure that any failures in non-critical callbacks do
    not interrupt data acquisition. It should *not* be applied to any
    critical callbacks, such as ones that perform data-saving, but is well
    suited to callbacks that perform non-critical streaming visualization or
    data processing.

    To debug the issue causing a failure, it can be convenient to turn this
    off and let the failures raise. To do this, set the environment variable
    ``BLUESKY_DEBUG_CALLBACKS=1``.

    Parameters
    ----------
    cls: callable
    to_wrap: List[String], optional
        Names of methods of cls to wrap. Default is ``['__call__']``.
    logger: logging.Logger, optional

    Examples
    --------

    Decorate a class to make sure it will not interrupt data acquisition if
    it fails.

    >>> @make_class_safe
    ... class Callback(event_model.DocumentRouter):
    ...     ...
    """

    if cls is None:
        return _partial(make_class_safe, to_wrap=to_wrap, logger=logger)

    if to_wrap is None:
        to_wrap = ["__call__"]

    for f_name in to_wrap:
        setattr(cls, f_name, make_callback_safe(getattr(cls, f_name), logger=logger))

    return cls


class CallbackBase(DocumentRouter):
    log = None


class CallbackCounter:
    "As simple as it sounds: count how many times a callback is called."

    # Wrap itertools.count in something we can use as a callback.
    def __init__(self):
        self.counter = count()
        self(None, {})  # Pass a fake doc to prime the counter (start at 1).

    def __call__(self, name, doc):
        self.value = next(self.counter)


def print_metadata(name, doc):
    "Print all fields except uid and time."
    for field, value in sorted(doc.items()):
        # uid is returned by the RunEngine, and time is self-evident
        if field not in ["time", "uid"]:
            print(f"{field}: {value}")


def collector(field, output):
    """
    Build a function that appends data to a list.

    This is useful for testing but not advised for general use. (There is
    probably a better way to do whatever you want to do!)

    Parameters
    ----------
    field : str
        the name of a data field in an Event
    output : mutable iterable
        such as a list

    Returns
    -------
    func : function
        expects one argument, an Event dictionary
    """

    def f(name, event):
        output.append(event["data"][field])

    return f


def format_num(x, max_len=11, pre=5, post=5):
    if (abs(x) > 10**pre or abs(x) < 10**-post) and x != 0:
        x = f"%.{post}e" % x
    else:
        x = f"%{pre}.{post}f" % x

    return x


def get_obj_fields(fields):
    """
    If fields includes any objects, get their field names using obj.describe()

    ['det1', det_obj] -> ['det1, 'det_obj_field1, 'det_obj_field2']"
    """
    string_fields = []
    for field in fields:
        if isinstance(field, str):
            string_fields.append(field)
        else:
            try:
                field_list = sorted(field.describe().keys())
            except AttributeError:
                raise ValueError(  # noqa: B904
                    "Fields must be strings or objects with a 'describe' method that return a dict."
                )
            string_fields.extend(field_list)
    return string_fields


class CollectThenCompute(CallbackBase):
    def __init__(self):
        self._start_doc = None
        self._stop_doc = None
        self._events = deque()
        self._descriptors = deque()

    def start(self, doc):
        self._start_doc = doc
        super().start(doc)

    def descriptor(self, doc):
        self._descriptors.append(doc)
        super().descriptor(doc)

    def event(self, doc):
        self._events.append(doc)
        super().event(doc)

    def stop(self, doc):
        self._stop_doc = doc
        self.compute()
        super().stop(doc)

    def reset(self):
        self._start_doc = None
        self._stop_doc = None
        self._events.clear()
        self._descriptors.clear()

    def compute(self):
        raise NotImplementedError("This method must be defined by a subclass.")


@make_class_safe(logger=logger)
class CollectLiveStream(CallbackBase):
    """A callback to handle (consolidate) data received in StreamDocuments

    It uses the following logic of handling a stream of Bluesky documents:

    1. Recieve a Descriptor document and cache it in the `_desc_docs` dictionary under the `descriptor_uid` key
    2. Receive a StreamResource document and cache it in the `_sres_docs` under the `stream_resource_uid` key.
       Determine the `data_key` to which this StreamResource corresponds and record it in the
       `_data_key_to_sres_uid` dictionary.
    3. When the first StreamDatum for a certain data_key is received, create a Consolidator object for it, keep
       it in the `stream_consolidators` dictionary under the corresponding `stream_resource_uid` key, and use
       the consolidators methods to consume the StreamDatum.
    4. For any subsequent StreamDatums, retrieve an existing Consolidator from `stream_consolidators`.

    """

    def __init__(self):
        self._start_doc = None
        self._stop_doc = None
        self._event_docs: Dict[str, Event] = {}  # Keyed by Event uid
        self._desc_docs: Dict[str, EventDescriptor] = {}  # Keyed by Descriptor uid
        self._sres_docs: Dict[str, StreamResource] = {}  # Keyed by Stream Resource uid
        self._data_key_to_sres_uid: Dict[str, List[str]] = collections.defaultdict(list)
        self.events = deque()
        self.stream_consolidators: Dict[str, ConsolidatorBase] = {}  # Keyed by Stream Resource uid

    def _ensure_resource_backcompat(self, doc: StreamResource) -> StreamResource:
        """Kept for back-compatibility with old StreamResource schema from event_model<1.20.0

        Will make changes to and return a shallow copy of StreamResource dictionary adhering to the new structure.
        """

        doc = copy.copy(doc)
        if ("mimetype" not in doc.keys()) and ("spec" not in doc.keys()):
            raise RuntimeError("StreamResource document is missing a mimetype or spec")
        else:
            doc["mimetype"] = doc.get("mimetype") or MIMETYPE_LOOKUP[doc.get("spec")]
        if "parameters" not in doc.keys():
            doc["parameters"] = doc.pop("resource_kwargs", {})
        if "uri" not in doc.keys():
            file_path = doc.pop("root").strip("/") + "/" + doc.pop("resource_path").strip("/")
            doc["uri"] = "file://localhost/" + file_path

        return doc

    def start(self, doc):
        self._start_doc = doc
        super().start(doc)

    def descriptor(self, doc: EventDescriptor):
        self._desc_docs[doc["uid"]] = doc
        super().descriptor(doc)

    def event(self, doc):
        # self._event_docs.append(doc)
        event_dict = {"seq_num": doc["seq_num"], "descriptor": doc["descriptor"]}
        event_dict.update(doc["data"])
        event_dict.update({f"ts_{c}": v for c, v in doc["timestamps"].items()})
        self.events.append(event_dict)

        super().event(doc)

    def stream_resource(self, doc: StreamResource):
        self._sres_docs[doc["uid"]] = self._ensure_resource_backcompat(doc)
        self._data_key_to_sres_uid[doc["data_key"]].append(doc["uid"])  # Multiple Streams Resources per data_key

    def _get_or_create_handler(self, sres_uid: str, desc_uid: Optional[str] = None):
        """Get a Handler / Stream Consolidator, if it already exists, or register it from a SR document"""

        if sres_uid in self.stream_consolidators.keys():
            handler = self.stream_consolidators[sres_uid]

        elif sres_uid in self._sres_docs.keys():
            # Initialise a new Bluesky Consolidator for the StreamResource
            sres_doc = self._sres_docs[sres_uid]
            if not desc_uid:
                warnings.warn(
                    textwrap.dedent(f"""Initialising a new Stream Consolidator without specified Descriptor uid.
                    Attempting to find a matching Descriptor document by the data_key: {sres_doc["data_key"]}."""),
                    stacklevel=2,
                )
                desc_found = False
                for desc_doc in self._desc_docs.values():
                    if sres_doc["data_key"] in desc_doc["data_keys"].keys():
                        desc_found = True
                        break
                if not desc_found:
                    raise RuntimeError(
                        "Unable to initialise a new Stream Consolidator without specified Descriptor uid."
                    )
            else:
                desc_doc = self._desc_docs[desc_uid]

            handler = consolidator_factory(sres_doc, desc_doc)
            self.stream_consolidators[sres_uid] = handler
        else:
            raise RuntimeError(f"Stream Resource {sres_uid} is referenced before being declared.")

        return handler

    def stream_datum(self, doc: StreamDatum):
        """Get the Stream Resource node and the associated handler (consolidator)"""
        handler = self._get_or_create_handler(doc["stream_resource"], desc_uid=doc["descriptor"])
        handler.consume_stream_datum(doc)

    def stop(self, doc):
        self._stop_doc = doc
        super().stop(doc)

    def get_handler(self, data_key):
        """Return a handler/consolidator for a data_key"""
        if data_key in self._data_key_to_sres_uid.keys():
            sres_uids = self._data_key_to_sres_uid[data_key]
            if len(sres_uids) > 0:
                # NOTE: If several StreamResources have the same data_key, returns handler for only the last Stream
                if sres_uids[-1] in self.stream_consolidators.keys():
                    handler = self.stream_consolidators[sres_uids[-1]]
                    return handler
        else:
            raise RuntimeError(f"No data received for {data_key}.")

        return None

    def get_adapter(self, data_key):
        return self.get_handler(data_key).get_adapter()


@make_class_safe(logger=logger)
class LiveTable(CallbackBase):
    """Live updating table

    Parameters
    ----------
    fields : list
         List of fields to add to the table.

    stream_name : str, optional
         The event stream to watch for

    print_header_interval : int, optional
         Reprint the header every this many lines, defaults to 50

    min_width : int, optional
         The minimum width is spaces of the data columns.  Defaults to 12

    default_prec : int, optional
         Precision to use if it can not be found in descriptor, defaults to 3

    extra_pad : int, optional
         Number of extra spaces to put around the printed data, defaults to 1

    separator_lines : bool, optional
        Add empty lines before and after the printed table, default True

    logbook : callable, optional
        Must take a sting as the first positional argument

           def logbook(input_str):
                pass

    out : callable, optional
        Function to call to 'print' a line.  Defaults to `print`
    """

    _FMTLOOKUP = {
        "s": "{pad}{{{k}: >{width}.{prec}{dtype}}}{pad}",
        "f": "{pad}{{{k}: >{width}.{prec}{dtype}}}{pad}",
        "g": "{pad}{{{k}: >{width}.{prec}{dtype}}}{pad}",
        "d": "{pad}{{{k}: >{width}{dtype}}}{pad}",
    }
    _FMT_MAP = {
        "number": "f",
        "integer": "d",
        "string": "s",
    }
    _fm_sty = namedtuple("fm_sty", ["width", "prec", "dtype"])  # type: ignore
    water_mark = "{st[plan_type]} {st[plan_name]} ['{st[uid]:.8s}'] (scan num: {st[scan_id]})"
    ev_time_key = "SUPERLONG_EV_TIMEKEY_THAT_I_REALLY_HOPE_NEVER_CLASHES"

    def __init__(
        self,
        fields,
        *,
        stream_name="primary",
        print_header_interval=50,
        min_width=12,
        default_prec=3,
        extra_pad=1,
        separator_lines=True,
        logbook=None,
        out=print,
    ):
        super().__init__()
        self._header_interval = print_header_interval
        # expand objects
        self._fields = get_obj_fields(fields)
        self._stream = stream_name
        self._start = None
        self._stop = None
        self._descriptors = set()
        self._pad_len = extra_pad
        self._extra_pad = " " * extra_pad
        self._min_width = min_width
        self._default_prec = default_prec
        self._separator_lines = separator_lines
        self._format_info = OrderedDict(
            [
                ("seq_num", self._fm_sty(10 + self._pad_len, "", "d")),
                (self.ev_time_key, self._fm_sty(10 + 2 * extra_pad, 10, "s")),
            ]
        )
        self._rows = []
        self.logbook = logbook
        self._sep_format = None
        self._out = out

    def descriptor(self, doc):
        def patch_up_precision(p):
            try:
                return int(p)
            except (TypeError, ValueError):
                return self._default_prec

        if doc["name"] != self._stream:
            return

        self._descriptors.add(doc["uid"])

        dk = doc["data_keys"]
        for k in self._fields:
            width = max(self._min_width, len(k) + 2, self._default_prec + 1 + 2 * self._pad_len)
            try:
                dk_entry = dk[k]
            except KeyError:
                # this descriptor does not know about this key
                continue

            if dk_entry["dtype"] not in self._FMT_MAP:
                warnings.warn(  # noqa: B028
                    "The key {} will be skipped because LiveTable "
                    "does not know how to display the dtype {}"
                    "".format(k, dk_entry["dtype"])
                )
                continue

            prec = patch_up_precision(dk_entry.get("precision", self._default_prec))
            fmt = self._fm_sty(width=width, prec=prec, dtype=self._FMT_MAP[dk_entry["dtype"]])

            self._format_info[k] = fmt

        self._sep_format = "+" + "+".join("-" * f.width for f in self._format_info.values()) + "+"
        self._main_fmnt = "|".join(
            "{{: >{w}}}{pad}".format(w=f.width - self._pad_len, pad=" " * self._pad_len)
            for f in self._format_info.values()
        )
        headings = [k if k != self.ev_time_key else "time" for k in self._format_info]
        self._header = "|" + self._main_fmnt.format(*headings) + "|"
        self._data_formats = OrderedDict(
            (
                k,
                self._FMTLOOKUP[f.dtype].format(
                    k=f"h{str(hash(k))}",
                    width=f.width - 2 * self._pad_len,
                    prec=f.prec,
                    dtype=f.dtype,
                    pad=self._extra_pad,
                ),
            )
            for k, f in self._format_info.items()
        )

        self._count = 0

        if self._separator_lines:
            self._print("\n")
        self._print(self._sep_format)
        self._print(self._header)
        self._print(self._sep_format)
        super().descriptor(doc)

    def event(self, doc):
        try:
            if ensure_uid(doc["descriptor"]) not in self._descriptors:
                return
            # shallow copy so we can mutate
            data = dict(doc["data"])
            self._count += 1
            if not self._count % self._header_interval:
                self._print(self._sep_format)
                self._print(self._header)
                self._print(self._sep_format)
            fmt_time = str(datetime.fromtimestamp(doc["time"]).time())
            data[self.ev_time_key] = fmt_time
            data["seq_num"] = doc["seq_num"]
            cols = [
                f.format(**{f"h{str(hash(k))}": data[k]})
                # Show data[k] if k exists in this Event and is 'filled'.
                # (The latter is only applicable if the data is
                # externally-stored -- hence the fallback to `True`.)
                if ((k in data) and doc.get("filled", {}).get(k, True))
                # Otherwise use a placeholder of whitespace.
                else " " * self._format_info[k].width
                for k, f in self._data_formats.items()
            ]
            self._print("|" + "|".join(cols) + "|")
        except Exception as ex:
            if self.log is not None:
                self.log.exception(ex)
            self._print(f"{{k:*^{self._min_width}}}".format(k=" failed to format row "))
            if os.environ.get("BLUESKY_DEBUG_CALLBACKS", False):
                raise ex
        super().event(doc)

    def stop(self, doc):
        if ensure_uid(doc["run_start"]) != self._start["uid"]:
            return

        # This sleep is just cosmetic. It improves the odds that the bottom
        # border is not printed until all the rows from events are printed,
        # avoiding this ugly scenario:
        #
        # |         4 | 22:08:56.7 |      0.000 |
        # +-----------+------------+------------+
        # generator scan ['6d3f71'] (scan num: 1)
        # Out[2]: |         5 | 22:08:56.8 |      0.000 |
        ttime.sleep(0.1)

        if self._sep_format is not None:
            self._print(self._sep_format)
        self._stop = doc

        wm = self.water_mark.format(st=self._start)
        self._out(wm)
        if self.logbook:
            self.logbook("\n".join([wm] + self._rows))
        if self._separator_lines:
            self._print("\n")
        super().stop(doc)

    def start(self, doc):
        self._rows = []
        self._start = doc
        self._stop = None
        self._sep_format = None
        super().start(doc)

    def _print(self, out_str):
        self._rows.append(out_str)
        self._out(out_str)
