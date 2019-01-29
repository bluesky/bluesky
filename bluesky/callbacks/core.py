"""
Useful callbacks for the Run Engine
"""
from itertools import count
import warnings
from collections import defaultdict, deque, namedtuple, OrderedDict, ChainMap
import time as ttime
from datetime import datetime
import logging
from ..utils import ensure_uid

from event_model import DocumentRouter


class CallbackBase:
    """
    A convenient platform for writing stateful callbacks.

    Examples
    --------
    A callback that prints the total number of events as they go by.

    >>> class Thing(CallbackBase):
    ...     def __init__(self, start_doc=None):
    ...         super().__init__(start_doc)
    ...         self.counter = 0
    ...
    ...     def event(self, doc):
    ...         self.counter += 1       
    ...         print(self.counter)
    """

    def __init__(self, start_doc=None):
        # Callbacks meant to be used with a RunRouter may *require* a start_doc
        # at __init__ time by making the start_doc argument required.
        # Others may not.
        #
        # If a start_doc is given, it is passed to the `start` method
        # immediately, here. If one is not given, it is up to the caller to
        # call `start(...)` later once the RunStart document is available.
        if start_doc is not None:
            self.start(start_doc)

    def __call__(self, name, doc):
        "Dispatch to methods expecting particular doc types."
        return getattr(self, name)(doc)

    def event(self, doc):
        pass

    def bulk_events(self, doc):
        pass

    def resource(self, doc):
        pass

    def datum(self, doc):
        pass

    def bulk_datum(self, doc):
        pass

    def descriptor(self, doc):
        pass

    def start(self, doc):
        pass

    def stop(self, doc):
        pass


class RunRouter(DocumentRouter):
    """
    Routes documents, by run, to callbacks it creates from factory functions.

    A RunRouter is callable, and it has the signature ``router(name, doc)``,
    suitable for subscribing to the RunEngine.

    It is configured with a list of factory functions that produce callbacks in
    a two-layered scheme, described below.

    Parameters
    ----------
    factories : list
        A list of callables with the signature::

            factory(start_doc) -> List[Callbacks], List[SubFactories]

        which should return two lists, which may be empty. All items in the
        first list should be callbacks --- callables with the signature::

            callback(name, doc)

        that will receive all subsequent documents from the run including the
        RunStop document. All items in the second list should be "subfactories"
        with the signature::

            subfactory(descriptor_doc) -> List[Callbacks]

        These will receive each of the EventDescriptor documents for the run,
        as they arrive. They must return one list, which may be empty,
        containing callbacks that will receive all Events that reference that
        EventDescriptor and finally the RunStop document for the run.
    """
    def __init__(self, factories):
        self.factories = factories

        # Map RunStart UID to "subfactory" functions that want all
        # EventDescriptors from that run.
        self._subfactories = defaultdict(list)

        # Callbacks that want all the documents from a given run, keyed on
        # RunStart UID.
        self._factory_cbs_by_start = defaultdict(list)

        # Callbacks that want all the documents from a given run, keyed on
        # each EventDescriptor UID in the run.
        self._factory_cbs_by_descriptor = defaultdict(list)

        # Callbacks that want documents related to a given EventDescriptor,
        # keyed on EventDescriptor UID.
        self._subfactory_cbs_by_descriptor = defaultdict(list)

        # Callbacks that want documents related to a given EventDescriptor,
        # keyed on the RunStart UID referenced by that EventDescriptor.
        self._subfactory_cbs_by_start = defaultdict(list)

        # Map RunStart UID to the list EventDescriptor. This is used to
        # facilitate efficient cleanup of the caches above.
        self._descriptors = defaultdict(list)

        # Map Resource UID to RunStart UID.
        self._resources = {}

        # Old-style Resources that do not have a RunStart UID
        self._unlabeled_resources = deque(maxlen=10000)

    def start(self, doc):
        uid = doc['uid']
        for factory in self.factories:
            callbacks, subfactories = factory(doc)
            self._factory_cbs_by_start[uid].extend(callbacks)
            self._subfactories[uid].extend(subfactories)

    def descriptor(self, doc):
        uid = doc['uid']
        start_uid = doc['run_start']
        # Let all the callbacks that want all documents for this run process
        # this descriptor.
        for cb in self._factory_cbs_by_start[start_uid]:
            cb('descriptor', doc)
        # Let all the subfactories add any relavant callbacks.
        for subfactory in self._subfactories[start_uid]:
            callbacks = subfactory(doc)
            self._subfactory_cbs_by_start[start_uid].extend(callbacks)
            self._subfactory_cbs_by_descriptor[uid].extend(callbacks)
        # Keep track of the RunStart UID -> [EventDescriptor UIDs] mapping for
        # purposes of cleanup in stop().
        self._descriptors[start_uid].append(uid)

    def event_page(self, doc):
        descriptor_uid = doc['descriptor']
        for cb in self._factory_cbs_by_descriptor[descriptor_uid]:
            cb('event_page', doc)
        for cb in self._subfactory_cbs_by_descriptor[descriptor_uid]:
            cb('event_page', doc)

    def datum_page(self, doc):
        resource_uid = doc['resource']
        try:
            start_uid = self._resources[resource_uid]
        except KeyError:
            if resource_uid in self._unlabeled_resources:
                # Old Resources do not have a reference to a RunStart document,
                # so in turn we cannot immediately tell which run these datum
                # documents belong to.
                # Fan them out to every run currently flowing through RunRouter. If
                # they are not applicable they will do no harm, and this is
                # expected to be an increasingly rare case.
                for cbs in self._factory_cbs_by_start.values():
                    for cb in cbs:
                        cb('datum_page', doc)
                for cbs in self._subfactory_cbs_by_start.values():
                    for cb in cbs:
                        cb('datum_page', doc)

    def resource(self, doc):
        try:
            start_uid = doc['run_start']
        except KeyError:
            # Old Resources do not have a reference to a RunStart document.
            # Fan them out to every run currently flowing through RunRouter. If
            # they are not applicable they will do no harm, and this is
            # expected to be an increasingly rare case.
            self._unlabeled_resources.append(doc['uid'])
            for cbs in self._factory_cbs_by_start.values():
                for cb in cbs:
                    cb('resource', doc)
            for cbs in self._subfactory_cbs_by_start.values():
                for cb in cbs:
                    cb('resource', doc)
        else:
            for cb in self._factory_cbs_by_start[start_uid]:
                cb('resource', doc)
            for cb in self._subfactory_cbs_by_start[start_uid]:
                cb('resource', doc)

    def stop(self, doc):
        start_uid = doc['run_start']
        for cb in self._factory_cbs_by_start[start_uid]:
            cb('stop', doc)
        for cb in self._subfactory_cbs_by_start[start_uid]:
            cb('stop', doc)
        # Clean up references.
        self._subfactories.pop(start_uid, None)
        self._factory_cbs_by_start.pop(start_uid, None)
        self._subfactory_cbs_by_start.pop(start_uid, None)
        for descriptor_uid in self._descriptors.pop(start_uid, ()):
            self._factory_cbs_by_descriptor.pop(descriptor_uid, None)
            self._subfactory_cbs_by_descriptor.pop(descriptor_uid, None)
        self._resources.pop(start_uid, None)


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
        if field not in ['time', 'uid']:
            print('{0}: {1}'.format(field, value))


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
        output.append(event['data'][field])

    return f


def format_num(x, max_len=11, pre=5, post=5):
    if (abs(x) > 10**pre or abs(x) < 10**-post) and x != 0:
        x = '%.{}e'.format(post) % x
    else:
        x = '%{}.{}f'.format(pre, post) % x

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
                raise ValueError("Fields must be strings or objects with a "
                                 "'describe' method that return a dict.")
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


class Table(CallbackBase):
    '''Live updating table

    Parameters
    ----------
    start_doc : dict
        RunStart document

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

    logbook : callable, optional
        Must take a sting as the first positional argument

           def logbook(input_str):
                pass

    out : callable, optional
        Function to call to 'print' a line.  Defaults to `print`
    '''
    _FMTLOOKUP = {'s': '{pad}{{{k}: >{width}.{prec}{dtype}}}{pad}',
                  'f': '{pad}{{{k}: >{width}.{prec}{dtype}}}{pad}',
                  'g': '{pad}{{{k}: >{width}.{prec}{dtype}}}{pad}',
                  'd': '{pad}{{{k}: >{width}{dtype}}}{pad}'}
    _FMT_MAP = {'number': 'f',
                'integer': 'd',
                'string': 's',
                }
    _fm_sty = namedtuple('fm_sty', ['width', 'prec', 'dtype'])
    water_mark = ("{st[plan_type]} {st[plan_name]} ['{st[uid]:.8s}'] "
                  "(scan num: {st[scan_id]})")
    ev_time_key = 'SUPERLONG_EV_TIMEKEY_THAT_I_REALLY_HOPE_NEVER_CLASHES'

    def __init__(self, start_doc, fields, *, stream_name='primary',
                 print_header_interval=50,
                 min_width=12, default_prec=3, extra_pad=1,
                 logbook=None, out=print):
        super().__init__(start_doc)
        self._header_interval = print_header_interval
        # expand objects
        self._fields = get_obj_fields(fields)
        self._stream = stream_name
        self._start = start_doc
        self._stop = None
        self._descriptors = set()
        self._pad_len = extra_pad
        self._extra_pad = ' ' * extra_pad
        self._min_width = min_width
        self._default_prec = default_prec
        self._format_info = OrderedDict([
            ('seq_num', self._fm_sty(10 + self._pad_len, '', 'd')),
            (self.ev_time_key, self._fm_sty(10 + 2 * extra_pad, 10, 's'))
        ])
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

        if doc['name'] != self._stream:
            return

        self._descriptors.add(doc['uid'])

        dk = doc['data_keys']
        for k in self._fields:
            width = max(self._min_width,
                        len(k) + 2,
                        self._default_prec + 1 + 2 * self._pad_len)
            try:
                dk_entry = dk[k]
            except KeyError:
                # this descriptor does not know about this key
                continue

            if dk_entry['dtype'] not in self._FMT_MAP:
                warnings.warn("The key {} will be skipped because LiveTable "
                              "does not know how to display the dtype {}"
                              "".format(k, dk_entry['dtype']))
                continue

            prec = patch_up_precision(dk_entry.get('precision',
                                                   self._default_prec))
            fmt = self._fm_sty(width=width,
                               prec=prec,
                               dtype=self._FMT_MAP[dk_entry['dtype']])

            self._format_info[k] = fmt

        self._sep_format = ('+' +
                            '+'.join('-' * f.width
                                     for f in self._format_info.values()) +
                            '+')
        self._main_fmnt = '|'.join(
            '{{: >{w}}}{pad}'.format(w=f.width - self._pad_len,
                                     pad=' ' * self._pad_len)
            for f in self._format_info.values())
        headings = [k if k != self.ev_time_key else 'time'
                    for k in self._format_info]
        self._header = ('|' +
                        self._main_fmnt.format(*headings) +
                        '|'
                        )
        self._data_formats = OrderedDict(
            (k, self._FMTLOOKUP[f.dtype].format(k=k,
                                                width=f.width-2*self._pad_len,
                                                prec=f.prec, dtype=f.dtype,
                                                pad=self._extra_pad))
            for k, f in self._format_info.items())

        self._count = 0

        self._print(self._sep_format)
        self._print(self._header)
        self._print(self._sep_format)
        super().descriptor(doc)

    def event(self, doc):
        # shallow copy so we can mutate
        if ensure_uid(doc['descriptor']) not in self._descriptors:
            return
        data = dict(doc['data'])
        self._count += 1
        if not self._count % self._header_interval:
            self._print(self._sep_format)
            self._print(self._header)
            self._print(self._sep_format)
        fmt_time = str(datetime.fromtimestamp(doc['time']).time())
        data[self.ev_time_key] = fmt_time
        data['seq_num'] = doc['seq_num']
        cols = [f.format(**{k: data[k]})
                # Show data[k] if k exists in this Event and is 'filled'.
                # (The latter is only applicable if the data is
                # externally-stored -- hence the fallback to `True`.)
                if ((k in data) and doc.get('filled', {}).get(k, True))
                # Otherwise use a placeholder of whitespace.
                else ' ' * self._format_info[k].width
                for k, f in self._data_formats.items()]
        self._print('|' + '|'.join(cols) + '|')
        super().event(doc)

    def stop(self, doc):
        if ensure_uid(doc['run_start']) != self._start['uid']:
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
            self.logbook('\n'.join([wm] + self._rows))
        super().stop(doc)

    def _print(self, out_str):
        self._rows.append(out_str)
        self._out(out_str)


class LiveTable(Table):
    '''Live updating table

    See also Table, a version of this with a life-cycle of one Run. This is an
    older implementation that has a life-cycle of an unlimited number of Runs.

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

    logbook : callable, optional
        Must take a sting as the first positional argument

           def logbook(input_str):
                pass

    out : callable, optional
        Function to call to 'print' a line.  Defaults to `print`
    '''
    def __init__(self, fields, *, stream_name='primary',
                 print_header_interval=50,
                 min_width=12, default_prec=3, extra_pad=1,
                 logbook=None, out=print):
        super().__init__(start_doc={}, fields=fields, stream_name=stream_name,
                         print_header_interval=print_header_interval,
                         min_width=min_width, default_prec=default_prec,
                         extra_pad=extra_pad, logbook=logbook, out=out)
    def start(self, doc):
        self._rows = []
        self._start = doc
        self._stop = None
        self._sep_format = None
