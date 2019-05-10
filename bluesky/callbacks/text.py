from collections import namedtuple, OrderedDict
from datetime import datetime
import sys
import time

from event_model import DocumentRouter, unpack_event_page

from .best_effort import hinted_fields


class TextTableFactory:
    """
    Subscribe a TextTable to a given stream with optional column filtering.

    This factory is interested in all Runs and in one Event stream.

    Parameters
    ----------
    stream_name : string
        The Event stream to represent as a table
    include : Iterable
        Set of columns to include in the table. If None (default) all fields are
        included. This parameter is mutually incompatible with 'exclude'.
    exclude : Iterable
        Set of columns to exclude from table. If None (default) no fields are
        excluded. This parameter is mutually incompatible with 'include'.
    """
    def __init__(self, stream_name, include=None, exclude=None, file=sys.stdout):
        self.stream_name = stream_name
        self.include = include
        self.exclude = exclude
        self.file = file

    def __repr__(self):
        return (f"{type(self).__name__}(stream_name={self.stream_name!r}, "
                f"include={self.include!r}, exclude={self.exclude!r}, "
                f"file={self.file!r})")

    def __call__(self, name, start_doc):
        def subfactory(name, descriptor_doc):
            if descriptor_doc.get('name') == self.stream_name:
                text_table = TextTable(
                    include=self.include,
                    exclude=self.exclude,
                    file=self.file)
                text_table.start(start_doc)
                text_table.descriptor(descriptor_doc)
                return [text_table]
            else:
                return []

        return [], [subfactory]


class TextTable(DocumentRouter):
    '''Print a text table that updates as documents are added.

    This is meant to represent one event stream from one run. It expected to
    see one 'start' document and one 'descriptor' document.

    Parameters
    ----------
    include : Iterable
        Set of columns to include in the table. If None (default) all fields are
        included. This parameter is mutually incompatible with 'exclude'.
    exclude : Iterable
        Set of columns to exclude from table. If None (default) no fields are
        excluded. This parameter is mutually incompatible with 'include'.
    print_header_interval : int, optional
         Reprint the header every this many lines, defaults to 50
    min_width : int, optional
         The minimum width is spaces of the data columns.  Defaults to 12
    default_prec : int, optional
         Precision to use if it can not be found in descriptor, defaults to 3
    extra_pad : int, optional
         Number of extra spaces to put around the printed data, defaults to 1
    out : writable, optional
        Defaults to ``sys.stdout``
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

    def __init__(self, include=None, exclude=None,
                 print_header_interval=50,
                 min_width=12, default_prec=3, extra_pad=1,
                 file=sys.stdout):
        if include is not None and exclude is not None:
            raise ValueError("include and exclude are mutally exclusive")
        self._include = include
        self._exclude = exclude
        self._header_interval = print_header_interval
        self._stop = None
        self._pad_len = extra_pad
        self._extra_pad = ' ' * extra_pad
        self._min_width = min_width
        self._default_prec = default_prec
        self._format_info = OrderedDict([
            ('seq_num', self._fm_sty(10 + self._pad_len, '', 'd')),
            (self.ev_time_key, self._fm_sty(10 + 2 * extra_pad, 10, 's'))
        ])
        self._sep_format = None
        self._file = file

    def start(self, doc):
        self._start = doc

    def descriptor(self, doc):
        def patch_up_precision(p):
            try:
                return int(p)
            except (TypeError, ValueError):
                return self._default_prec

        dk = doc['data_keys']
        if self._include is not None:
            fields = self._include
        elif self._exclude is not None:
            fields = set(dk) - set(exclude)
        else:
            fields = dk
        for k in fields:
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

    def event_page(self, doc):
        # Do the actual work in the 'event' method in this case --- no point in
        # trying to vectorize over pages.
        event = self.event  # Avoid attribute lookup in hot loop.
        for event_doc in unpack_event_page(doc):
            event(event_doc)

    def event(self, doc):
        # shallow copy so we can mutate
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

    def stop(self, doc):
        # This sleep is just cosmetic. It improves the odds that the bottom
        # border is not printed until all the rows from events are printed,
        # avoiding this ugly scenario:
        #
        # |         4 | 22:08:56.7 |      0.000 |
        # +-----------+------------+------------+
        # generator scan ['6d3f71'] (scan num: 1)
        # Out[2]: |         5 | 22:08:56.8 |      0.000 |
        time.sleep(0.1)

        if self._sep_format is not None:
            self._print(self._sep_format)
        self._stop = doc

        wm = self.water_mark.format(st=self._start)
        self._print(wm)
        super().stop(doc)

    def _print(self, out_str):
        print(out_str, flush=True, file=self._file)


def heading_printer(name, doc):
    """
    This prints a text header summarizing metadata from the 'start' document.

    This factory uses the 'start' document and requires no further information.
    """
    tt = datetime.fromtimestamp(doc['time']).utctimetuple()
    print("Transient Scan ID: {0}     Time: {1}".format(
        doc['scan_id'],
        time.strftime("%Y/%m/%d %H:%M:%S", tt)))
    print("Persistent Unique Scan ID: '{0}'".format(doc['uid']))
    return [], []


def new_stream_printer(name, doc):
    """
    This prints line of text each time a unique stream is begun.

    This factory uses every 'descriptor' document but requires no further
    information.
    """
    streams = set()
    def subfactory(name, descriptor_doc):
        name = descriptor_doc.get('name')
        if name is None:  # 'name' is missing from very old documents
            return
        if name not in streams:
            streams.add(name)
            print(f"New stream: {name!r}")
        return []
    return [], [subfactory]


class BaselinePrinterFactory:
    """
    Print a summary of the baseline readings at the beginning and end of a run.

    This factory is interested in all Runs and the 'baseline' Event stream.
    """
    def __init__(self, include=None, exclude=None,
                 stream_name='baseline', file=sys.stdout):
        self.include = include
        self.exclude = exclude
        self.stream_name = stream_name
        self.file = file

    def __repr__(self):
        return f'{type(self).__name__}(file={self.file!r})'

    def __call__(self, name, start_doc):
        def subfactory(name, descriptor_doc):
            if descriptor_doc.get('name') == self.stream_name:
                cb = BaselinePrinter(include=self.include, exclude=self.exclude)
                cb.start(start_doc)
                cb.descriptor(descriptor_doc)
                return [cb]
            else:
                return []
        return [], [subfactory]


class BaselinePrinter(DocumentRouter):
    """
    Print baseline readings.

    Parameters
    ----------

    include : Iterable
        Set of columns to include in the table. If None (default) all fields are
        included. This parameter is mutually incompatible with 'exclude'.
    exclude : Iterable
        Set of columns to exclude from table. If None (default) no fields are
        excluded. This parameter is mutually incompatible with 'include'.
    """
    def __init__(self, include=None, exclude=None, file=sys.stdout):
        # TODO include, exclude
        self._descriptor = None
        self._baseline_toggle = True
        self._file = file

    def descriptor(self, doc):
        self._descriptor = doc

    def event_page(self, doc):
        # Do the actual work in the 'event' method in this case, since baseline
        # readings always come one at a time.
        event_method = self.event  # Avoid attribute lookup in hot loop.
        for event_doc in unpack_event_page(doc):
            event_method(event_doc)

    def event(self, doc):
        columns = hinted_fields(self._descriptor)
        self._baseline_toggle = not self._baseline_toggle
        if self._baseline_toggle:
            subject = 'End-of-run'
        else:
            subject = 'Start-of-run'
        print('{} baseline readings:'.format(subject), file=self._file)
        border = '+' + '-' * 32 + '+' + '-' * 32 + '+'
        print(border, file=self._file)
        for k, v in doc['data'].items():
            if k not in columns:
                continue
            print('| {:>30} | {:<30} |'.format(k, v), file=self._file)
        print(border, file=self._file)
