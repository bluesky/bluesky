#!/usr/bin/env python


"""Generation of file names for each event in databroker header.
"""


import os.path
import re

from dataportal import get_events


class DataNaming(object):
    """
    Build filenames from fields in databroker header.

    Attributes
    ----------
    default_template : str, class attribute
        Default value for the `template` when not specified.

    template : str
        filename template where curly brackets are filled with entries from
        databroker header using the Python format mini-language.  Template
        segments denoted with "<>", e.g., '<e.data[cs700]>' are silently
        omitted when their expansion fails.  The template expressions
        recognize the following variables:

        h    -- Header
        e    -- Event object from the Header h.
        N    -- sequence number of the Event e.
        start    -- abbreviation for h.start
        stop     -- abbreviation for h.stop if it exists
        scan_id  -- abbreviation for h.start.scan_id

    prefix : str
        constant output directory that is prepended to the generated name.

    Parameters
    ----------
    template : str, optional
        Set the `template` attribute.  Use the `default_template` value
        by default.
    prefix : str, optional
        Set the fixed ouput path `prefix`.
    """

    default_template = 'scan{scan_id:05d}_{N:03d}<-T{e.data[cs700]:03.1f}>.tiff'


    def __init__(self, template=None, prefix=''):
        self.template = self.default_template if template is None else template
        self.prefix = prefix
        return


    def __repr__(self):
        nm = type(self).__name__
        fmt = "{0}(template={1.template!r}, prefix={1.prefix!r})"
        return fmt.format(nm, self)


    def __call__(self, h):
        """
        Generate names from fields in databroker header h.

        Parameters
        ----------
        h : Header
            a header from the databroker

        Returns
        -------
        list
            Filenames generated for the header.
        """
        cfmt = ConditionalFormat(self.template)
        td = dict(h=h, start=h.start, scan_id=h.start.scan_id)
        td['stop'] = getattr(h, 'stop', None)
        events = get_events(h, fill=False)
        names = [cfmt.format(**td) for td['N'], td['e'] in enumerate(events)]
        rv = [os.path.join(self.prefix, n) for n in names]
        return rv


    _template = None

    @property
    def template(self):
        """
        Template for generating names from databroker fields.

        Raises
        ------
        ValueError
            When a new template does not include event index {N}
            or if it has unbalanced segment markers '<>'.
        """
        return self._template

    @template.setter
    def template(self, value):
        self._validate_template(value)
        self._template = value
        return


    @staticmethod
    def _validate_template(t):
        "Raise ValueError for invalid template string."
        hasN = re.search(r'\{N\b', t)
        if not hasN:
            raise ValueError("template must include '{N}'")
        ConditionalFormat.validate(t)
        return

# class DataNaming

# ----------------------------------------------------------------------------

class ConditionalFormat:
    """
    Extend format mini-language with conditional segments '<segment>'.

    String segments denoted with `<>`, for example "{first}< {mi}.> {last}"
    are silently omitted when their format expansion fails.  The format
    fields must use named keyword arguments, positional arguments are not
    supported.

    Parameters
    ----------
    s : str
        The template string as for builtin str.format and optional
        segments marked by "<>".

    Raises
    ------
    ValueError
        When string has unbalanced or nested segment markers '<>'.
    """


    @staticmethod
    def validate(s):
        """
        Check syntax of conditional format template `s`.

        Raises
        ------
        ValueError
            When string has unbalanced or nested segment markers '<>'.
        """
        sliteral = re.sub('[{][^}]*[}]', '', s)
        cleft = sliteral.count('<')
        cright = sliteral.count('>')
        if cleft != cright:
            raise ValueError("Unbalanced segment markers '<', '>'")
        if re.search('<[^>]<|>[^<]*>', sliteral):
            raise ValueError("Nested or misordered segment markers '<', '>'")
        return


    def __init__(self, s):
        self.validate(s)
        self._s = s
        self._segments = tuple(self._split_template(s))
        return


    def __repr__(self):
        rv = "ConditionalFormat({!r})".format(self.s)
        return rv


    def __str__(self):
        return self.s


    def format(self, **kwargs):
        """
        Perform a string formatting operation.

        Omit optional segments denoted by '<>' when their expansion fails.
        This function takes only keyword arguments, positional arguments
        are not supported.
        """
        parts = []
        for seg, isopt in self._segments:
            try:
                s = seg.format(**kwargs)
            except (AttributeError, KeyError):
                if not isopt:  raise
                s = ''
            parts.append(s)
        rv = ''.join(parts)
        return rv


    @property
    def s(self):
        "The formatted template string with optional segments in '<>'."
        return self._s


    @staticmethod
    def _split_template(t):
        """Split template string at '<segment>' markers.

        Returns
        -------
        list
            Template split into a list of `(component, optional)` pairs
            where the `optional` flag marks up an optional components.
        """
        segments = [[]]
        barebrac = re.split(r'(\{[^}]*\})', t)
        isopt = False
        isbracket = False
        while barebrac:
            w = barebrac.pop(0)
            delim = '>' if isopt else '<'
            if isbracket or not delim in w:
                segments[-1].append(w)
                isbracket = not isbracket
                continue
            assert delim in w
            wb, we = w.split(delim, 1)
            segments[-1].append(wb)
            segments.append([])
            isopt = not isopt
            barebrac.insert(0, we)
            continue
        rv = []
        for i, seg in enumerate(segments):
            isopt = bool(i % 2)
            s = ''.join(seg)
            if s:  rv.append((s, isopt))
        return rv

# class ConditionalFormat
