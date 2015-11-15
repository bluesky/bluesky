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
        segments denoted with "<>", e.g., '<segment>' are silently omitted,
        when their expansion fails.  The template expressions recognize the
        following variables:

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
        tparts = self._split_template(self.template)
        td = dict(h=h, start=h.start, scan_id=h.start.scan_id)
        td['stop'] = getattr(h, 'stop', None)
        events = get_events(h, fill=False)
        rv = [self._makename(tparts, td)
                for td['N'], td['e'] in enumerate(events)]
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


    def _makename(self, tparts, td):
        nmparts = []
        for seg, isopt in tparts:
            try:
                s = seg.format(**td)
            except (AttributeError, KeyError):
                if not isopt:  raise
                s = ''
            nmparts.append(s)
        rv = ''.join(nmparts)
        rv = os.path.join(self.prefix, rv)
        return rv


    @staticmethod
    def _validate_template(t):
        "Raise ValueError for invalid template string."
        hasN = re.search(r'\{N\b', t)
        if not hasN:
            raise ValueError("template must include '{N}'")
        tfixed = re.sub('[{][^}]*[}]', '', t)
        cleft = tfixed.count('<')
        cright = tfixed.count('>')
        if cleft != cright:
            raise ValueError("Unbalanced segment markers '<', '>'")
        if re.search('<[^>]<|>[^<]*>', t):
            raise ValueError("Nested or misordered segment markers '<', '>'")
        return


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

# class DataNaming
