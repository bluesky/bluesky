#!/usr/bin/env python


"""Export of 2D image data from databroker header in TIFF format.
"""

import os.path
import datetime

from bluesky.datanaming import DataNaming


class TiffExporter(object):
    """
    Export detector image data from databroker header in a TIFF format.

    Build a callable object which saves image data from a databroker
    header.  Output files are built from a template that may include values
    from the header and its associated events.  TiffExporter can either
    skip or overwrite already existing files.  The TIFF file modification
    time is by default set to the timestamp of the exported event.  The
    timestamp is also used to identify already exported images, possibly
    under a different naming scheme.

    Attributes
    ----------
    default_fetch : str or callable object, class attribute
        Default value for the `fetch` argument in initialization.
        Default is 'pe1_image_lightfield'.
    fetch : callable object
        Function which generates 2D output arrays from databroker Header.
        When set to a string NAME, extract arrays using
        ``databroker.get_images(h, NAME)``.
    naming : DataNaming instance
        Callable object which generates output file names from databroker
        Header.  Use ``naming(h)`` to obtain a list of output names.
        To adjust naming scheme, modifying the `naming.template` and
        `naming.prefix` attributes.
    use_mtime : bool
        When True (default), write output files with modification time set
        according to the exported Event.  Also use modification time to
        identify existing outputs.  When False, write files with current
        modification time and only check if target paths exist.
        Default is `True`.
    mtime_window : float
        Precision in file modification times when searching for existing
        output files.   Ignored when `use_mtime` is `False`.

    Parameters
    ----------
    fetch : str or callable object, optional
        Set function that will generate 2D arrays from databroker Header.
        When string, obtain arrays using ``get_images(h, NAME)``.
        Use the `default_fetch` value when not specified.
    template : str, optional
        Set `naming.template`, a filename template where curly brackets
        are filled with entries from databroker Header.  See the DataNaming
        class for more details.
    prefix : str, optional
        Set `naming.prefix` a constant output directory that is prefixed
        to the generated filenames.
        Default is ''.

    See Also
    --------
    bluesky.datanaming.DataNaming : file name generation from a header.
    """

    # class attributes and defaults

    default_fetch = 'pe1_image_lightfield'
    use_mtime = True
    mtime_window = 0.05


    def __init__(self, fetch=None, template=None, prefix=None):
        self.fetch = self.default_fetch if fetch is None else fetch
        self.naming = DataNaming(template, prefix)
        return


    def __repr__(self):
        nm = type(self).__name__
        rv = ("<{0} fetch:{1.fetch} naming:{1.naming} "
               "use_mtime:{1.use_mtime}>").format(nm, self)
        return rv


    def __call__(self, h, select=None, dryrun=False, overwrite=False):
        """Export sequentially-numbered TIFF files from databroker header.

        Parameters
        ----------
        h : Header
            a header from the databroker
        select : integer indices or mask array or slice, optional
            Export image arrays only at corresponding indices, for example,
            ``[4, 5]`` or ``numpy.s_[-3:]``.  Export all images by default.
        dryrun : bool, optional
            When True, display what would be done without taking any
            action.
        overwrite : bool, optional
            Replace existing tiff files when True.

        Returns
        -------
        list
            Exported output files.
        """
        import tifffile
        from databroker import get_events
        if dryrun:
            dryordo = lambda msg, f, *args, **kwargs:  print(msg)
        else:
            dryordo = lambda msg, f, *args, **kwargs:  f(*args, **kwargs)
        setmtime = lambda f, t: os.utime(f, (os.path.getatime(f), t))
        noop = lambda : None
        msgrm = "remove existing output {}"
        msgskip = "skip {f} already saved as {o}"
        outputfiles = self.naming(h)
        eventtimes = [e.time for e in get_events(h, fill=False)]
        n = len(eventtimes)
        selection = _makesetofindices(n, select)
        dircache = {}
        outputfrom = {
                f : self.outputFileExists(f,
                    mtime=self.use_mtime and etime, dircache=dircache)
                for f, etime in zip(outputfiles, eventtimes)}
        imgs = self.fetch(h)
        rv = []
        for i, f, img, etime in zip(range(n), outputfiles, imgs, eventtimes):
            if not i in selection:  continue
            existingoutputs = outputfrom[f]
            # skip this image when overwrite is False
            if not overwrite:
                for o in existingoutputs:
                    print(msgskip.format(f=f, o=o))
                if existingoutputs:  continue
            assert overwrite or not existingoutputs
            for o in existingoutputs:
                dryordo(msgrm.format(o), os.remove, o)
            msg = "write image data to {}".format(f)
            dryordo(msg, tifffile.imsave, f, img)
            rv.append(f)
            if self.use_mtime:
                isotime = datetime.datetime.fromtimestamp(etime).isoformat(' ')
                msg = "adjust image file mtime to {}".format(isotime)
                dryordo(msg, setmtime, f, etime)
        return rv


    def findExistingOutputs(self, h, select=None):
        """Find any already saved output files given a databroker header.

        When `use_mtime` is active, find files with the same extension
        and output directory that have the same modification time.

        Parameters
        ----------
        h : Header
            a header from the databroker
        select : integer indices or mask array or slice, optional
            Find outputs only for the events at specified indices, for
            example, ``[0, 1, 2]`` or ``numpy.s_[-3:]``.  Search all
            events by default.

        Returns
        -------
        list
            Existing absolute output paths or an empty list.
        """
        from databroker import get_events
        # avoid repeated calls to os.listdir
        dircache = {}
        filenames = self.naming(h)
        eventtimes = [e.time for e in get_events(h, fill=False)]
        if not self.use_mtime:
            eventtimes = len(eventtimes) * [None]
        n = len(eventtimes)
        selection = _makesetofindices(n, select)
        rv = []
        for i, f, etime in zip(range(n), filenames, eventtimes):
            if not i in selection:  continue
            rv.extend(self.outputFileExists(f, etime, dircache=dircache))
        return rv


    def outputFileExists(self, filename, mtime, dircache=None):
        """Check for already saved instances of the given output path.

        When `mtime` is non-zero, include files with the same extension
        and directory that have that modification time.

        Parameters
        ----------
        filename : str
            output filename to be checked
        mtime : float or None
            modification time in epoch seconds to be used for locating
            outputs with a different naming scheme.  Do not use when
            zero or None.
        dircache : dict, optional
            cached outputs from `os.listdir`.  For internal use only.

        Returns
        -------
        list
            Existing absolute output paths or an empty list.
        """
        fa = os.path.abspath(filename)
        rv = [fa] if os.path.exists(filename) else []
        if not mtime:  return rv
        outputdir = os.path.abspath(os.path.dirname(filename))
        ext = os.path.splitext(filename)
        dircache = dircache if dircache is not None else {}
        if not outputdir in dircache:
            ofiles = [os.path.join(outputdir, f)
                    for f in os.listdir(outputdir)]
            ofiles = [f for f in ofiles
                    if f.endswith(ext) and os.path.isfile(f)]
            dircache[outputdir] = ofiles
        for f in dircache[outputdir]:
            if f == fa:
                assert rv[0] == f
                continue
            dt = abs(os.path.getmtime(f) - mtime)
            if dt < self.mtime_window:
                rv.append(f)
        return rv

    # Properties -------------------------------------------------------------

    @property
    def fetch(self):
        """Function that generates image arrays from a databroker headers.

        Must be a callable object or a string.  When set to a string NAME,
        use get_images(headers, NAME).
        """
        return self._fetch

    @fetch.setter
    def fetch(self, value):
        if isinstance(value, str):
            from functools import partial
            from databroker import get_images
            self._fetch = partial(get_images, name=value)
        elif callable(value):
            self._fetch = value
        else:
            emsg = "The fetch attribute must be a string or callable object."
            raise TypeError(emsg)
        return

# class TiffExporter


def _makesetofindices(n, select):
    import numpy
    indices = numpy.arange(n)
    if select is not None:
        indices = indices[select].flat
    return set(indices)
