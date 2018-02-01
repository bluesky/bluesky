import os
import time as ttime
from .core import CallbackBase
from ..utils import ensure_uid
import numpy as np


class BrokerCallbackBase(CallbackBase):
    """
    Base class for callbacks which need filled documents

    Parameters
    ----------
    fields: Iterable of str
        Names of data field in an Event
    db: Broker instance, optional
        The Broker instance to pull the data from
    """

    def __init__(self, fields, *, db=None):
        self.db = db
        self.fields = fields
        self.descriptor_dict = {}

    def clear(self):
        self.descriptor_dict.clear()

    def stop(self, doc):
        self.clear()

    def descriptor(self, doc):
        self.descriptor_dict = {doc['uid']: doc}

    def event(self, doc):
        # the subset of self.fields that are (1) in the doc and (2) unfilled
        # and (3) external
        fields = [field for field in self.fields
                  if (field in doc['data'] and
                      not doc.get('filled', {}).get(field) and
                      'external' in self.descriptor_dict[
                          doc['descriptor']]['data_keys'][field])]
        if fields:
            if self.db is None:
                raise RuntimeError('Either the data must be pre-loaded or '
                                   'a Broker instance must be provided '
                                   'via the db parameter of '
                                   'BrokerCallbackBase.')
            res, = self.db.fill_events(
                events=[doc],
                descriptors=[self.descriptor_dict[doc['descriptor']]],
                fields=fields)
            doc['data'].update(**res['data'])  # modify in place


class LiveImage(BrokerCallbackBase):
    """
    Stream 2D images in a cross-section viewer.

    Parameters
    ----------
    field : string
        name of data field in an Event
    fs: Registry instance
        The Registry instance to pull the data from
    cmap : str,  colormap, or None
        color map to use.  Defaults to gray
    norm : Normalize or None
       Normalization function to use
    limit_func : callable, optional
        function that takes in the image and returns clim values
    auto_redraw : bool, optional
    interpolation : str, optional
        Interpolation method to use. List of valid options can be found in
        CrossSection2DView.interpolation
    """

    def __init__(self, field, *, db=None, cmap=None, norm=None,
                 limit_func=None, auto_redraw=True, interpolation=None,
                 window_title=None):
        from xray_vision.backend.mpl.cross_section_2d import CrossSection
        import matplotlib.pyplot as plt
        super().__init__((field,), db=db)
        fig = plt.figure()
        self.field = field
        self.cs = CrossSection(fig, cmap, norm,
                               limit_func, auto_redraw, interpolation)
        if window_title:
            self.cs._fig.canvas.set_window_title(window_title)
        self.cs._fig.show()

    def event(self, doc):
        super().event(doc)
        data = doc['data'][self.field]
        self.update(data)

    def update(self, data):
        self.cs.update_image(data)
        self.cs._fig.canvas.draw_idle()


def post_run(callback, db, fill=False):
    """Trigger a callback to process all the Documents from a run at the end.

    This function does not receive the Document stream during collection.
    It retrieves the complete set of Documents from the DataBroker after
    collection is complete.

    Parameters
    ----------
    callback : callable
        Expected signature ::

            def func(doc_name, doc):
                pass

    db : Broker
        The databroker instance to use

    fill : boolean, optional
        Whether to deference externally-stored data in the documents.
        False by default.

    Returns
    -------
    func : function
        a function that accepts a RunStop Document

    Examples
    --------
    Print a table with full (lossless) result set at the end of a run.

    >>> table = LiveTable(['det', 'motor'])
    >>> RE(scan(motor, [det], [1,2,3]), {'stop': post_run(table)})
    +------------+-------------------+----------------+----------------+
    |   seq_num  |             time  |           det  |         motor  |
    +------------+-------------------+----------------+----------------+
    |         3  |  14:02:32.218348  |          5.00  |          3.00  |
    |         2  |  14:02:32.158503  |          5.00  |          2.00  |
    |         1  |  14:02:32.099807  |          5.00  |          1.00  |
    +------------+-------------------+----------------+----------------+

    """
    def f(name, doc):
        if name != 'stop':
            return
        uid = ensure_uid(doc['run_start'])
        header = db[uid]
        for name, doc in header.documents(fill=fill):
            callback(name, doc)
        # Depending on the order that this callback and the
        # databroker-insertion callback were called in, the databroker might
        # not yet have the 'stop' document that we currently have, so we'll
        # use our copy instead of expecting the header to include one.
        if name != 'stop':
            callback('stop', doc)

    return f


def make_restreamer(callback, db):
    """
    Run a callback whenever a uid is updated.

    Parameters
    ----------
    callback : callable
        expected signature is `f(name, doc)`

    db : Broker
        The databroker instance to use

    Example
    -------
    Run a callback whenever a uid is updated.

    >>> def f(name, doc):
    ...     # do stuff
    ...
    >>> g = make_restreamer(f, db)

    To use this `ophyd.callbacks.LastUidSignal`:

    >>> last_uid_signal.subscribe(g)
    """
    def cb(value, **kwargs):
        return db.process(db[value], callback)

    return cb


def verify_files_saved(name, doc, db):
    "This is a brute-force approach. We retrieve all the data."
    ttime.sleep(0.1)  # Wait for data to be saved.
    if name != 'stop':
        return
    print("  Verifying that all the run's Documents were saved...")
    try:
        header = db[ensure_uid(doc['run_start'])]
    except Exception as e:
        print("  Verification Failed! Error: {0}".format(e))
        return
    else:
        print('\x1b[1A\u2713')  # print a checkmark on the previous line
    print("  Verifying that all externally-stored files are accessible...")
    try:
        list(db.get_events(header, fill=True))
    except Exception as e:
        print("  Verification Failed! Error: {0}".format(e))
    else:
        print('\x1b[1A\u2713')  # print a checkmark on the previous line


class LiveTiffExporter(BrokerCallbackBase):
    """
    Save TIFF files.

    Incorporate metadata and data from individual data points in the filenames.

    Parameters
    ----------
    field : str
        a data key, e.g., 'image'
    template : str
        A templated file path, where curly brackets will be filled in with
        the attributes of 'start', 'event', and (for image stacks) 'i',
        a sequential number.
        e.g., "dir/scan{start[scan_id]}_by_{start[experimenter]}_{i}.tiff"
    dryrun : bool
        default to False; if True, do not write any files
    overwrite : bool
        default to False, raising an OSError if file exists
    db : Broker, optional
        The databroker instance to use, if not provided use databroker
        singleton

    Attributes
    ----------
    filenames : list of filenames written in ongoing or most recent run
    """

    def __init__(self, field, template, dryrun=False, overwrite=False,
                 db=None):
        try:
            import tifffile
        except ImportError:
            print("Tifffile is required by this callback. Please install"
                  "tifffile and then try again."
                  "\n\n\tpip install tifffile\n\nor\n\n\tconda install "
                  "tifffile")
            raise
        else:
            # stash a reference so the module is accessible in self._save_image
            self._tifffile = tifffile

        try:
            import doct
        except ImportError:
            print('doct is required by LiveTiffExporter')
        else:
            self._doct = doct

        self.field = field
        super().__init__((field,), db=db.fs)
        self.template = template
        self.dryrun = dryrun
        self.overwrite = overwrite
        self.filenames = []
        self._start = None

    def _save_image(self, image, filename):
        if not self.overwrite:
            if os.path.isfile(filename):
                raise OSError("There is already a file at {}. Delete "
                              "it and try again.".format(filename))
        if not self.dryrun:
            self._tifffile.imsave(filename, np.asarray(image))
        self.filenames.append(filename)

    def start(self, doc):
        self.filenames = []
        # Convert doc from dict into dottable dict, more convenient
        # in Python format strings: doc.key == doc['key']
        self._start = self._doct.Document('start', doc)
        super().start(doc)

    def event(self, doc):
        if self.field not in doc['data']:
            return
        super().event(doc)
        image = np.asarray(doc['data'][self.field])
        if image.ndim == 2:
            filename = self.template.format(start=self._start, event=doc)
            self._save_image(image, filename)
        if image.ndim == 3:
            for i, plane in enumerate(image):
                filename = self.template.format(i=i, start=self._start,
                                                event=doc)
                self._save_image(plane, filename)

    def stop(self, doc):
        self._start = None
        self.filenames = []
        super().stop(doc)
