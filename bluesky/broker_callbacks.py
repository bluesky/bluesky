from databroker import DataBroker as db, get_events
import filestore.api as fsapi
from metadatastore.commands import run_start_given_uid, descriptors_by_start
from xray_vision.backend.mpl.cross_section_2d import CrossSection
from .callbacks import CallbackBase


class LiveImage(CallbackBase):
    """
    Stream 2D images in a cross-section viewer.

    Parameters
    ----------
    field : string
        name of data field in an Event

    Note
    ----
    Requires a matplotlib fix that is not released as of this writing. The
    relevant commit is a951b7.
    """
    def __init__(self, field):
        from matplotlib.pyplot import figure
        super().__init__()
        self.field = field
        fig = figure()
        self.cs = CrossSection(fig)
        self.cs._fig.show()

    def event(self, doc):
        uid = doc['data'][self.field]
        data = fsapi.retrieve(uid)
        self.cs.update_image(data)
        self.cs._fig.canvas.draw()
        self.cs._fig.canvas.flush_events()


def post_run(callback):
    """
    Trigger a callback to process all the Documents from a run at the end.

    This function does not receive the Document stream during collection.
    It retrieves the complete set of Documents from the DataBroker after
    collection is complete.

    Parameters
    ----------
    callback : callable
        Expected signature ::

            def func(doc_name, doc):
                pass

    Returns
    -------
    func : function
        a function that acepts a RunStop Document

    Examples
    --------
    Print a table with full (lossless) result set at the end of a run.

    >>> s = Ascan(motor, [det1], [1,2,3])
    >>> table = LiveTable(['det1', 'motor'])
    >>> RE(s, {'stop': post_run(table)})
    +------------+-------------------+----------------+----------------+
    |   seq_num  |             time  |          det1  |         motor  |
    +------------+-------------------+----------------+----------------+
    |         3  |  14:02:32.218348  |          5.00  |          3.00  |
    |         2  |  14:02:32.158503  |          5.00  |          2.00  |
    |         1  |  14:02:32.099807  |          5.00  |          1.00  |
    +------------+-------------------+----------------+----------------+
    """
    def f(name, stop_doc):
        if name != 'stop':
            return
        uid = stop_doc['run_start']
        start = run_start_given_uid(uid)
        descriptors = descriptors_by_start(uid)
        # For convenience, I'll rely on the broker to get Events.
        header = db[uid]
        events = get_events(header)
        callback('start', start)
        for d in descriptors:
            callback('descriptor', d)
        for e in events:
            callback('event', e)
        callback('stop', stop_doc)
    return f


def make_tiff_exporter(fetch=None, template=None, prefix=None, **kwargs):
    """
    Build a function that, given a header, exports tiff files.

    The file names will incorporate the contents of the Header.

    Parameters
    ----------
    fetch : str or callable object, optional
        Set function that will generate 2D arrays from databroker Header.
        When string, obtain arrays using ``get_images(h, NAME)``.
        See the TiffExporter class for more details.
    template : str, optional
        Set filename template where curly brackets are filled with entries
        from databroker Header.  See the DataNaming class for more details.
    prefix : str, optional
        Set filename prefix, a constant output directory that is prepended
        to generated filenames.
        Default is ''.
    kwargs : TiffExporter attributes, optional
        Use to set any additional attributes of the constructed TiffExporter
        object.  Raise AttributeError when TiffExporter does not have such
        attribute.

    Returns
    -------
    TiffExporter
        a callable object that accepts a header and saves TIFF files.

    See Also
    --------
    bluesky.datanaming.TiffExporter : configurable TIFF export from a header.
    bluesky.datanaming.DataNaming : file name generation from a header.
    """
    from bluesky.tiffexporter import TiffExporter
    te = TiffExporter(fetch=fetch, template=template, prefix=prefix)
    # validate and apply keyword arguments
    emsg = "Invalid keyword argument {}, TiffExporter has no such attribute."
    for n, v in kwargs.items():
        if not hasattr(te, n):
            raise AttributeError(emsg.format(n))
        setattr(te, n, v)
    return te
