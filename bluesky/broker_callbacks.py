from dataportal import DataBroker as db
import filestore
import filestore.api as fsapi
from metadatastore.commands import find_event_descriptors, find_run_starts
import matplotlib.pyplot as plt
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
        super().__init__()
        self.field = field
        fig = plt.figure()
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
        a function that accepts all four Documents

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
    def f(stop_doc):
        uid = stop_doc['run_start']
        start, = find_run_starts(uid=uid)
        descriptors = find_event_descriptors(run_start=uid)
        # For convenience, I'll rely on the broker to get Events.
        header = db[uid]
        events = db.fetch_events(header)
        callback.start(start)
        for d in descriptors:
            callback.descriptor(d)
        for e in events:
            callback.event(e)
        callback.stop(stop_doc)
    return f
