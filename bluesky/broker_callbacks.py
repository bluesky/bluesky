import os
import time as ttime
from databroker import DataBroker as db, get_events
import filestore.api as fsapi
from metadatastore.commands import run_start_given_uid, descriptors_by_start
import matplotlib.pyplot as plt
from xray_vision.backend.mpl.cross_section_2d import CrossSection
from .callbacks import CallbackBase
import tifffile
import numpy as np
from databroker import get_images, DataBroker as db


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


def make_tiff_exporter(field, template):
    """
    Build a function that, given a header, exports tiff files.

    The file names will incorporate the contents of the Header.

    Parameters
    ----------
    field : str
        a data key, e.g., 'pe1_image_lightfield'
    template : str
        A templated file path, where curly brackets will be filled in with
        the attributes of 'h', a Header, and 'N', a sequential number.
        e.g., "dir/scan{h.start.scan_id}_by_{h.start.experimenter}_{N}.tiff"

    Returns
    -------
    f : function
        a function that accepts a header and saves TIFF files
    """
    # validate user input
    if not '{N}' in template:
        raise ValueError("template must include '{N}'")

    def f(h, dryrun=False):
        imgs = get_images(h, field)
        # Fill in h, defer filling in N.
        _template = template.format(h=h, N='{N}')
        filenames = [_template.format(N=i) for i in range(len(imgs))]
        # First check that none of the filenames exist.
        for filename in filenames:
            if os.path.isfile(filename):
                raise FileExistsError("There is already a file at {}. Delete "
                                      "it and try again.".format(filename))
        if not dryrun:
            # Write files.
            for filename, img in zip(filenames, imgs):
                tifffile.imsave(filename, np.asarray(img))
        return filenames

    # Write a customized docstring for f based on what is specifcally does.
    f.__doc__ = """
Export sequentially-numbered TIFF files from the {field} field.

Parameters
----------
h : Header
    a header from the databroker
dryrun : bool
    Set to True to return list of filenames without actually writing files.
    False by default.

Returns
-------
filenames : list
    list of filenames where files were written (or, if dryrun, *would* be
    written)
""".format(field=field)
    return f


def verify_files_saved(name, doc):
    "This is a brute-force approach. We retrieve all the data."
    ttime.sleep(0.1)  # Wati for data to be saved.
    if name != 'stop':
        return
    print("  Verifying that all the run's Documents were saved...")
    try:
        header = db[doc['run_start']]
    except Exception as e:
        print("  Verification Failed! Error: {0}".format(e))
        return
    else:
        print('\x1b[1A\u2713')  # print a checkmark on the previous line
    print("  Verifying that all externally-stored files are accessible...")
    try:
        list(get_events(header, fill=True))
    except Exception as e:
        print("  Verification Failed! Error: {0}".format(e))
    else:
        print('\x1b[1A\u2713')  # print a checkmark on the previous line
