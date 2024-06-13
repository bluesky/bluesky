import os
import time as ttime
from typing import Optional

import numpy as np
from matplotlib.colors import Colormap, Normalize

from bluesky.callbacks._mpl_image_cross_section import CrossSection, InterpolationEnum
from bluesky.callbacks.core import CallbackBase

from ..utils import ensure_uid

cursor_up = "\x1b[1a"
check_mark = "\u2713"


def print_checkmark_on_previous_line():
    print(f"{cursor_up}{check_mark}")


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
        if name != "stop":
            return
        uid = ensure_uid(doc["run_start"])
        header = db[uid]
        for name, doc in header.documents(fill=fill):
            callback(name, doc)
        # Depending on the order that this callback and the
        # databroker-insertion callback were called in, the databroker might
        # not yet have the 'stop' document that we currently have, so we'll
        # use our copy instead of expecting the header to include one.
        if name != "stop":
            callback("stop", doc)

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
    if name != "stop":
        return
    print("  Verifying that all the run's Documents were saved...")
    try:
        header = db[ensure_uid(doc["run_start"])]
    except Exception as e:
        print(f"  Verification Failed! Error: {e}")
        return
    else:
        print_checkmark_on_previous_line()
    print("  Verifying that all externally-stored files are accessible...")
    try:
        list(db.get_events(header, fill=True))
    except Exception as e:
        print(f"  Verification Failed! Error: {e}")
    else:
        print_checkmark_on_previous_line()


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
        self.descriptor_dict = {doc["uid"]: doc}

    def event(self, doc):
        """
        Processes an event by fetching unfilled external fields from a database
        and updating the document in place.

        Args:
        doc (dict): The event document containing data and descriptor references.

        Raises:
        RuntimeError: If no database instance (self.db) is available when needed.
        """
        # Extracting the descriptor based on the document's descriptor reference
        descriptor = self.descriptor_dict[doc["descriptor"]]

        # Finding all fields that are external, present in the document, but not filled
        fields_to_fill = [
            field
            for field in self.fields
            if field in doc["data"]
            and not doc.get("filled", {}).get(field)
            and "external" in descriptor["data_keys"][field]
        ]

        # If there are no fields to fill, exit early (avoiding unnecessary checks and operations)
        if not fields_to_fill:
            return

        # Ensuring that a database instance is available for fetching data
        if self.db is None:
            raise RuntimeError(
                "Either the data must be pre-loaded or "
                "a Broker instance must be provided "
                "via the db parameter of "
                "BrokerCallbackBase."
            )

        # Fetching the data for the unfilled fields from the database
        (filled_event,) = self.db.fill_events(events=[doc], descriptors=[descriptor], fields=fields_to_fill)

        # Updating the document's data in place with the data fetched from the database
        doc["data"].update(filled_event["data"])


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

    def __init__(
        self,
        data_field_name: str,
        *,
        db=None,
        cmap: Optional[Colormap] = None,
        norm: Optional[Normalize] = None,
        limit_func=None,
        auto_redraw: bool = True,
        interpolation: InterpolationEnum = InterpolationEnum.NONE,
        window_title=None,
        cross_section: Optional[CrossSection] = None,
    ):
        import matplotlib.pyplot as plt

        super().__init__((data_field_name,), db=db)
        fig = plt.figure()
        self.field = data_field_name
        self.cross_section = (
            CrossSection(fig, cmap, norm, limit_func, auto_redraw, interpolation)
            if cross_section is None
            else cross_section
        )
        # if window_title:
        # todo that causes error. what is the type of canvas?
        self.cross_section._figure.canvas.set_window_title(window_title)
        self.cross_section._figure.show()

    def event(self, doc):
        super().event(doc)
        data = doc["data"][self.field]
        self.update(data)

    def update(self, data):
        self.cross_section.update_image(data)
        self.cross_section.draw_idle()


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

    def __init__(self, field, template, dryrun=False, overwrite=False, db=None):
        try:
            import tifffile
        except ImportError:
            print(
                "Tifffile is required by this callback. Please install"
                "tifffile and then try again."
                "\n\n\tpip install tifffile\n\nor\n\n\tconda install "
                "tifffile"
            )
            raise
        else:
            # stash a reference so the module is accessible in self._save_image
            self._tifffile = tifffile

        try:
            import doct
        except ImportError:
            print("doct is required by LiveTiffExporter")
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
                raise OSError(f"There is already a file at {filename}. Delete it and try again.")
        if not self.dryrun:
            self._tifffile.imsave(filename, np.asarray(image))
        self.filenames.append(filename)

    def start(self, doc):
        self.filenames = []
        # Convert doc from dict into dottable dict, more convenient
        # in Python format strings: doc.key == doc['key']
        self._start = self._doct.Document("start", doc)
        super().start(doc)

    def event(self, doc):
        if self.field not in doc["data"]:
            return
        super().event(doc)
        image = np.asarray(doc["data"][self.field])
        if image.ndim == 2:
            filename = self.template.format(start=self._start, event=doc)
            self._save_image(image, filename)
        if image.ndim == 3:
            for i, plane in enumerate(image):
                filename = self.template.format(i=i, start=self._start, event=doc)
                self._save_image(plane, filename)

    def stop(self, doc):
        self._start = None
        self.filenames = []
        super().stop(doc)
