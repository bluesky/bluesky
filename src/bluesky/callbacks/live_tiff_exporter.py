import os

import numpy as np

from bluesky.callbacks.live_image import BrokerCallbackBase


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
