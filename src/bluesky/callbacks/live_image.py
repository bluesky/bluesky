from matplotlib.colors import Colormap, Normalize

from bluesky.callbacks.core import CallbackBase
from bluesky.callbacks.mpl_package.cross_section import CrossSection
from bluesky.callbacks.mpl_package.interpolation import InterpolationEnum


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
        cmap: str | Colormap | None = None,
        norm: Normalize | None = None,
        limit_func=None,
        auto_redraw: bool = True,
        interpolation: InterpolationEnum = None,
        window_title=None,
    ):
        import matplotlib.pyplot as plt

        super().__init__((data_field_name,), db=db)
        fig = plt.figure()
        self.field = data_field_name
        self.cross_section = CrossSection(fig, cmap, norm, limit_func, auto_redraw, interpolation)
        if window_title:
            self.cross_section._figure.canvas.set_window_title(window_title)
        self.cross_section._figure.show()

    def event(self, doc):
        super().event(doc)
        data = doc["data"][self.field]
        self.update(data)

    def update(self, data):
        self.cross_section.update_image(data)
        self.cross_section.draw_idle()
