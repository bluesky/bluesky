try:
    from typing_extensions import Literal, Protocol, TypedDict, runtime_checkable
except ImportError:
    from typing import Literal, Protocol, TypedDict, runtime_checkable

from abc import abstractmethod
from asyncio import CancelledError
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)


# TODO: these are not placed in Events by RE yet
class ReadingOptional(TypedDict, total=False):
    """A dictionary containing the optional per-reading metadata of a piece of scan data"""

    #: * -ve: alarm unknown, e.g. device disconnected
    #: * 0: ok, no alarm
    #: * +ve: there is an alarm
    #:
    #: The exact numbers are transport specific
    alarm_severity: int
    #: A descriptive message if there is an alarm
    message: str


class Reading(ReadingOptional):
    """A dictionary containing the value and timestamp of a piece of scan data"""

    #: The current value, as a JSON encodable type or numpy array
    value: Any
    #: Timestamp in seconds since the UNIX epoch
    timestamp: float


Dtype = Literal["string", "number", "array", "boolean", "integer"]


# https://github.com/bluesky/event-model/blob/master/event_model/schemas/event_descriptor.json
# Just the data_key definition
class DescriptorOptional(TypedDict, total=False):
    """A dictionary containing optional per-scan metadata of a series of Readings"""

    #: Where the data is stored if it is stored external to the events
    external: str
    #: The names for dimensions of the data. Empty list if scalar data
    dims: Sequence[str]
    #: Numpy dtype representation of the data. E.g. <u2
    dtype_str: str
    #: Number of digits after decimal place if a floating point number
    precision: int
    #: Engineering units of the value
    units: str


class Descriptor(DescriptorOptional):
    """A dictionary containing the source and datatype of a series of Readings"""

    #: The source of the data, e.g. an EPICS Process Variable
    source: str
    #: The JSON type of the data in the event. Deprecated for dtype_str
    dtype: Dtype
    #: The shape of the data, e.g. ``[5,5 ]`` for a 5x5 array.
    #: An empty list ``[]`` indicates scalar data.
    shape: Sequence[int]


# https://github.com/bluesky/event-model/blob/master/event_model/schemas/event.json
# But exclude descriptor, seq_num, uid added by RE
class PartialEventOptional(TypedDict, total=False):
    """A dictionary containing optional Event data"""

    #: If any data key is in an external asset it should be present in this
    #: dictionary with value False
    filled: Dict[str, bool]


class PartialEvent(PartialEventOptional):
    """A dictionary containing Event data"""

    #: The event data for each data key
    data: Dict[str, Any]
    #: The timestamps for each data key
    timestamps: Dict[str, float]
    #: The timestamp that the event was taken at
    time: float


# https://github.com/bluesky/event-model/blob/master/event_model/schemas/resource.json
# But exclude run_start added by RE
class PartialResourceOptional(TypedDict, total=False):
    """A dictionary containing optional information needed to load an external resource"""

    #: Whether the path is a posix or windows path. If not given default to posix
    path_semantics: Literal["posix", "windows"]


class PartialResource(TypedDict):
    """A dictionary containing information needed to load an external resource"""

    #: Hint about the format of the resource, and how it should be loaded
    spec: str
    #: Relative path, should not change over lifecycle of the resource
    resource_path: str
    #: Context-dependent root (which may change based on where data is accessed from)
    #: that resource_path is relative to
    root: str
    #: Additional parameters for reading the Resource
    resource_kwargs: Dict[str, Any]
    #: UID that can be referenced in a Datum
    uid: str


# https://github.com/bluesky/event-model/blob/master/event_model/schemas/datum.json
class Datum(TypedDict):
    """A dictionary containing information to load event data from a Resource"""

    #: The UID of a Resource
    resource: str
    #: UID that can be referenced in an Event
    datum_id: str
    #: Additional parameters for reading the Datum
    datum_kwargs: Dict[str, Any]


Asset = Union[
    Tuple[Literal["resource"], PartialResource], Tuple[Literal["datum"], Datum]
]

T = TypeVar("T")
SyncOrAsync = Union[T, Awaitable[T]]

StatusException = Union[Exception, CancelledError]


@runtime_checkable
class Status(Protocol):
    @abstractmethod
    def add_callback(self, callback: Callable[["Status"], None]) -> None:
        """Add a callback function to be called upon completion.

        The function must take the status as an argument.

        If the Status object is done when the function is added, it should be
        called immediately.
        """
        ...

    @abstractmethod
    def exception(self, timeout: Optional[float] = 0.0) -> Optional[StatusException]:
        ...

    @property
    @abstractmethod
    def done(self) -> bool:
        """If done return True, otherwise return False."""
        ...

    @property
    @abstractmethod
    def success(self) -> bool:
        """If done return whether the operation was successful."""
        ...


@runtime_checkable
class HasName(Protocol):
    @property
    @abstractmethod
    def name(self) -> str:
        """Used to populate object_keys in the Event Descriptor

        https://blueskyproject.io/event-model/event-descriptors.html#object-keys"""
        ...


@runtime_checkable
class HasParent(Protocol):
    @property
    @abstractmethod
    def parent(self) -> Optional[Any]:
        """``None``, or a reference to a parent device.

        Used by the RE to stop duplicate stages.
        """
        ...


@runtime_checkable
class HasChildren(Protocol):
    @abstractmethod
    def children(self) -> Iterator[Tuple[str, Any]]:
        """``None``, or an iterator returning a tuple of str, Any.

        each tuple contains the name of the child, and the reference to it.
        """
        ...


@runtime_checkable
class WritesExternalAssets(Protocol):
    @abstractmethod
    def collect_asset_docs(self) -> Iterator[Asset]:
        """Create the resource and datum documents describing data in external source.

        Example yielded values:

        .. code-block:: python

            ('resource', {
                'path_semantics': 'posix',
                'resource_kwargs': {'frame_per_point': 1},
                'resource_path': 'det.h5',
                'root': '/tmp/tmpcvxbqctr/',
                'spec': 'AD_HDF5',
                'uid': '9123df61-a09f-49ae-9d23-41d4d6c6d788'
            })
            # or
            ('datum', {
                'datum_id': '9123df61-a09f-49ae-9d23-41d4d6c6d788/0',
                'datum_kwargs': {'point_number': 0},
                'resource': '9123df61-a09f-49ae-9d23-41d4d6c6d788'}
            })
        """


@runtime_checkable
class Configurable(Protocol):
    @abstractmethod
    def read_configuration(self) -> SyncOrAsync[Dict[str, Reading]]:
        """Same API as ``read`` but for slow-changing fields related to configuration.
        e.g., exposure time. These will typically be read only once per run.

        This can be a standard function or an ``async`` function.
        """
        ...

    @abstractmethod
    def describe_configuration(self) -> SyncOrAsync[Dict[str, Descriptor]]:
        """Same API as ``describe``, but corresponding to the keys in
        ``read_configuration``.

        This can be a standard function or an ``async`` function.
        """
        ...


@runtime_checkable
class Triggerable(Protocol):
    @abstractmethod
    def trigger(self) -> Status:
        """Return a ``Status`` that is marked done when the device is done triggering."""
        ...


@runtime_checkable
class Readable(HasName, Protocol):
    @abstractmethod
    def read(self) -> SyncOrAsync[Dict[str, Reading]]:
        """Return an OrderedDict mapping string field name(s) to dictionaries
        of values and timestamps and optional per-point metadata.

        This can be a standard function or an ``async`` function.

        Example return value:

        .. code-block:: python

            OrderedDict(('channel1',
                         {'value': 5, 'timestamp': 1472493713.271991}),
                         ('channel2',
                         {'value': 16, 'timestamp': 1472493713.539238}))
        """
        ...

    @abstractmethod
    def describe(self) -> SyncOrAsync[Dict[str, Descriptor]]:
        """Return an OrderedDict with exactly the same keys as the ``read``
        method, here mapped to per-scan metadata about each field.

        This can be a standard function or an ``async`` function.

        Example return value:

        .. code-block:: python

            OrderedDict(('channel1',
                         {'source': 'XF23-ID:SOME_PV_NAME',
                          'dtype': 'number',
                          'shape': []}),
                        ('channel2',
                         {'source': 'XF23-ID:SOME_PV_NAME',
                          'dtype': 'number',
                          'shape': []}))
        """
        ...


@runtime_checkable
class Movable(Protocol):
    @abstractmethod
    def set(self, value) -> Status:
        """Return a ``Status`` that is marked done when the device is done moving."""
        ...


class Location(TypedDict, Generic[T]):
    """A dictionary containing the location of a Device"""

    #: Where the Device was requested to move to
    setpoint: T
    #: Where the Device actually is at the moment
    readback: T


@runtime_checkable
class Locatable(Movable, Protocol):
    @abstractmethod
    def locate(self) -> SyncOrAsync[Location]:
        """Return the current location of a Device.

        While a ``Readable`` reports many values, a ``Movable`` will have the
        concept of location. This is where the Device currently is, and where it
        was last requested to move to. This protocol formalizes how to get the
        location from a ``Movable``.
        """
        ...


@runtime_checkable
class Flyable(HasName, Protocol):
    @abstractmethod
    def kickoff(self) -> Status:
        """Begin acculumating data.

        Return a ``Status`` and mark it done when acqusition has begun.
        """
        ...

    @abstractmethod
    def complete(self) -> Status:
        """Return a ``Status`` and mark it done when acquisition has completed."""
        ...

    @abstractmethod
    def collect(self) -> Iterator[PartialEvent]:
        """Yield dictionaries that are partial Event documents.

        They should contain the keys 'time', 'data', and 'timestamps'.
        A 'uid' is added by the RunEngine.
        """
        ...

    @abstractmethod
    def describe_collect(self) -> SyncOrAsync[Dict[str, Dict[str, Descriptor]]]:
        """This is like ``describe()`` on readable devices, but with an extra layer of nesting.

        Since a flyer can potentially return more than one event stream, this is a dict
        of stream names (strings) mapped to a ``describe()``-type output for each.

        This can be a standard function or an ``async`` function.
        """
        ...


@runtime_checkable
class Stageable(Protocol):
    # TODO: we were going to extend these to be able to return plans, what
    # signature should they have?
    @abstractmethod
    def stage(self) -> Union[Status, List[Any]]:
        """An optional hook for "setting up" the device for acquisition.

        It should return a ``Status`` that is marked done when the device is
        done staging.
        """
        ...

    @abstractmethod
    def unstage(self) -> Union[Status, List[Any]]:
        """A hook for "cleaning up" the device after acquisition.

        It should return a ``Status`` that is marked done when the device is finished
        unstaging.
        """
        ...


@runtime_checkable
class Pausable(Protocol):
    @abstractmethod
    def pause(self) -> SyncOrAsync[None]:
        """Perform device-specific work when the RunEngine pauses.

        This can be a standard function or an ``async`` function.
        """
        ...

    @abstractmethod
    def resume(self) -> SyncOrAsync[None]:
        """Perform device-specific work when the RunEngine resumes after a pause.

        This can be a standard function or an ``async`` function.
        """
        ...


@runtime_checkable
class Stoppable(Protocol):
    @abstractmethod
    def stop(self, success=True) -> SyncOrAsync[None]:
        """Safely stop a device that may or may not be in motion.

        The argument ``success`` is a boolean.
        When ``success`` is true, bluesky is stopping the device as planned
        and the device should stop "normally".
        When ``success`` is false, something has gone wrong and the device
        may wish to take defensive action to make itself safe.

        This can be a standard function or an ``async`` function.
        """
        ...


Callback = Callable[[Dict[str, Reading]], None]


@runtime_checkable
class Subscribable(HasName, Protocol):
    @abstractmethod
    def subscribe(self, function: Callback) -> None:
        """Subscribe to updates in value of a device.

        When the device has a new value ready, it should call ``function``
        with something that looks like the output of ``read()``.

        Needed for :doc:`monitored <async>`.
        """
        ...

    @abstractmethod
    def clear_sub(self, function: Callback) -> None:
        """Remove a subscription."""
        ...


@runtime_checkable
class Checkable(Protocol):
    @abstractmethod
    def check_value(self, value: Any) -> SyncOrAsync[None]:
        """Test for a valid setpoint without actually moving.

        This should accept the same arguments as ``set``. It should raise an
        Exception if the argument represent an illegal setting --- e.g. a
        position that would move a motor outside its limits or a temperature
        controller outside of its settable range.

        This method is used by simulators that check limits. If not implemented
        those simulators should assume all values are valid, but may warn.

        This method may be used during a scan, so should not write to any Signals

        This can be a standard function or an ``async`` function.
        """
        ...


class Hints(TypedDict, total=False):
    """A dictionary of optional hints for visualization"""

    #: A list of the interesting fields to plot
    fields: List[str]
    #: Partition fields (and their stream name) into dimensions for plotting
    #:
    #: ``'dimensions': [(fields, stream_name), (fields, stream_name), ...]``
    dimensions: List[Tuple[List[str], str]]
    #: Include this if scan data is sampled on a regular rectangular grid
    gridding: Literal["rectilinear", "rectilinear_nonsequential"]


@runtime_checkable
class HasHints(HasName, Protocol):
    @property
    @abstractmethod
    def hints(self) -> Hints:
        """A dictionary of suggestions for best-effort visualization and processing.

        This does not affect what data is read or saved; it is only
        a suggestion to enable automated tools to provide helpful information
        with minimal guidance from the user. See :ref:`hints`.
        """
        ...


def check_supports(obj, protocol: Type[T]) -> T:
    """Check that an object supports a protocol

    This exists so that multiple protocol checks can be run in a mypy
    compatible way, e.g.::

        triggerable = check_supports(obj, Triggerable)
        triggerable.trigger()
        readable = check_supports(obj, Readable)
        readable.read()
    """
    assert isinstance(obj, protocol), "%s does not implement all %s methods" % (
        obj,
        protocol.__name__,
    )
    return obj
