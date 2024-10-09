from abc import abstractmethod
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)

from event_model.documents import Datum, StreamDatum, StreamResource
from event_model.documents.event import PartialEvent

# Including Dtype here because ophyd imports Dtype directly from protocols, not event-model.
from event_model.documents.event_descriptor import DataKey, Dtype
from event_model.documents.event_page import PartialEventPage
from event_model.documents.resource import PartialResource
from typing_extensions import TypedDict

# Squashes warning
Dtype = Dtype  # type: ignore

try:
    from typing import ParamSpec
except ImportError:
    from typing_extensions import ParamSpec  # type: ignore


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


T = TypeVar("T")
P = ParamSpec("P")


class Reading(Generic[T], ReadingOptional):
    """A dictionary containing the value and timestamp of a piece of scan data"""

    #: The current value, as a JSON encodable type or numpy array
    value: T
    #: Timestamp in seconds since the UNIX epoch
    timestamp: float


Asset = Union[
    Tuple[Literal["resource"], PartialResource],
    Tuple[Literal["datum"], Datum],
]


StreamAsset = Union[
    Tuple[Literal["stream_resource"], StreamResource],
    Tuple[Literal["stream_datum"], StreamDatum],
]


SyncOrAsync = Union[T, Awaitable[T]]
SyncOrAsyncIterator = Union[Iterator[T], AsyncIterator[T]]


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
    def exception(self, timeout: Optional[float] = 0.0) -> Optional[BaseException]: ...

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
        """Used to populate object_keys in the Event DataKey

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
class WritesExternalAssets(Protocol):
    @abstractmethod
    def collect_asset_docs(self) -> SyncOrAsyncIterator[Asset]:
        """Create the resource and datum documents describing data in external
            source.

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
        ...


@runtime_checkable
class WritesStreamAssets(Protocol):
    @abstractmethod
    def collect_asset_docs(self, index: Optional[int] = None) -> SyncOrAsyncIterator[StreamAsset]:
        """Create the resource and datum documents describing data in external
            source up to a given index if provided.

            An index will be provided when using stream resources and datums. The asset
            docs will be collected from multiple streams and synchronised on the
            highest common stream index.

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
        ...

    @abstractmethod
    def get_index(self) -> SyncOrAsync[int]:
        """Retrive the current index of writer."""
        ...


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
    def describe_configuration(self) -> SyncOrAsync[Dict[str, DataKey]]:
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
class Preparable(Protocol):
    @abstractmethod
    def prepare(self, value) -> Status:
        """Prepare a device for scanning.

        This method provides similar functionality to ``Stageable.stage`` and
        ``Movable.set``, with key differences:

        ``Stageable.stage``
        ^^^^^^^^^^^^^^^^^^^
        Staging a device translates to, "I'm going to use this in a scan, but I'm
        not sure how". Preparing it translates to, "I'm about to do a step or a fly
        scan with these parameters". Staging should be universal across many different
        types of scans, however prepare is specific to an input value passed in.

        ``Movable.set``
        ^^^^^^^^^^^^^^^
        For some devices, preparation for a scan could involve multiple soft or
        hardware signals being configured and/or set. ``prepare`` therefore allows
        these to be bundled together, along with other logic.

        For example, a Flyable device should have the following methods called on it to
        perform a fly-scan:

            prepare(flyscan_params)
            kickoff()
            complete()

        If the device is a detector, ``collect_asset_docs`` can be called repeatedly
        while ``complete`` is not done to publish frames. Alternatively, to step-scan a
        detector,

            prepare(frame_params) to setup N software triggered frames
            trigger() to take N frames
            collect_asset_docs() to publish N frames

        Returns a Status that is marked done when the device is ready for a scan.
        """
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
    def describe(self) -> SyncOrAsync[Dict[str, DataKey]]:
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
class Collectable(HasName, Protocol):
    @abstractmethod
    def describe_collect(self) -> SyncOrAsync[Union[Dict[str, DataKey], Dict[str, Dict[str, DataKey]]]]:
        """This is like ``describe()`` on readable devices, but with an extra layer of nesting.

        Since a flyer can potentially return more than one event stream, this is either
            * a dict of stream names (strings) mapped to a ``describe()``-type output for each.
            * a ``describe()``-type output of the descriptor name passed in with the ``name``
                argument of the message.

        This can be a standard function or an ``async`` function.
        """
        ...


@runtime_checkable
class EventCollectable(Collectable, Protocol):
    @abstractmethod
    def collect(self) -> SyncOrAsyncIterator[PartialEvent]:
        """Yield dictionaries that are partial Event documents.

        They should contain the keys 'time', 'data', and 'timestamps'.
        A 'uid' is added by the RunEngine.
        """
        ...


@runtime_checkable
class EventPageCollectable(Collectable, Protocol):
    @abstractmethod
    def collect_pages(self) -> SyncOrAsyncIterator[PartialEventPage]:
        """Yield dictionaries that are partial EventPage documents.

        They should contain the keys 'time', 'data', and 'timestamps'.
        A 'uid' is added by the RunEngine.
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


@runtime_checkable
class NamedMovable(Movable, HasHints, Protocol):
    """A movable object that has a name and hints."""

    ...


def check_supports(obj: T, protocol: Type[Any]) -> T:
    """Check that an object supports a protocol

    This exists so that multiple protocol checks can be run in a mypy
    compatible way, e.g.::

        triggerable = check_supports(obj, Triggerable)
        triggerable.trigger()
        readable = check_supports(obj, Readable)
        readable.read()
    """
    assert isinstance(obj, protocol), "%s does not implement all %s methods" % (obj, protocol.__name__)  # noqa: UP031
    return obj


# Descriptor with previous name on imports for backwards compatibility.
Descriptor = DataKey
