try:
    from typing import Protocol, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, runtime_checkable

from typing import Dict, Any, Optional, Callable, Generator, List, Union, NoReturn


Configuration = Dict[str, Dict[str, Any]]


@runtime_checkable
class Status(Protocol):
    def add_callback(self, callback: Callable[["Status"], NoReturn]) -> NoReturn:
        """Add a callback function to be called upon completion.

        The function must take the status as an argument.

        If the Status object is done when the function is added, it should be
        called immediately.
        """
        ...

    @property
    def done(self) -> bool:
        ...

    @property
    def success(self) -> bool:
        ...


@runtime_checkable
class Readable(Protocol):
    @property
    def parent(self) -> Optional[Any]:
        """``None``, or a reference to a parent device."""
        ...

    def trigger(self) -> Status:
        """Return a ``Status`` that is marked done when the device is done triggering.

        If the device does not need to be triggered, simply return a ``Status``
        that is marked done immediately.
        """
        ...

    def read(self) -> Configuration:
        """Return an OrderedDict mapping field name(s) to values and timestamps.

        The field names must be strings. The values can be any JSON-encodable
        type or a numpy array, which the RunEngine will convert to (nested)
        lists. The timestamps should be UNIX time (seconds since 1970).

        Example return value:

        .. code-block:: python

            OrderedDict(('channel1',
                         {'value': 5, 'timestamp': 1472493713.271991}),
                         ('channel2',
                         {'value': 16, 'timestamp': 1472493713.539238}))
        """
        ...

    def describe(self) -> Configuration:
        """Return an OrderedDict with exactly the same keys as the ``read``
        method, here mapped to metadata about each field.

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

        We refer to each entry as a "data key." These fields are required:

        * source (a descriptive string --- e.g., an EPICS Process Variable)
        * dtype: one of the JSON data types: {'number', 'string', 'array'}
        * shape: list of integers (dimension sizes) --- e.g., ``[5, 5]`` for a
          5x5 array. Use empty list ``[]`` to indicate a scalar.

        Optional additional fields (precision, units, etc.) are allowed.
        The optional field ``external`` should be used to provide information
        about references to externally-stored data, such as large image arrays.
        """
        ...

    def read_configuration(self) -> Configuration:
        """Same API as ``read`` but for slow-changing fields related to configuration.

        (e.g., exposure time)

        These will typically be read only once per run.

        Of course, for simple cases, you can effectively omit this complexity
        by returning an empty dictionary.
        """
        ...

    def describe_configuration(self) -> Configuration:
        """Same API as ``describe``, but corresponding to the keys in ``read_configuration``."""
        ...

    @property
    def name(self) -> str:
        ...


@runtime_checkable
class Movable(Readable, Protocol):
    def set(self, value) -> Status:
        """Return a ``Status`` that is marked done when the device is done moving."""
        ...


@runtime_checkable
class Flyable(Protocol):
    @property
    def name(self) -> str:
        ...

    @property
    def parent(self) -> Optional[Any]:
        """``None``, or a reference to a parent device."""
        ...

    def kickoff(self) -> Status:
        """Begin acculumating data.

        Return a ``Status`` and mark it done when acqusition has begun.
        """
        ...

    def complete(self) -> Status:
        """Return a ``Status`` and mark it done when acquisition has completed."""
        ...

    def collect(self) -> Generator[Dict[str, Any], None, None]:
        """Yield dictionaries that are partial Event documents.

        They should contain the keys 'time', 'data', and 'timestamps'.
        A 'uid' is added by the RunEngine.
        """
        ...

    def describe_collect(self) -> Dict[str, Configuration]:
        """This is like ``describe()`` on readable devices, but with an extra layer of nesting.

        Since a flyer can potentially return more than one event stream, this is a dict
        of stream names (strings) mapped to a ``describe()``-type output for each.
        """
        ...

    # Remaining same as Readable
    def read_configuration(self) -> Configuration:
        """Same as for a Readable device."""
        ...

    def describe_configuration(self) -> Configuration:
        """Same as for a Readable device."""
        ...


@runtime_checkable
class Stageable(Protocol):
    def stage(self) -> List[Any]:
        """An optional hook for "setting up" the device for acquisition.

        It should return a list of devices including itself and any other
        devices that are staged as a result of staging this one.
        (The ``parent`` attribute expresses this relationship: a device should
        be staged/unstaged whenever its parent is staged/unstaged.)
        """
        ...

    def unstage(self) -> List[Any]:
        """A hook for "cleaning up" the device after acquisition.

        It should return a list of devices including itself and any other
        devices that are unstaged as a result of unstaging this one.
        """
        ...


@runtime_checkable
class Pausable(Protocol):
    def pause(self) -> Union[NoReturn, None]:
        """Perform device-specific work when the RunEngine pauses."""
        ...

    def resume(self) -> Union[NoReturn, None]:
        """Perform device-specific work when the RunEngine resumes after a pause."""
        ...


@runtime_checkable
class Stoppable(Protocol):
    def stop(self, sucess=True) -> Union[NoReturn, None]:
        """Safely stop a device that may or may not be in motion.

        The argument ``success`` is a boolean.
        When ``success`` is true, bluesky is stopping the device as planned
        and the device should stop "normally".
        When ``success`` is false, something has gone wrong and the device
        may wish to take defensive action to make itself safe.
        """
        ...


@runtime_checkable
class Subscribable(Protocol):
    def subscribe(self, function: Callable) -> Union[NoReturn, None]:
        """Subscribe to updates in value of a device.

        When the device has a new value ready, it should call ``function``
        asynchronously in a separate thread.

        Needed for :doc:`monitored <async>`.
        """
        ...

    def clear_sub(self, function: Callable) -> Union[NoReturn, None]:
        """Remove a subscription."""
        ...


@runtime_checkable
class Checkable(Protocol):
    def check_value(self, value: Any) -> Union[NoReturn, None]:
        """Test for a valid setpoint without actually moving.

        This should accept the same arguments as ``set``. It should raise an
        Exception if the argument represent an illegal setting --- e.g. a
        position that would move a motor outside its limits or a temperature
        controller outside of its settable range.

        This method is used by simulators that check limits. If not implemented
        those simulators should assume all values are valid, but may warn.
        """
        ...


@runtime_checkable
class Hinted(Protocol):
    @property
    def hints(self) -> Dict:
        """A dictionary of suggestions for best-effort visualization and processing.

        This does not affect what data is read or saved; it is only
        a suggestion to enable automated tools to provide helpful information
        with minimal guidance from the user. See :ref:`hints`.
        """
        ...
