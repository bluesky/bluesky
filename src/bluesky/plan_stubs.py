import itertools
import operator
import time
import typing
import uuid
import warnings
from collections.abc import Awaitable, Callable, Hashable, Iterable, Mapping, Sequence
from functools import reduce
from typing import Any, Literal

from cycler import cycler

from bluesky.suspenders import SuspenderBase

try:
    # cytools is a drop-in replacement for toolz, implemented in Cython
    from cytools import partition
except ImportError:
    from toolz import partition

from event_model import ComposeEvent
from event_model.documents import EventDescriptor

from .protocols import (
    Configurable,
    Flyable,
    Locatable,
    Location,
    Movable,
    PartialEvent,
    Preparable,
    Readable,
    Reading,
    Stageable,
    Status,
    Stoppable,
    Triggerable,
    check_supports,
)
from .utils import (
    CustomPlanMetadata,
    Msg,
    MsgGenerator,
    ScalarOrIterableFloat,
    all_safe_rewind,
    ensure_generator,
    get_hinted_fields,
    merge_cycler,
    plan,
    separate_devices,
    short_uid,
)
from .utils import (
    short_uid as _short_uid,
)

#: Any plan function that takes a reading given a list of Readables
TakeReading = Callable[[Sequence[Readable]], MsgGenerator[Mapping[str, Reading]]]


@plan
def declare_stream(
    *objs: Readable, name: str, collect: bool = False
) -> MsgGenerator[tuple[EventDescriptor, ComposeEvent]]:
    """
    Bundle future readings into a new Event document.

    Parameters
    ----------
    objs :
        objects whose readings will be present in the stream
    name : string, optional
        name given to event stream, used for convenient identification
        default is 'primary'
    collect : bool, optional
        collect as well as describe when declaring the stream
        default is `False`

    Yields
    ------
    msg : Msg
        Msg('create', name=name)

    See Also
    --------
    :func:`bluesky.plan_stubs.save`
    """
    return (yield Msg("declare_stream", None, *separate_devices(objs), name=name, collect=collect))


@plan
def create(name: str = "primary") -> MsgGenerator:
    """
    Bundle future readings into a new Event document.

    Parameters
    ----------
    name : string, optional
        name given to event stream, used for convenient identification
        default is 'primary'

    Yields
    ------
    msg : Msg
        Msg('create', name=name)

    See Also
    --------
    :func:`bluesky.plan_stubs.save`
    """
    return (yield Msg("create", name=name))


@plan
def save() -> MsgGenerator:
    """
    Close a bundle of readings and emit a completed Event document.

    Yields
    ------
    msg : Msg
        Msg('save')

    See Also
    --------
    :func:`bluesky.plan_stubs.create`
    """
    return (yield Msg("save"))


@plan
def drop() -> MsgGenerator:
    """
    Drop a bundle of readings without emitting a completed Event document.

    Yields
    ------
    msg : Msg
        Msg('drop')

    See Also
    --------
    :func:`bluesky.plan_stubs.save`
    :func:`bluesky.plan_stubs.create`
    """
    return (yield Msg("drop"))


@plan
def read(obj: Readable) -> MsgGenerator[Reading]:
    """
    Take a reading and add it to the current bundle of readings.

    Parameters
    ----------
    obj : Device or Signal

    Yields
    ------
    msg : Msg
        Msg('read', obj)

    Returns
    -------
    reading :
        Reading object representing information recorded
    """
    return (yield Msg("read", obj))


@typing.overload
def locate(obj: Locatable, squeeze: Literal[True] = True) -> Location: ...  # type: ignore[overload-overlap]
@typing.overload
def locate(*objs: Locatable, squeeze: bool = True) -> list[Location]: ...
@plan
def locate(*objs, squeeze=True):
    """
    Locate some Movables and return their locations.

    Parameters
    ----------
    obj : Device or Signal
    sqeeze: bool
        If True, return the result as a list.
        If False, always return a list of retults even with a single object.

    Yields
    ------
     msg : Msg
        ``Msg('locate', obj1, ..., objn, squeeze=True)``
    """
    return (yield Msg("locate", *objs, squeeze=squeeze))


@plan
def monitor(obj: Readable, *, name: str | None = None, **kwargs) -> MsgGenerator:
    """
    Asynchronously monitor for new values and emit Event documents.

    Parameters
    ----------
    obj : Device or Signal
    args :
        passed through to ``obj.subscribe()``
    name : string, optional
        name of event stream; default is None
    kwargs :
        passed through to ``obj.subscribe()``

    Yields
    ------
    msg : Msg
        ``Msg('monitor', obj, *args, **kwargs)``

    See Also
    --------
    :func:`bluesky.plan_stubs.unmonitor`
    """
    return (yield Msg("monitor", obj, name=name, **kwargs))


@plan
def unmonitor(obj: Readable) -> MsgGenerator:
    """
    Stop monitoring.

    Parameters
    ----------
    obj : Device or Signal

    Yields
    ------
    msg : Msg
        Msg('unmonitor', obj)

    See Also
    --------
    :func:`bluesky.plan_stubs.monitor`
    """
    return (yield Msg("unmonitor", obj))


@plan
def null() -> MsgGenerator:
    """
    Yield a no-op Message. (Primarily for debugging and testing.)

    Yields
    ------
    msg : Msg
        Msg('null')
    """
    return (yield Msg("null"))


@plan
def abs_set(
    obj: Movable,
    *args: Any,
    group: Hashable | None = None,
    wait: bool = False,
    **kwargs,
) -> MsgGenerator[Status]:
    """
    Set a value. Optionally, wait for it to complete before continuing.

    Parameters
    ----------
    obj : Device
    args :
        passed to obj.set()
    group : string (or any hashable object), optional
        identifier used by 'wait'
    wait : boolean, optional
        If True, wait for completion before processing any more messages.
        False by default.
    kwargs :
        passed to obj.set()

    Yields
    ------
    msg : Msg

    Returns
    -------
    status :
        Status that completes when the value is set. If `wait` is True,
        this will always be complete by the time it is returned.

    See Also
    --------
    :func:`bluesky.plan_stubs.rel_set`
    :func:`bluesky.plan_stubs.wait`
    :func:`bluesky.plan_stubs.mv`
    """
    if wait and group is None:
        group = str(uuid.uuid4())
    ret = yield Msg("set", obj, *args, group=group, **kwargs)
    if wait:
        yield Msg("wait", None, group=group)
    return ret


@plan
def rel_set(
    obj: Movable,
    *args: Any,
    group: Hashable | None = None,
    wait: bool = False,
    **kwargs,
) -> MsgGenerator[Status]:
    """
    Set a value relative to current value. Optionally, wait before continuing.

    Parameters
    ----------
    obj : Device
    args :
        passed to obj.set()
    group : string (or any hashable object), optional
        identifier used by 'wait'; None by default
    wait : boolean, optional
        If True, wait for completion before processing any more messages.
        False by default.
    kwargs :
        passed to obj.set()

    Yields
    ------
    msg : Msg

    Returns
    -------
    status :
        Status that completes when the value is set. If `wait` is True,
        this will always be complete by the time it is returned.

    See Also
    --------
    :func:`bluesky.plan_stubs.abs_set`
    :func:`bluesky.plan_stubs.wait`
    """
    from .preprocessors import relative_set_wrapper

    return (yield from relative_set_wrapper(abs_set(obj, *args, group=group, wait=wait, **kwargs)))


# The format (device1, value1, device2, value2, ...)
# is not currently able to be represented in python's type system
@plan
def mv(
    *args: Movable | Any,
    group: Hashable | None = None,
    timeout: float | None = None,
    **kwargs,
) -> MsgGenerator[tuple[Status, ...]]:
    """
    Move one or more devices to a setpoint. Wait for all to complete.

    If more than one device is specified, the movements are done in parallel.

    Parameters
    ----------
    args :
        device1, value1, device2, value2, ...
    group : string, optional
        Used to mark these as a unit to be waited on.
    timeout : float, optional
        Specify a maximum time that the move(s) can be waited for.
    kwargs :
        passed to obj.set()

    Yields
    ------
    msg : Msg

    Returns
    -------
    statuses :
        Tuple of n statuses, one for each move operation

    See Also
    --------
    :func:`bluesky.plan_stubs.abs_set`
    :func:`bluesky.plan_stubs.mvr`
    """
    group = group or str(uuid.uuid4())
    status_objects = []

    cyl = reduce(operator.add, [cycler(obj, [val]) for obj, val in partition(2, args)])
    (step,) = merge_cycler(cyl)
    for obj, val in step.items():
        ret = yield Msg("set", obj, val, group=group, **kwargs)
        status_objects.append(ret)
    yield Msg("wait", None, group=group, timeout=timeout)
    return tuple(status_objects)


mov = mv  # synonym


@plan
def mvr(
    *args: Movable | Any, group: Hashable | None = None, timeout: float | None = None, **kwargs
) -> MsgGenerator[tuple[Status, ...]]:
    """
    Move one or more devices to a relative setpoint. Wait for all to complete.

    If more than one device is specified, the movements are done in parallel.

    Parameters
    ----------
    args :
        device1, value1, device2, value2, ...
    group : string, optional
        Used to mark these as a unit to be waited on.
    timeout : float, optional
        Specify a maximum time that the move(s) can be waited for.
    kwargs :
        passed to obj.set()

    Yields
    ------
    msg : Msg

    Returns
    -------
    statuses :
        Tuple of n statuses, one for each move operation

    See Also
    --------
    :func:`bluesky.plan_stubs.rel_set`
    :func:`bluesky.plan_stubs.mv`
    """
    objs = []
    for obj, val in partition(2, args):  # noqa: B007
        objs.append(obj)

    from .preprocessors import relative_set_decorator

    @relative_set_decorator(objs)
    def inner_mvr():
        return (yield from mv(*args, group=group, timeout=timeout, **kwargs))

    return (yield from inner_mvr())


movr = mvr  # synonym


@plan
def rd(obj: Readable, *, default_value: Any = 0) -> MsgGenerator[Any]:
    """Reads a single-value non-triggered object

    This is a helper plan to get the scalar value out of a Device
    (such as an EpicsMotor or a single EpicsSignal).

    For devices that implement the Locatable protocol, the location is canonical
    and is returned without parsing the read keys.

    For devices that have more than one read key the following rules are used:

    - if exactly 1 field is hinted that value is used
    - if no fields are hinted and there is exactly 1 value in the
      reading that value is used
    - if more than one field is hinted an Exception is raised
    - if no fields are hinted and there is more than one key in the reading an
      Exception is raised

    The devices is not triggered and this plan does not create any Events

    Parameters
    ----------
    obj : Device
        The device to be read

    default_value : Any
        The value to return when not running in a "live" RunEngine.
        This come ups when ::

           ret = yield Msg('read', obj)
           assert ret is None

        the plan is passed to `list` or some other iterator that
        repeatedly sends `None` into the plan to advance the
        generator.

    Returns
    -------
    val : Any or None
        The "single" value of the device

    """
    # Location is canonical if it exists
    if isinstance(obj, Locatable):
        location = yield Msg("locate", obj)
        if location is None:
            # list-ify mode
            return default_value
        else:
            return location["readback"]

    hints = get_hinted_fields(obj)
    if len(hints) > 1:
        msg = (
            f"Your object {obj} ({obj.name}.{getattr(obj, 'dotted_name', '')}) "
            f"has {len(hints)} items hinted ({hints}).  We do not know how to "
            "pick out a single value.  Please adjust the hinting by setting the "
            "kind of the components of this device or by reading one of its components"
        )
        raise ValueError(msg)
    elif len(hints) == 0:
        hint = None
        if hasattr(obj, "read_attrs"):
            if len(obj.read_attrs) != 1:
                msg = (
                    f"Your object {obj} ({obj.name}.{getattr(obj, 'dotted_name', '')}) "
                    f"and has {len(obj.read_attrs)} read attrs.  We do not know how to "
                    "pick out a single value.  Please adjust the hinting/read_attrs by "
                    "setting the kind of the components of this device or by reading one "
                    "of its components"
                )

                raise ValueError(msg)
    # len(hints) == 1
    else:
        (hint,) = hints

    ret = yield from read(obj)

    # list-ify mode
    if ret is None:
        return default_value

    if hint is not None:
        return ret[hint]["value"]  # type: ignore

    # handle the no hint 1 field case
    try:
        (data,) = ret.values()
    except ValueError as er:
        msg = (
            f"Your object {obj} ({obj.name}.{getattr(obj, 'dotted_name', '')}) "
            f"and has {len(ret)} read values.  We do not know how to pick out a "
            "single value.  Please adjust the hinting/read_attrs by setting the "
            "kind of the components of this device or by reading one of its components"
        )

        raise ValueError(msg) from er
    else:
        return data["value"]  # type: ignore


@plan
def stop(obj: Stoppable) -> MsgGenerator:
    """
    Stop a device.

    Parameters
    ----------
    obj : Device

    Yields
    ------
    msg : Msg
    """
    return (yield Msg("stop", obj))


@plan
def trigger(
    obj: Triggerable,
    *,
    group: Hashable | None = None,
    wait: bool = False,
) -> MsgGenerator[Status]:
    """
    Trigger and acquisition. Optionally, wait for it to complete.

    Parameters
    ----------
    obj : Device
    group : string (or any hashable object), optional
        identifier used by 'wait'; None by default
    wait : boolean, optional
        If True, wait for completion before processing any more messages.
        False by default.

    Yields
    ------
    msg : Msg

    Returns
    -------
    status :
        Status that completes when trigger is complete. If `wait` is True,
        this will always be complete by the time it is returned.

    """
    ret = yield Msg("trigger", obj, group=group)
    if wait:
        yield Msg("wait", None, group=group)
    return ret


@plan
def sleep(time: float) -> MsgGenerator:
    """
    Tell the RunEngine to sleep, while asynchronously doing other processing.

    This is not the same as ``import time; time.sleep()`` because it allows
    other actions, like interruptions, to be processed during the sleep.

    Parameters
    ----------
    time : float
        seconds

    Yields
    ------
    msg : Msg
        Msg('sleep', None, time)
    """
    return (yield Msg("sleep", None, time))


@plan
def wait(
    group: Hashable | None = None,
    *,
    timeout: float | None = None,
    error_on_timeout: bool = True,
    watch: Sequence[str] = (),
):
    """
    Wait for all statuses in a group to report being finished.

    Parameters
    ----------
    group : string (or any hashable object), optional
        Identifier given to `abs_set`, `rel_set`, `trigger`; None by default
    timeout : float, optional
        The maximum duration, in seconds, to wait for all objects in the group to complete.
        If the timeout expires and `error_on_timeout` is set to True, a TimeoutError is raised.

    error_on_timeout : bool, Defaults to True
        Specifies the behavior when the timeout is reached:
        - If True, a TimeoutError is raised if the operations do not complete within the specified timeout.
        - If False, the method returns once all objects are done.
    watch : set of watch groups, optional
        Additional groups to monitor while waiting for the primary group. Raises an exception if any watched group
        fails.
    Yields
    ------
    msg : Msg
        Msg('wait', None, group=group, error_on_timeout=error_on_timeout, timeout=timeout)
    """
    return (yield Msg("wait", None, group=group, error_on_timeout=error_on_timeout, timeout=timeout, watch=watch))


_wait = wait  # for internal references to avoid collision with 'wait' kwarg


@plan
def checkpoint() -> MsgGenerator:
    """
    If interrupted, rewind to this point.

    Yields
    ------
    msg : Msg
        Msg('checkpoint')

    See Also
    --------
    :func:`bluesky.plan_stubs.clear_checkpoint`
    """
    return (yield Msg("checkpoint"))


@plan
def clear_checkpoint() -> MsgGenerator:
    """
    Designate that it is not safe to resume. If interrupted or paused, abort.

    Yields
    ------
    msg : Msg
        Msg('clear_checkpoint')

    See Also
    --------
    :func:`bluesky.plan_stubs.checkpoint`
    """
    return (yield Msg("clear_checkpoint"))


@plan
def pause() -> MsgGenerator:
    """
    Pause and wait for the user to resume.

    Yields
    ------
    msg : Msg
        Msg('pause')

    See Also
    --------
    :func:`bluesky.plan_stubs.deferred_pause`
    :func:`bluesky.plan_stubs.sleep`
    """
    return (yield Msg("pause", None, defer=False))


@plan
def deferred_pause() -> MsgGenerator:
    """
    Pause at the next checkpoint.

    Yields
    ------
    msg : Msg
        Msg('pause', defer=True)

    See Also
    --------
    :func:`bluesky.plan_stubs.pause`
    :func:`bluesky.plan_stubs.sleep`
    """
    return (yield Msg("pause", None, defer=True))


@plan
def input_plan(prompt: str = "") -> MsgGenerator[str]:
    """
    Prompt the user for text input.

    Parameters
    ----------
    prompt : str
        prompt string, e.g., 'enter user name' or 'enter next position'

    Yields
    ------
    msg : Msg
        Msg('input', prompt=prompt)

    Returns
    -------
    input :
    """
    return (yield Msg("input", prompt=prompt))


@plan
def prepare(obj: Preparable, *args, group: Hashable | None = None, wait: bool = False, **kwargs):
    """
    Prepare a device ready for trigger or kickoff.

    Parameters
    ----------
    obj : Preparable
        Device with 'prepare' method
    group : string (or any hashable object), optional
        identifier used by 'wait'
    wait : boolean, optional
        If True, wait for completion before processing any more messages.
        False by default.
    kwargs
        passed through to ``obj.prepare()``

    Yields
    ------
    msg : Msg
        Msg('kickoff', obj)

    See Also
    --------
    :func:`bluesky.plan_stubs.complete`
    :func:`bluesky.plan_stubs.collect`
    :func:`bluesky.plan_stubs.wait`
    """
    ret = yield Msg("prepare", obj, *args, group=group, **kwargs)
    if wait:
        yield from _wait(group=group)
    return ret


@plan
def kickoff(
    obj: Flyable,
    *,
    group: Hashable | None = None,
    wait: bool = False,
    **kwargs,
) -> MsgGenerator[Status]:
    """
    Kickoff one fly-scanning device.

    Parameters
    ----------
    obj : fly-able Device with 'kickoff', and 'complete' methods.
    group : string (or any hashable object), optional
        identifier used by 'wait'.
    wait : boolean, optional
        If True, wait for completion before processing any more messages.
        False by default.
    kwargs
        passed through to ``obj.kickoff()``

    Yields
    ------
    msg : Msg
        Msg('kickoff', obj)

    Returns
    -------
    status :
        Status of kickoff operation. If `wait` is True,
        this will always be complete by the time it is returned.

    See Also
    --------
    :func:`bluesky.plan_stubs.complete`
    :func:`bluesky.plan_stubs.collect`
    :func:`bluesky.plan_stubs.wait`
    """
    ret = yield Msg("kickoff", obj, group=group, **kwargs)
    if wait:
        yield from _wait(group=group)
    return ret


@plan
def kickoff_all(*args, group: Hashable | None = None, wait: bool = True, **kwargs):
    """
    Kickoff one or more fly-scanning devices.

    Parameters
    ----------
    *args : Any fly-able
        Device with 'kickoff', and 'complete' methods.
    group : string (or any hashable object), optional
        identifier used by 'wait'.
    wait : boolean, optional
        If True, wait for completion before processing any more messages.
        True by default.
    kwargs
        passed through to 'kickoff' for each device

    Yields
    ------
    msg : Msg
        Msg('kickoff', obj)

    See Also
    --------
    :func:`bluesky.plan_stubs.complete`
    :func:`bluesky.plan_stubs.collect`
    :func:`bluesky.plan_stubs.wait`
    """
    objs = [check_supports(arg, Flyable) for arg in args]
    group = group or str(uuid.uuid4())
    statuses: list[Status] = []

    for obj in objs:
        ret = yield Msg("kickoff", obj, group=group, **kwargs)
        statuses.append(ret)
    if wait:
        yield from _wait(group=group)

    return tuple(statuses)


@plan
def complete(
    obj: Flyable,
    *,
    group: Hashable | None = None,
    wait: bool = False,
    **kwargs,
) -> MsgGenerator[Status]:
    """
    Tell a flyable, 'stop collecting, whenever you are ready'.

    A flyable returns a status object. Some flyers respond to this
    command by stopping collection and returning a finished status
    object immediately. Other flyers finish their given course and
    finish whenever they finish, irrespective of when this command is
    issued.

    Parameters
    ----------
    obj : fly-able
        Device with 'kickoff' and 'complete' methods.
    group : string (or any hashable object), optional
        identifier used by 'wait'
    wait : boolean, optional
        If True, wait for completion before processing any more messages.
        False by default.
    kwargs
        passed through to ``obj.complete()``

    Yields
    ------
    msg : Msg
        a 'complete' Msg and maybe a 'wait' message

    Returns
    -------
    status :
        Status of complete operation. If `wait` is True,
        this will always be complete by the time it is returned.

    See Also
    --------
    :func:`bluesky.plan_stubs.kickoff`
    :func:`bluesky.plan_stubs.collect`
    :func:`bluesky.plan_stubs.wait`
    """
    ret = yield Msg("complete", obj, group=group, **kwargs)
    if wait:
        yield from _wait(group=group)
    return ret


@plan
def complete_all(*args, group: Hashable | None = None, wait: bool = False, **kwargs):
    """
    Tell one or more flyable objects, 'stop collecting, whenever you are ready'.

    A flyable returns a status object. Some flyers respond to this
    command by stopping collection and returning a finished status
    object immediately. Other flyers finish their given course and
    finish whenever they finish, irrespective of when this command is
    issued.

    Parameters
    ----------
    *args : Any fly-able
        Device with 'kickoff' and 'complete' methods.
    group : string (or any hashable object), optional
        identifier used by 'wait'
    wait : boolean, optional
        If True, wait for completion before processing any more messages.
        False by default.
    kwargs
        passed through to 'complete' for each device

    Yields
    ------
    msg : Msg
        a 'complete' Msg and maybe a 'wait' message

    See Also
    --------
    :func:`bluesky.plan_stubs.kickoff`
    :func:`bluesky.plan_stubs.collect`
    :func:`bluesky.plan_stubs.wait`
    """
    objs = [check_supports(arg, Flyable) for arg in args]
    group = group or str(uuid.uuid4())
    statuses: list[Status] = []

    for obj in objs:
        ret = yield Msg("complete", obj, group=group, **kwargs)
        statuses.append(ret)
    if wait:
        yield from _wait(group=group)

    return tuple(statuses)


@plan
def collect(
    obj: Flyable, *args, stream: bool = False, return_payload: bool = True, name: str | None = None
) -> MsgGenerator[list[PartialEvent]]:
    """
    Collect data cached by one or more fly-scanning devices and emit documents.

    Parameters
    ----------
    obj : A device with 'kickoff', 'complete', and 'collect' methods.
    stream : boolean, optional
        If False (default), emit Event documents in one bulk dump. If True,
        emit events one at time.
    return_payload: boolean, optional
        If True (default), return the collected Events. If False, return None.
        Using ``stream=True`` and ``return_payload=False`` together avoids
        accumulating the documents in memory: they are emitted as they are
        collected, and they are not accumulated.
    name: str, optional
        If not None, will collect for the named string specifically, else collect will be performed
        on all streams.

    Yields
    ------
    msg : Msg
        Msg('collect', obj)

    See Also
    --------
    :func:`bluesky.plan_stubs.kickoff`
    :func:`bluesky.plan_stubs.complete`
    :func:`bluesky.plan_stubs.wait`
    """
    return (yield Msg("collect", obj, *args, stream=stream, return_payload=return_payload, name=name))


@plan
def collect_while_completing(flyers, dets, flush_period=None, stream_name=None, watch: Sequence[str] = ()):
    """
    Collect data from one or more fly-scanning devices and emit documents, then collect and emit
    data from one or more Collectable detectors until all are done.

    Parameters
    ----------
    flyers: An iterable sequence of fly-able devices with 'kickoff', 'complete' and
        'collect' methods.
    dets: An iterable sequence of collectable devices with 'describe_collect' method.
    flush_period: float, int
        Time period in seconds between each yield from collect while waiting for triggered
        objects to be done
    stream_name: str, optional
        If not None, will collect for the named string specifically, else collect will be performed
        on all streams.
    watch: set of watch groups, optional
        Additional groups to monitor while collecting from flyers.
    Yields
    ------
    msg : Msg
        A 'complete' message or 'collect' message

    See Also
    --------
    :func:`bluesky.plan_stubs.complete`
    :func:`bluesky.plan_stubs.collect`
    """
    group = short_uid(label="complete")
    yield from complete_all(*flyers, group=group, wait=False)
    done = False
    while not done:
        done = yield from wait(group=group, timeout=flush_period, error_on_timeout=False, watch=watch)
        yield from collect(*dets, name=stream_name)


@plan
def configure(
    obj: Configurable,
    *args,
    **kwargs,
) -> MsgGenerator[Mapping[str, Reading]]:
    """
    Change Device configuration and emit an updated Event Descriptor document.

    Parameters
    ----------
    obj : Device
    args
        passed through to ``obj.configure()``
    kwargs
        passed through to ``obj.configure()``

    Yields
    ------
    msg : Msg
        ``Msg('configure', obj, *args, **kwargs)``

    Returns
    -------
    configuration:
        Tuple of old and new configuration as returned by
        obj.read_configuration()
    """
    return (yield Msg("configure", obj, *args, **kwargs))


@plan
def stage(
    obj: Stageable,
    *,
    group: Hashable | None = None,
    wait: bool | None = None,
) -> MsgGenerator[Status | list[Any]]:
    """
    'Stage' a device (i.e., prepare it for use, 'arm' it).

    Parameters
    ----------
    obj : Device
    group : string (or any hashable object), optional
        identifier used by 'wait'; None by default
    wait : boolean, optional
        If True, wait for completion before processing any more messages.
        False by default.

    Yields
    ------
    msg : Msg

    Returns
    -------
    stage :
        Either a status representing the stage operation or a list of
        staged values for backward compatibility.

    See Also
    --------
    :func:`bluesky.plan_stubs.unstage`
    :func:`bluesky.plan_stubs.stage_all`
    """
    ret = yield Msg("stage", obj, group=group)
    old_style = not isinstance(ret, Status)
    if old_style:
        if (wait is None) or wait:
            # Old-style devices will just block. We do not need to explicitly wait.
            pass
        else:  # wait is False-y
            # No way to tell old-style devices not to wait
            raise RuntimeError(f"{obj}: Is an old style device and cannot be told not to wait")
    else:
        if wait:
            yield Msg("wait", None, group=group)
    return ret


@plan
def stage_all(
    *args: Stageable,
    group: Hashable | None = None,
) -> MsgGenerator[None]:
    """
    'Stage' one or more devices (i.e., prepare them for use, 'arm' them).

    Parameters
    ----------
    args :
        device1, device2, device3, ...
    group : string (or any hashable object), optional
        identifier used by 'wait'; None by default

    Yields
    ------
    msg : Msg

    See Also
    --------
    :func:`bluesky.plan_stubs.stage`
    :func:`bluesky.plan_stubs.unstage_all`
    """
    group = group or str(uuid.uuid4())
    status_objects = []

    for obj in args:
        ret = yield Msg("stage", obj, group=group)
        if isinstance(ret, Status):
            status_objects.append(ret)

    if status_objects:
        yield Msg("wait", None, group=group)


@plan
def unstage(
    obj: Stageable,
    *,
    group: Hashable | None = None,
    wait: bool | None = None,
) -> MsgGenerator[Status | list[Any]]:
    """
    'Unstage' a device (i.e., put it in standby, 'disarm' it).

    Parameters
    ----------
    obj : Device
    group : string (or any hashable object), optional
        identifier used by 'wait'; None by default
    wait : boolean, optional
        If True, wait for completion before processing any more messages.
        False by default.

    Yields
    ------
    msg : Msg

    Returns
    -------
    unstage :
        Either a status representing the stage operation or a list of
        staged values for backward compatibility.

    See Also
    --------
    :func:`bluesky.plan_stubs.stage`
    :func:`bluesky.plan_stubs.unstage_all`
    """
    ret = yield Msg("unstage", obj, group=group)
    old_style = not isinstance(ret, Status)
    if old_style:
        if (wait is None) or wait:
            # Old-style devices will just block. We do not need to explicitly wait.
            pass
        else:
            # No way to tell old-style devices not to wait
            raise RuntimeError(f"{obj}: Is an old style device and cannot be told not to wait")
    else:
        if wait:
            yield Msg("wait", None, group=group)
    return ret


@plan
def unstage_all(*args: Stageable, group: Hashable | None = None) -> MsgGenerator[None]:
    """
    'Unstage' one or more devices (i.e., put them in standby, 'disarm' them).

    Parameters
    ----------
    args :
        device1, device2, device3, ...
    group : string (or any hashable object), optional
        identifier used by 'wait'; None by default

    Yields
    ------
    msg : Msg

    See Also
    --------
    :func:`bluesky.plan_stubs.unstage`
    :func:`bluesky.plan_stubs.stage_all`
    """
    group = group or str(uuid.uuid4())
    status_objects = []

    for obj in args:
        ret = yield Msg("unstage", obj, group=group)
        if isinstance(ret, Status):
            status_objects.append(ret)

    if status_objects:
        yield Msg("wait", None, group=group)


@plan
def subscribe(name: str, func: Callable[[str, Mapping[str, Any]], None]) -> MsgGenerator[int]:
    """
    Subscribe the stream of emitted documents.

    Parameters
    ----------
    name : {'all', 'start', 'descriptor', 'event', 'stop'}
    func : callable
        Expected signature: ``f(name, doc)`` where ``name`` is one of the
        strings above ('all, 'start', ...) and ``doc`` is a dict

    Yields
    ------
    msg : Msg
        Msg('subscribe', None, func, name)

    Returns
    -------
    token :
        Unique identifier for a subscription

    See Also
    --------
    :func:`bluesky.plan_stubs.unsubscribe`
    """
    return (yield Msg("subscribe", None, func, name))


@plan
def unsubscribe(token: int) -> MsgGenerator:
    """
    Remove a subscription.

    Parameters
    ----------
    token : int
        token returned by processing a 'subscribe' message

    Yields
    ------
    msg : Msg
        Msg('unsubscribe', token=token)

    See Also
    --------
    :func:`bluesky.plan_stubs.subscribe`
    """
    return (yield Msg("unsubscribe", token=token))


@plan
def install_suspender(suspender: SuspenderBase) -> MsgGenerator:
    """
    Install a suspender during a plan.

    Parameters
    ----------
    suspender : :class:`bluesky.suspenders.SuspenderBase`
        The suspender to install

    Yields
    ------
    msg : Msg
        Msg('install_suspender', None, suspender)

    See Also
    --------
    :func:`bluesky.plan_stubs.remove_suspender`
    """
    return (yield Msg("install_suspender", None, suspender))


@plan
def remove_suspender(suspender: SuspenderBase) -> MsgGenerator:
    """
    Remove a suspender during a plan.

    Parameters
    ----------
    suspender : :class:`bluesky.suspenders.SuspenderBase`
        The suspender to remove

    Yields
    ------
    msg : Msg
        Msg('remove_suspender', None, suspender)

    See Also
    --------
    :func:`bluesky.plan_stubs.install_suspender`
    """
    return (yield Msg("remove_suspender", None, suspender))


@plan
def open_run(md: CustomPlanMetadata | None = None) -> MsgGenerator[str]:
    """
    Mark the beginning of a new 'run'. Emit a RunStart document.

    Parameters
    ----------
    md : dict, optional
        metadata

    Yields
    ------
    msg : Msg
        ``Msg('open_run', **md)``

    Returns
    -------
    uuid :
        Unique ID for the run

    See Also
    --------
    :func:`bluesky.plans_stubs.close_run`
    """
    return (yield Msg("open_run", **(md or {})))


@plan
def close_run(exit_status: str | None = None, reason: str | None = None) -> MsgGenerator[str]:
    """
    Mark the end of the current 'run'. Emit a RunStop document.

    Parameters
    ----------
    exit_status : {None, 'success', 'abort', 'fail'}
        The exit status to report in the Stop document
    reason : str, optional
        Long-form description of why the run ended

    Yields
    ------
    msg : Msg
        Msg('close_run')

    Returns
    -------
    uuid :
        Unique ID for the run

    See Also
    --------
    :func:`bluesky.plans_stubs.open_run`
    """
    return (yield Msg("close_run", exit_status=exit_status, reason=reason))


@plan
def wait_for(futures: Iterable[Callable[[], Awaitable[Any]]], **kwargs) -> MsgGenerator:
    """
    Low-level: wait for a list of ``asyncio.Future`` objects to set (complete).

    Parameters
    ----------
    futures : iterable
        iterable collection of coroutine functions that take no arguments
    kwargs
        passed through to ``asyncio.wait()``

    Yields
    ------
    msg : Msg
        ``Msg('wait_for', None, futures, **kwargs)``

    See Also
    --------
    :func:`bluesky.plan_stubs.wait`
    """
    return (yield Msg("wait_for", None, futures, **kwargs))


@plan
def trigger_and_read(devices: Sequence[Readable], name: str = "primary") -> MsgGenerator[Mapping[str, Reading]]:
    """
    Trigger and read a list of detectors and bundle readings into one Event.

    Parameters
    ----------
    devices : list
        devices to trigger (if they have a trigger method) and then read
    name : string, optional
        event stream name, a convenient human-friendly identifier; default
        name is 'primary'

    Returns
    -------
    readings:
        dict of device name to recorded information

    Yields
    ------
    msg : Msg
        messages to 'trigger', 'wait' and 'read'
    """
    from .preprocessors import contingency_wrapper

    # If devices is empty, don't emit 'create'/'save' messages.
    if not devices:
        yield from null()
    devices = separate_devices(devices)  # remove redundant entries
    rewindable = all_safe_rewind(devices)  # if devices can be re-triggered

    def inner_trigger_and_read():
        grp = _short_uid("trigger")
        no_wait = True
        for obj in devices:
            if isinstance(obj, Triggerable):
                no_wait = False
                yield from trigger(obj, group=grp)
        # Skip 'wait' if none of the devices implemented a trigger method.
        if not no_wait:
            yield from wait(group=grp)
        yield from create(name)

        def read_plan():
            ret = {}  # collect and return readings to give plan access to them
            for obj in devices:
                reading = yield from read(obj)
                if reading is not None:
                    ret.update(reading)
            return ret

        def standard_path():
            yield from save()

        def exception_path(exp):
            yield from drop()
            raise exp

        ret = yield from contingency_wrapper(read_plan(), except_plan=exception_path, else_plan=standard_path)
        return ret

    from .preprocessors import rewindable_wrapper

    return (yield from rewindable_wrapper(inner_trigger_and_read(), rewindable))


@plan
def broadcast_msg(
    command: str,
    objs: Iterable[Any],
    *args,
    **kwargs,
) -> MsgGenerator[Any]:
    """
    Generate many copies of a message, applying it to a list of devices.

    Parameters
    ----------
    command : string
    objs : iterable
    ``*args``
        args for message
    ``**kwargs``
        kwargs for message

    Yields
    ------
    msg : Msg

    Returns
    -------
    any : out from RunEngine, if any
    """
    return_vals = []
    for o in objs:
        ret = yield Msg(command, o, *args, **kwargs)
        return_vals.append(ret)

    return return_vals


@plan
def repeater(
    n: int | None,
    gen_func: Callable[..., MsgGenerator],
    *args,
    **kwargs,
) -> MsgGenerator[None]:
    """
    Generate n chained copies of the messages from gen_func

    Parameters
    ----------
    n : int or None
        total number of repetitions; if None, infinite
    gen_func : callable
        returns generator instance
    ``*args``
        args for gen_func
    ``**kwargs``
        kwargs for gen_func

    Yields
    ------
    msg : Msg

    See Also
    --------
    :func:`bluesky.plan_stubs.caching_repeater`
    """
    it: Any
    it = range
    if n is None:
        n = 0
        it = itertools.count

    for j in it(n):  # noqa: B007
        yield from gen_func(*args, **kwargs)


@plan
def caching_repeater(n: int | None, plan: MsgGenerator) -> MsgGenerator[None]:
    """
    Generate n chained copies of the messages in a plan.

    This is different from ``repeater`` above because it takes in a
    generator or iterator, not a function that returns one.

    Parameters
    ----------
    n : int or None
        total number of repetitions; if None, infinite
    plan : iterable

    Yields
    ------
    msg : Msg

    See Also
    --------
    :func:`bluesky.plan_stubs.repeater`
    """
    warnings.warn("The caching_repeater will be removed in a future version of bluesky.", stacklevel=2)
    gen: Any
    if n is None:
        gen = itertools.count(0)
    else:
        gen = range(n)

    lst_plan = list(plan)
    for _ in gen:
        yield from (m for m in lst_plan)


@plan
def one_shot(detectors: Sequence[Readable], take_reading: TakeReading | None = None) -> MsgGenerator[None]:
    """Inner loop of a count.

    This is the default function for ``per_shot`` in count plans.

    Parameters
    ----------
    detectors : Sequence[OphydObj]
        devices to read

    take_reading : plan, optional
        function to do the actual acquisition ::

           def take_reading(dets, name='primary'):
                yield from ...

        Callable[List[OphydObj], Optional[str]] -> Generator[Msg], optional

        Defaults to `trigger_and_read`

    Yields
    ------
    msg : Msg
    """
    take_reading = trigger_and_read if take_reading is None else take_reading
    yield Msg("checkpoint")
    yield from take_reading(list(detectors))  # type: ignore  # Movable issue


@plan
def one_1d_step(
    detectors: Sequence[Readable],
    motor: Movable,
    step: Any,
    take_reading: TakeReading | None = None,
) -> MsgGenerator[Mapping[str, Reading]]:
    """
    Inner loop of a 1D step scan

    This is the default function for ``per_step`` param in 1D plans.

    Parameters
    ----------
    detectors : list or tuple
        devices to read
    motor : Settable
        The motor to move
    step : Any
        Where to move the motor to
    take_reading : plan, optional
        function to do the actual acquisition ::

           def take_reading(dets, name='primary'):
                yield from ...

        Callable[List[OphydObj], Optional[str]] -> Generator[Msg], optional

        Defaults to `trigger_and_read`

    Yields
    ------
    msg : Msg

    Returns
    -------
    readings :
        dict of device names to recorded information
    """
    take_reading = trigger_and_read if take_reading is None else take_reading

    def move():
        grp = _short_uid("set")
        yield Msg("checkpoint")
        yield Msg("set", motor, step, group=grp)
        yield Msg("wait", None, group=grp)

    yield from move()
    return (yield from take_reading(list(detectors) + [motor]))  # type: ignore


@plan
def move_per_step(step: Mapping[Movable, Any], pos_cache: dict[Movable, Any]) -> MsgGenerator[None]:
    """
    Inner loop of an N-dimensional step scan without any readings

    This can be used as a building block for custom ``per_step`` stubs.

    Parameters
    ----------
    step : dict
        mapping motors to positions in this step
    pos_cache : dict
        mapping motors to their last-set positions

    Yields
    ------
    msg : Msg
    """
    yield Msg("checkpoint")
    grp = _short_uid("set")
    for motor, pos in step.items():
        if pos == pos_cache[motor]:
            # This step does not move this motor.
            continue
        yield Msg("set", motor, pos, group=grp)
        pos_cache[motor] = pos
    yield Msg("wait", None, group=grp)


@plan
def one_nd_step(
    detectors: Sequence[Readable],
    step: Mapping[Movable, Any],
    pos_cache: dict[Movable, Any],
    take_reading: TakeReading | None = None,
) -> MsgGenerator[None]:
    """
    Inner loop of an N-dimensional step scan

    This is the default function for ``per_step`` param`` in ND plans.

    Parameters
    ----------
    detectors : list or tuple
        devices to read
    step : dict
        mapping motors to positions in this step
    pos_cache : dict
        mapping motors to their last-set positions
    take_reading : plan, optional
        function to do the actual acquisition ::

           def take_reading(dets, name='primary'):
                yield from ...

        Callable[List[OphydObj], Optional[str]] -> Generator[Msg], optional

        Defaults to `trigger_and_read`

    Yields
    ------
    msg : Msg
    """
    take_reading = trigger_and_read if take_reading is None else take_reading
    motors = step.keys()
    yield from move_per_step(step, pos_cache)
    yield from take_reading(list(detectors) + list(motors))  # type: ignore  # Movable issue


@plan
def repeat(
    plan: Callable[[], MsgGenerator],
    num: int | None = 1,
    delay: ScalarOrIterableFloat = 0.0,
) -> MsgGenerator[Any]:
    """
    Repeat a plan num times with delay and checkpoint between each repeat.

    This is different from ``repeater`` and ``caching_repeater`` in that it
    adds ``checkpoint`` and optionally ``sleep`` messages if delay is provided.
    This is intended for users who need the structure of ``count`` but do not
    want to reimplement the control flow.

    Parameters
    ----------
    plan: callable
        Callable that returns an iterable of Msg objects
    num : integer, optional
        number of readings to take; default is 1

        If None, capture data until canceled
    delay : iterable or scalar, optional
        time delay between successive readings; default is 0

    Yields
    ------
    msg : Msg

    Returns
    -------
    any : output of original plan

    Notes
    -----
    If ``delay`` is an iterable, it must have at least ``num - 1`` entries or
    the plan will raise a ``ValueError`` during iteration.
    """
    # Create finite or infinite counter
    iterator: Iterable
    if num is None:
        iterator = itertools.count()
    else:
        iterator = range(num)

    # If delay is a scalar, repeat it forever. If it is an iterable, leave it.
    if not isinstance(delay, Iterable):
        delay = itertools.repeat(delay)
    else:
        try:
            num_delays = len(delay)  # type: ignore
        except TypeError:
            # No way to tell in advance if we have enough delays.
            pass
        else:
            if num and num - 1 > num_delays:
                raise ValueError("num=%r but delays only provides %r entries" % (num, num_delays))  # noqa: UP031
        delay = iter(delay)

    def repeated_plan():
        for i in iterator:
            now = time.time()  # Intercept the flow in its earliest moment.
            yield Msg("checkpoint")
            yield from ensure_generator(plan())
            try:
                d = next(delay)
            except StopIteration as stop:
                if i + 1 == num:
                    break
                elif num is None:
                    break
                else:
                    # num specifies a number of iterations less than delay
                    raise ValueError("num=%r but delays only provides %r entries" % (num, i)) from stop  # noqa: UP031
            if d is not None:
                d = d - (time.time() - now)
                if d > 0:  # Sleep if and only if time is left to do it.
                    yield Msg("sleep", None, d)

    return (yield from repeated_plan())
