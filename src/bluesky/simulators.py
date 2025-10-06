from collections.abc import Callable, Generator, Sequence
from itertools import dropwhile
from time import time
from typing import (
    Any,
    Literal,
    cast,
)
from warnings import warn

from bluesky.log import logger as LOGGER
from bluesky.preprocessors import print_summary_wrapper
from bluesky.run_engine import call_in_bluesky_event_loop, in_bluesky_event_loop
from bluesky.utils import Msg, maybe_await

from .protocols import Checkable, Readable, Reading

END = "end"


def plot_raster_path(plan, x_motor, y_motor, ax=None, probe_size=None, lw=2):
    """Plot the raster path for this plan

    Parameters
    ----------
    plan : iterable
       Must yield `Msg` objects and not be a co-routine

    x_motor, y_motor : str
       Names of the x and y motors

    ax : matplotlib.axes.Axes
       The axes to plot to, if none, make new figure + axes

    probe_size : float, optional
       If not None, use as radius of probe (in same units as motor positions)

    lw : float, optional
        Width of lines drawn between points
    """
    import matplotlib.pyplot as plt
    from matplotlib import collections as mcollections
    from matplotlib import patches as mpatches

    if ax is None:
        ax = plt.subplots()[1]
    ax.set_aspect("equal")

    cur_x = cur_y = None
    traj = []
    for msg in plan:
        cmd = msg.command
        if cmd == "set":
            if msg.obj.name == x_motor:
                cur_x = msg.args[0]
            if msg.obj.name == y_motor:
                cur_y = msg.args[0]
        elif cmd == "save":
            traj.append((cur_x, cur_y))

    x, y = zip(*traj)
    (path,) = ax.plot(x, y, marker="", linestyle="-", lw=lw)
    ax.set_xlabel(x_motor)
    ax.set_ylabel(y_motor)
    if probe_size is None:
        read_points = ax.scatter(x, y, marker="o", lw=lw)
    else:
        circles = [mpatches.Circle((_x, _y), probe_size, facecolor="black", alpha=0.5) for _x, _y in traj]

        read_points = mcollections.PatchCollection(circles, match_original=True)
        ax.add_collection(read_points)
    return {"path": path, "events": read_points}


def summarize_plan(plan):
    """Print summary of plan

    Prints a minimal version of the plan, showing only moves and
    where events are created.

    Parameters
    ----------
    plan : iterable
        Must yield `Msg` objects
    """
    for msg in print_summary_wrapper(plan):  # noqa: B007
        ...


print_summary = summarize_plan  # back-compat


def check_limits(plan):
    """Run check_limits_async in the RE"""
    if in_bluesky_event_loop():
        raise RuntimeError("Can't call check_limits() from within RE, use await check_limits_async() instead")
    call_in_bluesky_event_loop(check_limits_async(plan))


async def check_limits_async(plan):
    """
    Check that a plan will not move devices outside of their limits.

    Parameters
    ----------
    plan : iterable
        Must yield `Msg` objects
    """
    ignore = []
    for msg in plan:
        obj = msg.obj
        if msg.command == "set" and obj not in ignore:
            if isinstance(obj, Checkable):
                await maybe_await(obj.check_value(msg.args[0]))
            else:
                warn(  # noqa: B028
                    f"{obj.name} has no check_value() method to check if {msg.args[0]} is within its limits."
                )
                ignore.append(obj)


class RunEngineSimulator:
    """Helps test a Bluesky plan by recording bluesky messages and
     injecting responses according to the bluesky Message Protocol.

    See bluesky docs for details of the message protocol.

    Basic usage consists of
    1) Registering various handlers to respond to anticipated messages in the experiment plan and fire
     any needed callbacks.
    2) Calling simulate_plan()
    3) Examining the returned message list and making asserts against them.

    Attributes
    ----------
    return_value
        The return value of the most recently executed plan.

    Examples
    --------

    >>> def my_plan():
    ...     pass

    >>> sim = RunEngineSimulator()
    >>> messages = sim.simulate_plan(my_plan())
    >>>
    """

    GROUP_ANY = "any"

    def __init__(self) -> None:
        self.message_handlers: list[_MessageHandler] = []
        self.callbacks: dict[int, tuple[Callable[[str, dict], None], str]] = {}
        self.next_callback_token: int = 0
        self.return_value: Any = None

    def add_handler_for_callback_subscribes(self):
        """Add a handler that registers all the callbacks from subscribe messages so we can call them later.

        You will need to call this as one of the first things if you wish to fire callbacks from the simulator.
        """
        self.message_handlers.append(
            _MessageHandler(
                lambda msg: msg.command == "subscribe",
                lambda msg: self._add_callback(msg.args),
            )
        )

    def add_handler(
        self,
        commands: str | Sequence[str],
        handler: Callable[[Msg], object],
        msg_filter: str | Callable[[Msg], bool] | None = None,
        index: int | Literal["end"] = 0,
    ):
        """Add the specified handler for a particular message.

        Parameters
        ----------
        commands
            The command name for the message as defined in bluesky Message Protocol, or a sequence if
            more than one matches.
        handler
            A lambda that accepts a Msg and returns an object; the object is sent to the current yield
            statement in the generator, and is used when reading values from devices, the structure of the
            object depends on device hinting.
        msg_filter
            This can either be a string corresponding to the Msg obj.name property to match, or it can
            be a predicate on the Msg which returns a bool. The default is None which will match all
            messages.
        index
            An optional integer indicating where to insert the handler, the default is 0 which
            prepends it so that newer handlers override older ones. Specify END to append the handler.

        """
        if isinstance(commands, str):
            commands = [commands]

        self.message_handlers.insert(
            cast(int, index if index != END else len(self.message_handlers)),
            _MessageHandler(
                lambda msg: msg.command in commands
                and (
                    msg_filter is None
                    or (callable(msg_filter) and msg_filter(msg))
                    or (msg.obj and msg.obj.name == msg_filter)
                ),
                handler,
            ),
        )

    def add_read_handler_for(self, obj: Readable, value: Any | None):
        """
        Convenience method to register a handler to return a result from a
        single-valued 'read' command.
        The behaviour is equivalent to
        add_read_handler_for_multiple(obj, **{obj.name: value})

        Examples
        --------
        In order to exercise a plan containing the following:

        >>> def trigger_and_return_pin_tip():
        ...     yield from bps.trigger(pin_tip)
        ...     tip_pos = yield from bps.rd(pin_tip.triggered_tip)
        ...     return tip_pos

        >>> sim = RunEngineSimulator()
        >>> sim.add_read_handler("pin_tip_detection-triggered_tip", (8, 5))
        >>> msgs = sim.simulate_plan(trigger_and_return_pin_tip())
        >>> assert sim.return_value == (8, 5)
        """
        self.add_read_handler_for_multiple(obj, **{obj.name: value})

    def add_read_handler_for_multiple(self, obj: Readable, **kwargs):
        """
        Convenience method to register a handler to return a result from a multi-valued
        'read' command.

        Parameters
        ----------
        obj
           The object to intercept reads for
        kwargs
            The handler will return a dict with the corresponding key-value pairs.
            If the value is a dict the result of the read command for the key is the supplied dict.
            Otherwise, the value of the read command for the key is
            a value of Reading(value=value, timestamp=time())

        Examples
        --------
        In order to exercise a plan containing the following:

        >>> def read_all_values():
        >>>     data = yield from bps.read(hw.ab_det)
        >>>     return data
        >>> sim = RunEngineSimulator()
        >>> sim.add_read_handler_for_multiple(hw.ab_det, a=11, b=12)
        >>> sim.simulate_plan(read_all_values())
        >>> assert sim.return_value["a"]["value"] == 11
        >>> assert sim.return_value["b"]["value"] == 12
        >>> assert isclose(sim.return_value["a"]["timestamp"], time(), abs_tol=1)
        >>> assert sim.return_value["b"]["timestamp"] == 1717071719
        """

        def handle_read(_):
            return {k: v if isinstance(v, dict) else Reading(value=v, timestamp=time()) for k, v in kwargs.items()}

        self.add_handler("read", handle_read, obj.name)

    def add_wait_handler(self, handler: Callable[[Msg], None], group: str = GROUP_ANY) -> None:
        """Add a wait handler for a particular message.

        Parameters
        ----------
        handler
            a lambda that accepts a Msg, use this to execute any code that simulates something that's
            supposed to complete when a group finishes
        group
            name of the group to wait for, default is GROUP_ANY which matches them all

        Examples
        --------
        In order to exercise the following plan:
        >>> def close_shutter():
        ...     yield from bps.abs_set(detector_motion.shutter, 1, group="my_group")
        ...     yield from bps.wait("my_group")
        ...     done = yield from bps.rd("detector_motion_shutter")
        ...     assert done == 1

        >>> sim = RunEngineSimulator()
        ... def simulate_detector_motion():
        ...     sim.add_read_handler("detector_motion_shutter", 1)
        ...
        ... sim.add_wait_handler(simulate_detector_motion, "my_group")
        ... sim.simulate_plan(close_shutter())
        """
        self.add_handler(
            "wait",
            handler,
            lambda msg: (group == RunEngineSimulator.GROUP_ANY or msg.kwargs["group"] == group),
        )

    def add_callback_handler_for(
        self,
        command: str,
        document_name: str,
        document: dict,
        msg_filter: Callable[[Msg], bool] | None = None,
    ):
        """Add a handler to fire a callback when a matching command is encountered.
        Equivalent to add_callback_for_multiple(command, [[(document_name, document)]], msg_filter)
        Parameters
        ----------
        command
            Name of the command to match.
        document_name
            Name of the document to match when used in single shot mode.
        document
            The document to fire in single shot mode.
        msg_filter
            Optional predicate on the message which triggers the callback
        """
        self.add_callback_handler_for_multiple(command, [[(document_name, document)]], msg_filter)

    def add_callback_handler_for_multiple(
        self,
        command: str,
        docs: Sequence[Sequence[tuple[str, dict]]],
        msg_filter: Callable[[Msg], bool] | None = None,
    ):
        """Add a handler to fire callbacks in sequence when a matching command is encountered.

        Parameters
        ----------
        command
            Name of the command to match.
        docs
            Sequence of Sequence of document_name, document tuples
            On each receipt of a matching message, the sequence contained in the next element of `docs`
            will be iterated over and each (document_name, document) pair will be fired to subscribed callbacks
            in order.
        msg_filter
            Optional predicate on the message which triggers the callback
        """
        it = iter(docs)

        def handle_command(_):
            for name, doc in next(it):
                self.fire_callback(name, doc)

        self.add_handler(command, handle_command, msg_filter)

    def fire_callback(self, document_name, document) -> None:
        """Fire all the callbacks registered for this document type in order to simulate something happening.

        Parameters
        ----------
        document_name
            document name as defined in the Bluesky Message Protocol 'subscribe' call,
            all subscribers filtering on this document name will be called
        document
            the document to send
        """
        for callback_func, callback_docname in self.callbacks.values():
            if callback_docname == "all" or callback_docname == document_name:
                callback_func(document_name, document)

    def simulate_plan(self, gen: Generator[Msg, Any, Any]) -> list[Msg]:
        """Simulate the RunEngine executing the plan.

        After executing the plan return_value is populated with the return value of the plan.

        Parameters
        ----------
        gen
            the generator function that executes the plan

        Returns
        -------
        list[Msg]
            list of the messages generated by the plan
        """
        messages = []
        send_value = None
        try:
            while msg := gen.send(send_value):
                send_value = None
                messages.append(msg)
                LOGGER.debug("<%s", msg)
                if handler := next((h for h in self.message_handlers if h.predicate(msg)), None):
                    send_value = handler.runnable(msg)

                if send_value:
                    LOGGER.debug(f">send {send_value}")
        except StopIteration as e:
            self.return_value = e.value
        return messages

    def _add_callback(self, msg_args):
        self.callbacks[self.next_callback_token] = msg_args
        self.next_callback_token += 1


def assert_message_and_return_remaining(
    messages: list[Msg],
    predicate: Callable[[Msg], bool],
    group: str | None = None,
):
    """Find the next message matching the predicate, assert that we found it.

    Returns
    -------
    list[Msg]
        All the remaining messages starting from the matched message which is included as a
        convenience to capture information that may be used in subsequent calls.
    """

    def not_matching(msg: Msg) -> bool:
        return (group is not None and (msg.kwargs and msg.kwargs.get("group") != group)) or not predicate(msg)

    matched = list(dropwhile(not_matching, messages))
    assert matched, f"Nothing matched predicate {predicate}"
    return matched


class _MessageHandler:
    def __init__(self, p: Callable[[Msg], bool], r: Callable[[Msg], object]):
        self.predicate = p
        self.runnable = r
