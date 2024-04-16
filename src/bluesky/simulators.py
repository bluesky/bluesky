from collections.abc import Sequence, Callable, Generator
from typing import Optional, Any
from warnings import warn

from bluesky.preprocessors import print_summary_wrapper
from bluesky.run_engine import call_in_bluesky_event_loop, in_bluesky_event_loop
from bluesky.utils import maybe_await, Msg

from .protocols import Checkable

from bluesky.log import logger as LOGGER

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
                warn(f"{obj.name} has no check_value() method" f" to check if {msg.args[0]} is within its limits.")  # noqa: B028
                ignore.append(obj)


class RunEngineSimulator:
    """This class facilitates testing of a Bluesky plan by recording bluesky messages and
     injecting responses according to the bluesky Message Protocol (see bluesky docs for details).

    Basic usage consists of
    1) Registering various handlers to respond to anticipated messages in the experiment plan and fire any
    needed callbacks.
    2) Calling simulate_plan()
    3) Examining the returned message list and making asserts against them

    An example usage:
    >>> def my_plan():
    ...     pass

    >>> sim = RunEngineSimulator()
    >>> messages = sim.simulate_plan(my_plan())
    >>>


    """

    GROUP_ANY = "any"

    def __init__(self):
        self.message_handlers = []
        self.callbacks = {}
        self.next_callback_token = 0
        self.return_value: Any

    def add_handler_for_callback_subscribes(self):
        """Add a handler that registers all the callbacks from subscribe messages so we can call them later.
        You probably want to call this as one of the first things unless you have a good reason not to.
        """
        self.message_handlers.append(
            MessageHandler(
                lambda msg: msg.command == "subscribe",
                lambda msg: self._add_callback(msg.args),
            )
        )

    def add_handler(self, handler: Callable[[Msg], object], commands: str | Sequence[str], obj_name: Optional[str] = None):
        """Add the specified handler for a particular message. The handler is prepended so that
        newer handlers override older ones.
        Args:
            handler: a lambda that accepts a Msg and returns an object; the object is sent to the current yield statement
            in the generator, and is used when reading values from devices, the structure of the object depends on device
            hinting.
            commands: the command name for the message as defined in bluesky Message Protocol, or a sequence if more
            than one matches
            obj_name: the name property of the obj to match, can be None (the default) as not all messages have a name
        """
        if isinstance(commands, str):
            commands = [commands]

        self.message_handlers.insert(0,
                                     MessageHandler(
                                         lambda msg: msg.command in commands
                                                     and (obj_name is None or (msg.obj and msg.obj.name == obj_name)),
                                         handler,
                                     )
                                     )

    def add_read_handler(self, obj_name, value, dict_key="values"):
        """
        Convenience method to register a handler to return a result from a 'read' command.
        Args:
             value: The value to return
             dict_key: The key in the result dictionary to populate with the value, by default this is 'values'

        Example:
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

        self.add_handler(lambda _: {dict_key: {"value": value}}, "read", obj_name)

    def add_wait_handler(
            self, handler: Callable[[Msg], None], group: str = GROUP_ANY
    ) -> None:
        """Add a wait handler for a particular message
        Args:
            handler: a lambda that accepts a Msg, use this to execute any code that simulates something that's
            supposed to complete when a group finishes
            group: name of the group to wait for, default is GROUP_ANY which matches them all

        Example:
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
        self.message_handlers.append(
            MessageHandler(
                lambda msg: msg.command == "wait"
                            and (group == RunEngineSimulator.GROUP_ANY or msg.kwargs["group"] == group),
                handler,
            )
        )

    def fire_callback(self, document_name, document) -> None:
        """Fire all the callbacks registered for this document type in order to simulate something happening
        Args:
             document_name: document name as defined in the Bluesky Message Protocol 'subscribe' call,
             all subscribers filtering on this document name will be called
             document: the document to send
        """
        for callback_func, callback_docname in self.callbacks.values():
            if callback_docname == "all" or callback_docname == document_name:
                callback_func(document_name, document)

    def simulate_plan(self, gen: Generator[Msg, object, object]) -> list[Msg]:
        """Simulate the RunEngine executing the plan
        Args:
            gen: the generator function that executes the plan
        Returns:
            a list of the messages generated by the plan
        Postcondition:
            return_value is populated with the return value of the plan
        """
        messages = []
        send_value = None
        try:
            while msg := gen.send(send_value):
                send_value = None
                messages.append(msg)
                LOGGER.debug(f"<{msg}")
                if handler := next(
                        (h for h in self.message_handlers if h.predicate(msg)), None
                ):
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
    self,
    messages: list[Msg],
    predicate: Callable[[Msg], bool],
    group: Optional[str] = None,
):
    """Find the next message matching the predicate, assert that we found it
    Return: all the remaining messages starting from the matched message"""
    indices = [
        i
        for i in range(len(messages))
        if (
            not group
            or (messages[i].kwargs and messages[i].kwargs.get("group") == group)
        )
        and predicate(messages[i])
    ]
    assert indices, f"Nothing matched predicate {predicate}"
    return messages[indices[0] :]


class MessageHandler:
    """"""
    def __init__(self, p: Callable[[Msg], bool], r: Callable[[Msg], object]):
        self.predicate = p
        self.runnable = r