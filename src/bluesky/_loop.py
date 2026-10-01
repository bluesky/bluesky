"""Crossing from a foreign thread onto the event loop."""

from __future__ import annotations

import asyncio
import concurrent.futures
import typing
from collections.abc import Awaitable, Callable
from inspect import iscoroutine

T = typing.TypeVar("T")


def running_on(loop: asyncio.AbstractEventLoop) -> bool:
    """Whether this thread is the one running ``loop``."""
    try:
        return asyncio.get_running_loop() is loop
    except RuntimeError:
        return False


def run_coro_on_loop(coro: Awaitable[T], loop: asyncio.AbstractEventLoop, *, timeout: float | None = None) -> T:
    """Run ``coro`` on ``loop`` from another thread and return its result.

    Raises what ``coro`` raises, and raises RuntimeError if called on ``loop``.
    """
    if running_on(loop):
        if iscoroutine(coro):
            # Close it, or it warns "never awaited" at collection.
            coro.close()
        raise RuntimeError(
            "This was called from the event loop it waits for, which cannot work. "
            "Loop-side code -- a subscriber, a msg_hook, or a plan -- reaches the "
            "runner directly instead."
        )
    future: concurrent.futures.Future[T] = asyncio.run_coroutine_threadsafe(coro, loop=loop)  # type: ignore[arg-type]
    try:
        return future.result(timeout=timeout)
    except BaseException:
        # Cancel on timeout or KeyboardInterrupt (a hard pause), so the task
        # does not outlive the wait. A no-op if already done.
        future.cancel()
        raise


def call_soon_or_now(loop: asyncio.AbstractEventLoop, func: Callable[..., typing.Any], *args: typing.Any) -> None:
    """Call ``func(*args)`` now on ``loop``'s thread, else schedule it there."""
    if running_on(loop):
        func(*args)
    else:
        try:
            loop.call_soon_threadsafe(func, *args)
        except RuntimeError:
            # Loop closed during teardown: drop it rather than raise into a device.
            pass
