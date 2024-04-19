import datetime
import sys
import types
import warnings
from collections import namedtuple
from typing import (
    Any,
    AsyncIterable,
    Dict,
    Optional,
    Tuple,
    Type,
    Union,
)

from bluesky.protocols import Asset, StreamAsset, WritesExternalAssets, WritesStreamAssets
from bluesky.utils.helper_functions import iterate_maybe_async


class Msg(namedtuple("Msg_base", ["command", "obj", "args", "kwargs", "run"])):
    """Namedtuple sub-class to encapsulate a message from the plan to the RE.

    This class provides 3 key features:

    1. dot access to the contents
    2. default values and a variadic signature for args / kwargs
    3. a nice repr
    """

    __slots__ = ()

    def __new__(cls, command, obj=None, *args, run=None, **kwargs):
        return super(Msg, cls).__new__(  # noqa: UP008
            cls, command, obj, args, kwargs, run
        )

    def __repr__(self):
        return (
            f"Msg({self.command!r}, obj={self.obj!r}, "
            f"args={self.args}, kwargs={self.kwargs}, run={self.run!r})"
        )


PLAN_TYPES: Tuple[Type, ...] = (types.GeneratorType,)
try:
    from types import CoroutineType
except ImportError:
    # < py35
    pass
else:
    PLAN_TYPES = PLAN_TYPES + (CoroutineType,)
    del CoroutineType


def ensure_generator(plan):
    """
    Ensure that the input is a generator.

    Parameters
    ----------
    plan : iterable or iterator

    Returns
    -------
    gen : coroutine
    """
    if isinstance(plan, Msg):
        return single_gen(plan)
    gen = iter(plan)  # no-op on generators; needed for classes
    if not isinstance(gen, PLAN_TYPES):
        # If plan does not support .send, we must wrap it in a generator.
        gen = (msg for msg in gen)

    return gen


def single_gen(msg):
    """Turn a single message into a plan

    If ``lambda x: yield x`` were valid Python, this would be equivalent.
    In Python 3.6 or 3.7 we might get lambda generators.

    Parameters
    ----------
    msg : Msg
        a single message

    Yields
    ------
    msg : Msg
        the input message
    """
    return (yield msg)


def ts_msg_hook(msg, file=sys.stdout):
    t = f"{datetime.datetime.now():%H:%M:%S.%f}"
    msg_fmt = "{: <17s} -> {!s: <15s} args: {}, kwargs: {}, run: {}".format(
        msg.command,
        msg.obj.name if hasattr(msg.obj, "name") else msg.obj,
        msg.args,
        msg.kwargs,
        f"'{msg.run}'" if isinstance(msg.run, str) else msg.run,
    )
    print(f"{t} {msg_fmt}", file=file)


already_warned: Dict[Any, bool] = {}


def warn_if_msg_args_or_kwargs(msg, meth, args, kwargs):
    if args or kwargs and not already_warned.get(msg.command):
        already_warned[msg.command] = True
        error_msg = f"""\
About to call {meth.__name__}() with args {args} and kwargs {kwargs}.
In the future the passing of Msg.args and Msg.kwargs down to hardware from
Msg("{msg.command}") may be deprecated. If you have a use case for these,
we would like to know about it, so please open an issue at
https://github.com/bluesky/bluesky/issues"""
        warnings.warn(error_msg)  # noqa: B028


async def maybe_collect_asset_docs(
    msg, obj, index: Optional[int] = None, *args, **kwargs
) -> AsyncIterable[Union[Asset, StreamAsset]]:
    # The if/elif statement must be done in this order because isinstance for protocol
    # doesn't check for exclusive signatures, and WritesExternalAssets will also
    # return true for a WritesStreamAsset as they both contain collect_asset_docs
    if isinstance(obj, WritesStreamAssets):
        warn_if_msg_args_or_kwargs(msg, obj.collect_asset_docs, args, kwargs)
        async for stream_doc in iterate_maybe_async(obj.collect_asset_docs(index, *args, **kwargs)):
            yield stream_doc
    elif isinstance(obj, WritesExternalAssets):
        warn_if_msg_args_or_kwargs(msg, obj.collect_asset_docs, args, kwargs)
        async for doc in iterate_maybe_async(obj.collect_asset_docs(*args, **kwargs)):
            yield doc
