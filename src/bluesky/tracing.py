import functools
from collections.abc import Callable
from typing import cast

from opentelemetry.trace import Tracer, get_tracer

from .protocols import P
from .utils import MsgGenerator

tracer = get_tracer(__name__)


def trace_plan(tracer: Tracer, span_name: str) -> Callable[[Callable[P, MsgGenerator]], Callable[P, MsgGenerator]]:
    """Wraps a generator function in tracer.start_as_current_span(span_name)"""

    def wrap(f: Callable[P, MsgGenerator]) -> Callable[P, MsgGenerator]:
        @functools.wraps(f)
        def wrap_f(*args: P.args, **kwargs: P.kwargs):
            with tracer.start_as_current_span(span_name):
                yield from f(*args, **kwargs)

        return cast(Callable[P, MsgGenerator], wrap_f)

    return wrap
