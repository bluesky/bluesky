try:
    from typing import Protocol, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, runtime_checkable

from typing import Dict, List, Any, TypeVar, Tuple, Optional, Callable, Generator
from collections.abc import MutableSequence, Iterable

A, B = TypeVar('A'), TypeVar('B')

class OrderedDictType(Dict[A, B]):
    ...

Configuration = OrderedDictType[str, Dict[str, Any]]

class StatusWatchCallable(Protocol):
    def __call__(
        self,
        name: Optional[str] = None, 
        current: Optional[Any] = None,
        initial: Optional[Any] = None,
        target: Optional[Any] = None,
        unit: Optional[str] = None,
        precision: Optional[int] = None,
        fraction: Optional[float] = None,
        time_elapsed: Optional[float] = None,
        time_remaining: Optional[float] = None,
    ):
        ...

class Status(Protocol):
    done: bool
    status: bool
    # I think finished_cb should be replaced with add_callback(Callable)
    finished_cb: Callable 
    def watch(self, StatusWatchCallable) -> None: ...

@runtime_checkable
class Readable(Protocol):
    name: str
    parent: Optional["Readable"]
    hints: Dict

    def trigger(self) -> Status: ...
    def read(self) -> Configuration: ...
    def describe(self) -> Configuration: ...
    def read_configuration(self) -> Configuration: ...
    def describe_configuration(self) -> Configuration: ...
    # def configure(self, *args, **kwargs) -> Tuple[Configuration, Configuration]: ...

@runtime_checkable
class Movable(Readable, Protocol):
    def set(self, *args, **kwargs) -> Status: ...

@runtime_checkable
class Flyable(Protocol):
    name: str
    parent: Optional[Any]

    def kickoff(self) -> Status: ...
    def complete(self) -> Status: ...
    def collect(self) -> Generator[Dict[str, Any], None, None]: ...
    def describe_collect(self) -> Dict[str, Configuration]: ...

    # Remaining same as Readable
    def read_configuration(self) -> Configuration: ...
    def describe_configuration(self) -> Configuration: ...
    # Not defined by ophyd flyer...
    # def configure(self, *args, **kwjargs) -> Tuple[Configuration, Configuration]: ...
