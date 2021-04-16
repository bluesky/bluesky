try:
    from typing import Protocol, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, runtime_checkable

from typing import Dict, Any, TypeVar, Optional, Callable, Generator


Configuration = Dict[str, Dict[str, Any]]


@runtime_checkable
class Status(Protocol):
    done: bool
    success: bool

    def add_callback(self, callback: Callable[["Status"], None]) -> None:
        ...


@runtime_checkable
class Readable(Protocol):
    name: str

    @property
    def parent(self) -> Optional["Readable"]:
        ...

    @property
    def hints(self) -> Dict:
        ...

    def trigger(self) -> Status:
        ...

    def read(self) -> Configuration:
        ...

    def describe(self) -> Configuration:
        ...

    def read_configuration(self) -> Configuration:
        ...

    def describe_configuration(self) -> Configuration:
        ...

    # def configure(self, *args, **kwargs) -> Tuple[Configuration, Configuration]: ...


@runtime_checkable
class Movable(Readable, Protocol):
    def set(self, *args, **kwargs) -> Status:
        ...


@runtime_checkable
class Flyable(Protocol):
    name: str
    parent: Optional[Any]

    def kickoff(self) -> Status:
        ...

    def complete(self) -> Status:
        ...

    def collect(self) -> Generator[Dict[str, Any], None, None]:
        ...

    def describe_collect(self) -> Dict[str, Configuration]:
        ...

    # Remaining same as Readable
    def read_configuration(self) -> Configuration:
        ...

    def describe_configuration(self) -> Configuration:
        ...

    # Not defined by ophyd flyer...
    # def configure(self, *args, **kwjargs) -> Tuple[Configuration, Configuration]: ...
