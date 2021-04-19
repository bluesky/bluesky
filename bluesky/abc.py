try:
    from typing import Protocol, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, runtime_checkable

from typing import Dict, Any, Optional, Callable, Generator, List, Union, Tuple


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
    def parent(self) -> Optional[Any]:
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


@runtime_checkable
class Configurable(Protocol):
    def configure(self, *args, **kwargs) -> Tuple[Configuration, Configuration]:
        ...


@runtime_checkable
class Movable(Readable, Protocol):
    def set(self, *args, **kwargs) -> Status:
        ...


@runtime_checkable
class Flyable(Protocol):
    name: str

    @property
    def parent(self) -> Optional[Any]:
        ...

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


@runtime_checkable
class Stageable(Protocol):
    def stage(self) -> List[Union[Readable, Flyable]]:
        ...

    def unstage(self) -> List[Union[Readable, Flyable]]:
        ...


@runtime_checkable
class Pausable(Protocol):
    def pause(self) -> None:
        ...

    def resume(self) -> None:
        ...


@runtime_checkable
class Subscribable(Protocol):
    def subscribe(self, function: Callable) -> None:
        ...

    def clear_sub(self, function: Callable) -> None:
        ...


@runtime_checkable
class Checkable(Protocol):
    def check_value(self, value: Any) -> None:
        ...
