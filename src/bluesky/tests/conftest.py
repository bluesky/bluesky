import asyncio
import os
import signal
import threading
import time
from typing import Any
from unittest.mock import patch

import numpy as np
import packaging
import pytest

from bluesky.protocols import HasHints, HasParent, Hints, NamedMovable, Readable, Status
from bluesky.run_engine import RunEngine, TransitionError
from bluesky.utils import SigintHandler


def _clean_event_loop(RE, loop):
    """Stop a RunEngine's background loop thread and close the loop.

    Without this the ``_UnixSelectorEventLoop`` (and its AF_UNIX self-pipe
    socketpair) is only released at interpreter shutdown, producing noisy
    ``ResourceWarning: unclosed event loop`` / ``unclosed <socket.socket ...>``
    messages.
    """
    if RE.state not in ("idle", "panicked"):
        try:
            RE.halt()
        except TransitionError:
            pass
    loop.call_soon_threadsafe(loop.stop)
    RE._th.join()
    loop.close()


@pytest.fixture(scope="function")
def make_RE(request):
    """Factory for ``RunEngine`` instances whose event loops are closed on teardown.

    This underpins the ready-to-use ``RE`` / ``single_RE`` fixtures and is
    only needed directly in the rare case where a test wants to construct several
    engines or pass unusual constructor arguments.  Every engine created via the
    returned factory has its background loop thread stopped and its event loop
    closed during teardown.
    """

    def factory(*args, **kwargs):
        loop = asyncio.new_event_loop()
        loop.set_debug(True)
        RE = RunEngine(*args, loop=loop, **kwargs)
        request.addfinalizer(lambda: _clean_event_loop(RE, loop))
        return RE

    return factory


@pytest.fixture(scope="function", params=[False, True])
def RE(request, make_RE):
    """A ready-to-use ``RunEngine`` parametrized over ``call_returns_result``.

    Tests using this fixture run twice, once with ``call_returns_result=False``
    and once with ``True``.
    """
    return make_RE({}, call_returns_result=request.param)


@pytest.fixture(scope="function")
def single_RE(make_RE):
    """A ready-to-use ``RunEngine`` that runs a test only once.

    Like ``RE`` but without the ``call_returns_result`` parametrization, for
    tests where running under both values adds no coverage.  ``call_returns_result``
    is set to ``True`` so plan results are available.
    """
    return make_RE({}, call_returns_result=True)


@pytest.fixture(scope="function")
def hw(tmp_path):
    import ophyd
    from ophyd.sim import hw

    # ophyd 1.4.0 added support for customizing the directory used by simulated
    # hardware that generates files
    if packaging.version.Version(ophyd.__version__) >= packaging.version.Version("1.4.0"):
        return hw(str(tmp_path))
    else:
        return hw()


class AlwaysSuccessfulStatus(Status):
    def add_callback(self, callback) -> None:
        callback(self)

    def exception(self, timeout=0.0):
        return None

    @property
    def done(self) -> bool:
        return True

    @property
    def success(self) -> bool:
        return True


class ReadableSignal(Readable, HasHints, HasParent):
    def __init__(self, name: str) -> None:
        self._name = name
        self._value = 0.0

    @property
    def name(self) -> str:
        return self._name

    @property
    def hints(self) -> Hints:
        return {
            "fields": [self._name],
            "dimensions": [],
            "gridding": "rectilinear",
        }

    @property
    def parent(self) -> Any | None:
        return None

    def read(self):
        return {self._name: {"value": self._value, "timestamp": time.time()}}

    def describe(self):
        return {self._name: {"source": self._name, "dtype": "number", "shape": []}}


class MovableSignal(ReadableSignal, NamedMovable):
    def __init__(self, name: str, initial_value: float = 0.0) -> None:
        super().__init__(name)
        self._value: float = initial_value

    def set(self, value: float) -> Status:
        self._value = value
        return AlwaysSuccessfulStatus()


# vendored from ophyd.sim
class NumpySeqHandler:
    specs = {"NPY_SEQ"}

    def __init__(self, filename, root=""):
        self._name = os.path.join(root, filename)

    def __call__(self, index):
        return np.load(f"{self._name}_{index}.npy")

    def get_file_list(self, datum_kwarg_gen):
        "This method is optional. It is not needed for access, but for export."
        return ["{name}_{index}.npy".format(name=self._name, **kwargs) for kwargs in datum_kwarg_gen]


@pytest.fixture(scope="function")
def db(request):
    """Return a data broker"""
    try:
        from databroker import temp

        db = temp()
        return db
    except ImportError:
        pytest.skip("Databroker v2 still missing temp.")
    except ValueError:
        pytest.skip("Intake is failing for unknown reasons.")


@pytest.fixture(autouse=True)
def cleanup_any_figures(request):
    import matplotlib.pyplot as plt

    "Close any matplotlib figures that were opened during a test."
    plt.close("all")


class DeterministicSigint:
    """Sends SIGINT signals with a fake monotonic clock so that every signal
    deterministically clears the 100ms debounce in SigintHandler.

    The fake clock advances by 0.2s per ``send()`` call, and each call blocks
    until the signal handler has finished, so ``_count`` increments reliably
    regardless of real wall-clock jitter.
    """

    def __init__(self):
        self._fake_time = 0.0
        self._handler_done = threading.Event()
        self._pid = os.getpid()
        self._orig_enter = SigintHandler.__enter__
        self._patcher = patch.object(SigintHandler, "__enter__", self._patched_enter)

    def _monotonic(self):
        return self._fake_time

    def _patched_enter(self, sigint_handler):
        with patch("bluesky.utils.time.monotonic", self._monotonic):
            result = self._orig_enter(sigint_handler)
        installed = signal.getsignal(signal.SIGINT)

        def synced_handler(signum, frame):
            try:
                with patch("bluesky.utils.time.monotonic", self._monotonic):
                    installed(signum, frame)
            finally:
                self._handler_done.set()

        signal.signal(signal.SIGINT, synced_handler)
        return result

    def send(self):
        """Send one SIGINT and wait for the handler to finish."""
        self._handler_done.clear()
        self._fake_time += 0.2
        os.kill(self._pid, signal.SIGINT)
        self._handler_done.wait()

    def __enter__(self):
        self._patcher.start()
        return self

    def __exit__(self, *exc):
        self._patcher.stop()


@pytest.fixture
def deterministic_sigint():
    """Fixture providing the ``DeterministicSigint`` class.  Tests should use
    it as a context manager around the code that runs the RE::

        with deterministic_sigint() as sigint:
            ...
            sigint.send()
    """
    return DeterministicSigint
