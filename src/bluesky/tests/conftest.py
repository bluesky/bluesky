import asyncio
import contextlib
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

CALL_RETURNS_RESULT_OPTION = "--include-call-returns-result-false"


def pytest_addoption(parser):
    parser.addoption(
        CALL_RETURNS_RESULT_OPTION,
        action="store_true",
        default=False,
        help=(
            "Also run the RE fixture with call_returns_result=False. "
            "By default only call_returns_result=True is exercised."
        ),
    )


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


def pytest_generate_tests(metafunc):
    """Parametrize the ``RE`` fixture over ``call_returns_result``.

    A fixture cannot both declare ``params`` and be re-parametrized by a hook,
    so the parametrization lives here (the only place that can see the
    command-line option): ``call_returns_result=True`` always runs, and the
    ``False`` variant is added only when ``--include-call-returns-result-false``
    is passed, doubling the ``RE``-based tests.
    """
    if RE.__name__ in metafunc.fixturenames:
        call_returns_result = [True]
        if metafunc.config.getoption(CALL_RETURNS_RESULT_OPTION):
            call_returns_result = [False, True]
        metafunc.parametrize(RE.__name__, call_returns_result, indirect=True)


@pytest.fixture(scope="function")
def RE(request, make_RE):
    """A ready-to-use ``RunEngine`` parametrized over ``call_returns_result``.

    Parametrization is supplied by :func:`pytest_generate_tests`: by default the
    fixture only runs with ``call_returns_result=True``. Pass
    ``--include-call-returns-result-false`` to also run the ``False`` variant.
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

    SIGINT belongs to this object for the whole ``with`` block.  While the
    RunEngine holds it, signals go to the RunEngine's handler and are counted
    in ``delivered``; once the RunEngine gives it back, they are counted in
    ``absorbed`` and discarded, the way an interactive prompt would swallow a
    Ctrl+C typed after a plan has already stopped.  Sender threads are started
    through ``send_after`` and joined before the real disposition is restored,
    so no signal can outlive the block and reach pytest.
    """

    def __init__(self, fake_clock=True):
        self._fake_clock = fake_clock
        self._fake_time = 0.0
        self._handler_done = threading.Event()
        self._senders: list[threading.Thread] = []
        self.delivered = 0
        self.absorbed = 0
        self._orig_enter = SigintHandler.__enter__
        self._orig_exit = SigintHandler.__exit__
        self._enter_patcher = patch.object(SigintHandler, "__enter__", self._patched_enter)
        self._exit_patcher = patch.object(SigintHandler, "__exit__", self._patched_exit)

    def _monotonic(self):
        return self._fake_time

    def _clock(self):
        """Patch the handler's clock, or leave the real one alone."""
        if self._fake_clock:
            return patch("bluesky.utils.time.monotonic", self._monotonic)
        return contextlib.nullcontext()

    def _patched_enter(self, sigint_handler):
        with self._clock():
            result = self._orig_enter(sigint_handler)
        installed = signal.getsignal(signal.SIGINT)

        def synced_handler(signum, frame):
            try:
                with self._clock():
                    installed(signum, frame)
            finally:
                self.delivered += 1
                self._handler_done.set()

        signal.signal(signal.SIGINT, synced_handler)
        return result

    def _patched_exit(self, sigint_handler, exc_type, exc, tb):
        result = self._orig_exit(sigint_handler, exc_type, exc, tb)
        signal.signal(signal.SIGINT, self._absorb)
        return result

    def _absorb(self, signum, frame):
        """Record a signal that arrived after the RunEngine released SIGINT.

        Only ever installed by ``_patched_exit``: installing it earlier would
        make it the disposition ``SigintHandler`` captures as
        ``_original_handler``, and the escape hatch would have nothing to
        raise ``KeyboardInterrupt`` into.
        """
        self.absorbed += 1
        self._handler_done.set()

    def send(self):
        """Send one SIGINT to the main thread and wait for the handler to finish."""
        self._handler_done.clear()
        self._fake_time += 0.2
        # Sent to the main thread rather than to the process: if a
        # process-directed signal arrives just as the main thread is entering
        # the untimed wait in DuringTask.block, the C-level handler sets the
        # flag but nothing interrupts the wait, and the Python-level handler
        # does not run until the plan ends for some other reason.
        signal.pthread_kill(threading.main_thread().ident, signal.SIGINT)
        if not self._handler_done.wait(timeout=10):
            raise RuntimeError("SIGINT was never handled")

    def send_after(self, event, count, timeout=5):
        """Send ``count`` SIGINTs from a background thread once ``event`` is set."""

        def sim_kill():
            event.wait(timeout=timeout)
            for _ in range(count):
                self.send()

        return self.background(sim_kill)

    def background(self, func):
        """Run ``func`` in a thread joined before SIGINT is handed back.

        For senders that interleave waits between hits, where ``send_after``
        does not fit.
        """
        thread = threading.Thread(target=func, daemon=True)
        self._senders.append(thread)
        thread.start()
        return thread

    def __enter__(self):
        self._true_original = signal.getsignal(signal.SIGINT)
        self._enter_patcher.start()
        self._exit_patcher.start()
        return self

    def __exit__(self, *exc):
        try:
            for thread in self._senders:
                thread.join(timeout=30)
        finally:
            self._exit_patcher.stop()
            self._enter_patcher.stop()
            signal.signal(signal.SIGINT, self._true_original)


@pytest.fixture
def sigint():
    """An entered ``DeterministicSigint`` on the real clock.

    For the tests that exercise the signal handler's own timing -- its 100 ms
    debounce and 10 s carry-over -- where a faked clock would test nothing.
    They still get the SIGINT ownership: hits arriving after the RunEngine has
    let go are absorbed rather than reaching pytest, and sender threads are
    joined before the real disposition is restored.  Entered for the whole
    test, so that ownership survives a failing assertion.
    """
    with DeterministicSigint(fake_clock=False) as sigint:
        yield sigint


@pytest.fixture
def blocking_motor():
    """A Movable whose sets stay in flight until ``status`` is finished.

    Every set returns the same ``Status``, so finishing it once releases the
    set the plan is waiting on and lets any later cleanup set complete.  The
    timer finishes it regardless, so that a test whose signals go astray fails
    rather than blocking forever.
    """
    from ophyd import StatusBase

    class BlockingMovable:
        def __init__(self):
            self.set_values = []
            self.set_called = threading.Event()
            self.status = StatusBase()

        def set(self, value):
            self.set_values.append(value)
            self.set_called.set()
            return self.status

    motor = BlockingMovable()
    unblock = threading.Timer(10, motor.status.set_finished)
    unblock.start()
    yield motor
    unblock.cancel()


@pytest.fixture
def deterministic_sigint():
    """Fixture providing the ``DeterministicSigint`` class.  Tests should use
    it as a context manager around the code that runs the RE::

        with deterministic_sigint() as sigint:
            ...
            sigint.send()
    """
    return DeterministicSigint
