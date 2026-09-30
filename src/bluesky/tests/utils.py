import asyncio
import contextlib
import sys
import tempfile
import threading
from collections import defaultdict


@contextlib.contextmanager
def _print_redirect():
    old_stdout = sys.stdout
    try:
        fout = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
        sys.stdout = fout
        yield fout
    finally:
        sys.stdout = old_stdout


class MsgCollector:
    def __init__(self, msg_hook=None):
        self.msgs = []
        self.msg_hook = msg_hook

    def __call__(self, msg):
        self.msgs.append(msg)
        if self.msg_hook:
            self.msg_hook(msg)


class DocCollector:
    def __init__(self):
        self.start = []
        self.stop = {}
        self.descriptor = defaultdict(list)
        self.event = {}

    def insert(self, name, doc):
        if name == "start":
            self.start.append(doc)
        elif name == "stop":
            self.stop[doc["run_start"]] = doc
        elif name == "descriptor":
            self.descriptor[doc["run_start"]].append(doc)
            self.event[doc["uid"]] = []
        elif name == "bulk_events":
            for k, v in doc.items():
                self.event[k].extend(v)
        else:
            self.event[doc["descriptor"]].append(doc)


def _fabricate_asycio_event(loop):
    th_ev = threading.Event()

    aio_event = None

    def really_make_the_event():
        nonlocal aio_event
        aio_event = asyncio.Event()
        th_ev.set()

    h = loop.call_soon_threadsafe(really_make_the_event)
    if not th_ev.wait(0.1):
        h.cancel()
        raise Exception("failed to make asyncio event")
    return aio_event


def _careful_event_set(ev):
    "Helper to set 'do not lock test suite' backup sets"

    def inner():
        try:
            ev.set()
        except RuntimeError:
            ...

    return inner


def _when_parked(loop, func, *args, max_turns=100):
    """Call ``func(*args)`` on ``loop`` once nothing else there is ready to run, so a plan has parked."""
    turns = 0

    def check():
        nonlocal turns
        # Reads asyncio's private ready queue, as ophyd-async's wait_for_pending_wakeups does.
        if not loop._ready:
            func(*args)
        elif turns < max_turns:
            turns += 1
            loop.call_soon(check)
        else:
            raise RuntimeError(f"Tasks still scheduling wakeups after {max_turns} turns")

    loop.call_soon_threadsafe(check)


async def _parked(max_yields=100):
    """Return once nothing else on the running loop is ready to run, so every plan has parked."""
    loop = asyncio.get_running_loop()
    for _ in range(max_yields):
        await asyncio.sleep(0)
        # Reads asyncio's private ready queue, as ophyd-async's wait_for_pending_wakeups does.
        if not loop._ready:
            return
    raise RuntimeError(f"Tasks still scheduling wakeups after {max_yields} yields")


class CallbackSignal:
    """A `bluesky.protocols.Subscribable` signal that calls back synchronously on `put`.

    Subscribing reports the current value before returning, as ophyd and
    ophyd-async do.
    """

    def __init__(self, value=0, name="callback_signal"):
        self.name = name
        self._value = value
        self._callbacks: list = []

    def subscribe_reading(self, function) -> None:
        self._callbacks.append(function)
        function(self.read())

    def clear_sub(self, function) -> None:
        self._callbacks.remove(function)

    def read(self) -> dict:
        return {self.name: {"value": self._value, "timestamp": 0.0}}

    def put(self, value) -> None:
        """Set the value and report it, on the calling thread."""
        self._value = value
        for function in list(self._callbacks):
            function(self.read())
