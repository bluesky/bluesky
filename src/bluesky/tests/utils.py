from collections import defaultdict
import contextlib
import tempfile
import sys
import threading
import asyncio


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
        elif name == 'bulk_events':
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
