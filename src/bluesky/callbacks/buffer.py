import atexit
import threading
from queue import Empty, Queue


class BufferingWrapper:
    """A wrapper for callbacks that processes documents in a separate thread.

    This class allows a callback to be executed in a background thread, processing
    documents as they are received. This prevent the blocking of RunEngine on any
    slow I/O operations by the callback. It handles graceful shutdown on exit or signal
    termination, ensuring that no new documents are accepted after shutdown has been
    initiated.

    The wrapped callback should be thread-safe and not subscribed to the RE directly.
    If it maintains shared mutable state, it must protect it using internal locking.

    This is mainly a development feature to allow subscribing (potentially many)
    experimental callbacks to a `RunEngine` without the risk of blocking the experiment.
    The use in production is currently not encouraged (at least not without a proper
    testing and risk assessment).

    Parameters
    ----------
        target : callable
            The instance of a callback that will be called with the documents.
            It should accept two parameters: `name` and `doc`.

    Usage
    -----
        tw = TiltedWriter(client)
        buff_tw = BufferingWrapper(tw)
        RE.subscribe(buff_tw)
    """

    def __init__(self, target):
        self._wrapped_callback = target
        self._queue = Queue()
        self._stop_event = threading.Event()
        self._shutdown_lock = threading.Lock()
        self._is_shutdown = False

        self._thread = threading.Thread(target=self._process_queue, daemon=True)
        self._thread.start()

        self._register_handlers()

    def __call__(self, name, doc):
        with self._shutdown_lock:
            if self._is_shutdown:
                raise RuntimeError("Cannot accept new data after shutdown.")
            self._queue.put((name, doc))

    def _process_queue(self):
        while True:
            try:
                if item := self._queue.get(timeout=1):
                    self._wrapped_callback(*item)  # Delegate to wrapped callback
                else:
                    break  # Received sentinel value to stop processing
            except Empty:
                if self._stop_event.is_set():
                    break

    def shutdown(self, wait=True):
        with self._shutdown_lock:
            if self._is_shutdown:
                return
            self._is_shutdown = True
            self._stop_event.set()
            self._queue.put(None)

            self._unregister_handlers()

        if wait:
            self._thread.join()
        print(f"{self._wrapped_callback.__class__.__name__} shut down gracefully.")

    def _signal_handler(self, signum, frame):
        print(f"Signal {signum} received. Shutting down {self._wrapped_callback.__class__.__name__}...")
        self.shutdown()
        raise SystemExit(0)

    def _register_handlers(self):
        if threading.current_thread() is threading.main_thread():
            try:
                atexit.register(self.shutdown)
                # signal.signal(signal.SIGINT, self._signal_handler)
                # signal.signal(signal.SIGTERM, self._signal_handler)
            except Exception as e:
                print(f"Failed to register signal handlers: {e}")

    def _unregister_handlers(self):
        try:
            atexit.unregister(self.shutdown)
        except Exception:
            pass

        # try:
        #     if signal.getsignal(signal.SIGINT) == self._signal_handler:
        #         signal.signal(signal.SIGINT, signal.SIG_DFL)
        #     if signal.getsignal(signal.SIGTERM) == self._signal_handler:
        #         signal.signal(signal.SIGTERM, signal.SIG_DFL)
        # except Exception:
        #     pass
