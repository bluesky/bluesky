import atexit
import logging
import threading
import time
from collections.abc import Callable
from queue import Empty, Full, Queue

logger = logging.getLogger(__name__)


class BufferingWrapper:
    """A wrapper for callbacks that processes documents in a separate thread.

    This class allows a callback to be executed in a background thread, processing
    documents as they are received. This prevent the blocking of RunEngine on any
    slow I/O operations by the callback. It handles graceful shutdown on exit or signal
    termination, ensuring that no new documents are accepted after shutdown has been
    initiated.

    The wrapped callback should be thread-safe and not subscribed to the RE directly.
    If it maintains shared mutable state, it must protect it using internal locking.

    Because documents are processed asynchronously, a run may not be fully written
    by the time `RE(plan)` returns. Code that reads a run back immediately after
    it completes (e.g. `db[-1]` or `db[uid]`) must call :meth:`flush` first to
    wait for the buffer to drain; otherwise it may observe a missing or incomplete
    run, or silently pick up the previous one.

    This is mainly a development feature to allow subscribing (potentially many)
    experimental callbacks to a `RunEngine` without the risk of blocking the experiment.
    The use in production is currently not encouraged (at least not without a proper
    testing and risk assessment).

    Parameters
    ----------
        target : callable
            The instance of a callback that will be called with the documents.
            It should accept two parameters: `name` and `doc`.
        queue_size : int, optional
            The maximum size of the internal queue. Default is 1,000,000.

    Usage
    -----
        tw = TiledWriter(client)
        buff_tw = BufferingWrapper(tw)
        RE.subscribe(buff_tw)
    """

    def __init__(self, target: Callable, queue_size: int = 1_000_000):
        self._wrapped_callback = target
        self._queue: Queue = Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._shutdown_lock = threading.Lock()

        self._thread = threading.Thread(target=self._process_queue, daemon=True)
        self._thread.start()

        atexit.register(self.shutdown)

    def __call__(self, name, doc):
        if self._stop_event.is_set():
            raise RuntimeError("Cannot accept new data after shutdown.")
            # TODO: This can be refactored using the upstream functionality (in Python >= 3.13)
            # https://docs.python.org/3/library/queue.html#queue.Queue.shutdown
        try:
            self._queue.put((name, doc))
        except Full as e:
            logger.exception(
                f"The buffer is full. The {self._wrapped_callback.__class__.__name__} can not keep up with the incoming data. "  # noqa
                f"Consider increasing the queue size or optimizing the callback processing: {e}"
            )
            raise RuntimeError(
                f"The buffer is full. The {self._wrapped_callback.__class__.__name__} can not keep up with the incoming data. "  # noqa
                "Consider increasing the queue size or optimizing the callback processing."
            ) from e
        except Exception as e:
            logger.exception(f"Failed to put document {name} in queue: {e}")
            raise RuntimeError(f"Failed to put document {name} in queue: {e}") from e

    def _process_queue(self):
        while True:
            try:
                item = self._queue.get(timeout=1)
            except Empty:
                if self._stop_event.is_set():
                    break
                continue
            try:
                if item is None:
                    break  # Received sentinel value to stop processing
                self._wrapped_callback(*item)  # Delegate to wrapped callback
            except Exception as e:
                logger.exception(f"Exception in {self._wrapped_callback.__class__.__name__}: {e}")
            finally:
                # Account for every retrieved item (including the sentinel and items whose
                # processing raised), so that `flush` can rely on the queue's unfinished-task counter.
                self._queue.task_done()

    def flush(self, timeout: "float | None" = None) -> bool:
        """Block until all documents enqueued so far have been processed.

        This provides a synchronization barrier for read-after-write patterns:
        a plan or script that reads a run back (e.g. `db[uid]`) immediately
        after `RE(plan)` returns must call `flush()` first, because the
        wrapped callback processes documents asynchronously and may not have
        written the run yet by the time the plan finishes.

        Documents enqueued by another thread *while* flushing extend the wait:
        the method returns once the queue has no unfinished items.

        Parameters
        ----------
        timeout : float, optional
            Maximum time to wait, in seconds. By default, wait indefinitely.

        Returns
        -------
        bool
            True if the buffer was fully drained; False if the timeout expired
            (or the worker thread is no longer running) with items remaining.
        """
        if timeout is not None:
            deadline = time.monotonic() + timeout
        with self._queue.all_tasks_done:
            while self._queue.unfinished_tasks:
                if not self._thread.is_alive():
                    # The worker has exited (e.g. after shutdown); any items
                    # still in the queue will never be processed.
                    return not self._queue.unfinished_tasks
                if timeout is None:
                    # Wake up periodically to recheck worker liveness.
                    self._queue.all_tasks_done.wait(timeout=1)
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._queue.all_tasks_done.wait(timeout=min(remaining, 1))
        return True

    def shutdown(self, wait: bool = True):
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        self._queue.put(None)

        atexit.unregister(self.shutdown)

        if wait:
            self._thread.join()

        logger.info(f"{self._wrapped_callback.__class__.__name__} shut down gracefully.")
