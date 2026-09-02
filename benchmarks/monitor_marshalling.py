#!/usr/bin/env python3
"""Measure the cost of marshalling monitor emission onto the event loop.

``PlanExecutor.emit`` normally hands a monitored signal's documents to
subscribers on whichever thread the signal called back on -- for a
synchronous ophyd v1 signal, that is the device's own thread, not the
RunEngine's event loop. ``PlanSession(marshal_monitor_emission=True)`` (or
the ``BLUESKY_MARSHAL_MONITOR_EMISSION`` environment variable) turns that
around: emission from a foreign thread is marshalled onto the loop with
``call_soon_threadsafe`` instead, exactly as status callbacks already are.

This script measures what that costs and what it buys, so the question can
be judged on numbers rather than argument alone. See bluesky#2050, which
raised the asymmetry and the tradeoffs, and the WIP PR built on top of it.

Self-contained; run with no arguments::

    python benchmarks/monitor_marshalling.py

Needs ophyd on the path (``pip install -e ".[dev]"`` gets it).
"""

from __future__ import annotations

import platform
import statistics
import sys
import threading
import time

from bluesky import Msg, RunEngine

try:
    import ophyd
except ImportError:  # pragma: no cover
    print("This benchmark needs ophyd installed: pip install -e '.[dev]'", file=sys.stderr)
    raise SystemExit(1) from None


def _new_RE(marshal: bool) -> RunEngine:
    """A RunEngine with the guard set directly.

    RunEngine.__init__ does not take marshal_monitor_emission -- it is a
    PlanSession parameter, and this PR does not touch RunEngine's signature
    (see the scope fences in the PR body). Setting the session's public
    attribute after construction has the same effect.
    """
    RE = RunEngine({})
    RE._session.marshal_monitor_emission = marshal
    return RE


def _percentile(data: list[float], pct: float) -> float:
    data = sorted(data)
    if len(data) == 1:
        return data[0]
    k = (len(data) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(data) - 1)
    if f == c:
        return data[f]
    return data[f] + (data[c] - data[f]) * (k - f)


def measure_latency(marshal: bool, n_samples: int = 300) -> tuple[float, float] | None:
    """Median and p95 latency, in ms, from sig.put() to the subscriber seeing
    the resulting Event document."""
    RE = _new_RE(marshal)
    sig = ophyd.Signal(name="sig", value=0)

    send_times: dict[int, float] = {}
    latencies_ms: list[float] = []
    done = threading.Event()

    def collector(name, doc):
        if name != "event":
            return
        value = doc["data"]["sig"]
        t_send = send_times.get(value)
        if t_send is not None:
            latencies_ms.append((time.perf_counter() - t_send) * 1000)
        if len(latencies_ms) >= n_samples:
            done.set()

    RE.subscribe(collector)

    monitoring = threading.Event()

    def putter():
        assert monitoring.wait(timeout=10)
        for i in range(1, n_samples + 1):
            send_times[i] = time.perf_counter()
            sig.put(i)
            time.sleep(0.002)  # ~500 Hz

    thread = threading.Thread(target=putter, daemon=True)
    thread.start()

    def plan():
        yield Msg("open_run")
        yield Msg("monitor", sig)
        monitoring.set()
        # Bounded by both the sample count and a wall-clock ceiling, so a
        # stuck run cannot hang the benchmark.
        for _ in range(3000):
            if done.is_set():
                break
            yield Msg("sleep", None, 0.01)
        yield Msg("unmonitor", sig)
        yield Msg("close_run")

    RE(plan())
    thread.join(timeout=5)

    if len(latencies_ms) < n_samples // 2:
        return None  # too few samples to be meaningful
    latencies_ms = latencies_ms[:n_samples]
    return statistics.median(latencies_ms), _percentile(latencies_ms, 95)


def measure_throughput(
    marshal: bool,
    duration_s: float = 1.0,
    monitor_hz: float = 500,
    slow_subscriber_s: float = 0.0,
) -> float:
    """Messages/sec the run loop sustains while a monitor fires in the
    background, optionally with a subscriber slow enough to matter."""
    RE = _new_RE(marshal)
    sig = ophyd.Signal(name="sig", value=0)

    if slow_subscriber_s:

        def collector(name, doc):
            if name == "event":
                time.sleep(slow_subscriber_s)

        RE.subscribe(collector)

    stop = threading.Event()
    counted = {"n": 0}

    def putter():
        period = 1.0 / monitor_hz
        while not stop.is_set():
            sig.put(1)
            time.sleep(period)

    def plan():
        yield Msg("open_run")
        yield Msg("monitor", sig)
        while not stop.is_set():
            counted["n"] += 1
            yield Msg("null")
        yield Msg("unmonitor", sig)
        yield Msg("close_run")

    thread = threading.Thread(target=putter, daemon=True)
    thread.start()
    timer = threading.Timer(duration_s, stop.set)
    timer.start()

    t0 = time.perf_counter()
    RE(plan())
    elapsed = time.perf_counter() - t0
    thread.join(timeout=10)

    return counted["n"] / elapsed


def main() -> None:
    print(f"Python {platform.python_version()} ({sys.implementation.name}) on {platform.platform()}")
    print(f"ophyd {ophyd.__version__}")
    print()

    rows: list[tuple[str, str, str]] = []

    print("Measuring monitor document latency (sig.put() -> subscriber sees the doc)...")
    for marshal, label in ((False, "off (today)"), (True, "on")):
        result = measure_latency(marshal)
        if result is None:
            rows.append((f"latency, {label}", "n/a", "too noisy to be meaningful"))
        else:
            median, p95 = result
            rows.append((f"latency, {label}", f"{median:.3f} ms median", f"{p95:.3f} ms p95"))

    print("Measuring plan throughput with a monitor firing in the background...")
    for marshal, label in ((False, "off (today)"), (True, "on")):
        hz = measure_throughput(marshal)
        rows.append((f"throughput, {label}", f"{hz:,.0f} msg/s", ""))

    print("Measuring plan throughput with a slow (10 ms) monitor subscriber...")
    print("(this is the number that decides the 'against' argument from #2050)")
    stall_rows = []
    for marshal, label in ((False, "off (today)"), (True, "on")):
        hz = measure_throughput(marshal, slow_subscriber_s=0.010, monitor_hz=120)
        stall_rows.append((f"stall case, {label}", f"{hz:,.0f} msg/s", ""))

    print()
    header = f"{'measurement':<28}{'value':<20}{'detail'}"
    print(header)
    print("-" * len(header))
    for name, value, detail in rows + stall_rows:
        print(f"{name:<28}{value:<20}{detail}")


if __name__ == "__main__":
    main()
