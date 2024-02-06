import pytest
from bluesky.callbacks.ml import AgentCallback
import pandas as pd
from bluesky.plans import scan


class DummyAgent:
    """Simple agent reports a statistical summary of data"""

    def __init__(self, verbose=True):
        self.independents = []
        self.dependents = []
        self.last_summary = None
        self.verbose = verbose
        self.n_reports = 0

    def tell(self, x, y):
        self.independents.append(x)
        self.dependents.append(y)
        self.last_summary = pd.Series(self.dependents).describe()

    def ask(self):
        raise NotImplementedError

    def report(self):
        self.n_reports += 1
        if self.verbose:
            print(self.last_summary)
        return self.last_summary


def test_bad_agent():
    class BadAgent:
        def tells(self, x, y):
            pass

        def reporting(self):
            pass

    class WorseAgent:
        def __init__(self):
            self.tell = None
            self.report = 4

    with pytest.warns(UserWarning):
        # Wrong names for function calls
        AgentCallback(BadAgent(), independent_key="", dependent_key="")
    with pytest.warns(UserWarning):
        # Not callable attributes
        AgentCallback(WorseAgent(), independent_key="", dependent_key="")


def test_simple(RE, hw):
    cb = AgentCallback(DummyAgent(), independent_key="motor", dependent_key="det")
    RE.subscribe(cb)
    RE(scan([hw.det], hw.motor, 1, 3, num=3))


def test_multi_agent(RE, hw):
    cb = AgentCallback(
        DummyAgent(verbose=False), DummyAgent(verbose=False), independent_key="motor", dependent_key="det"
    )
    assert cb.agents[0] is not cb.agents[1]
    RE.subscribe(cb)
    RE(scan([hw.det], hw.motor, 1, 3, num=3))
    assert (cb.agents[0].last_summary == cb.agents[1].last_summary).all()


def test_closure(RE, hw):
    agent = DummyAgent(verbose=False)
    cb = AgentCallback(agent, independent_key="motor", dependent_key="det")
    RE.subscribe(cb)
    RE(scan([hw.det], hw.motor, 1, 3, num=3))
    assert agent.last_summary.astype(bool).all()


def test_report_on_event(RE, hw):
    agent = DummyAgent(verbose=False)
    cb = AgentCallback(agent, independent_key="motor", dependent_key="det", report_on_stop=False)
    RE.subscribe(cb)
    RE(scan([hw.det], hw.motor, 1, 3, num=3))
    assert agent.n_reports == 3


def test_report_on_stop(RE, hw):
    agent = DummyAgent(verbose=False)
    cb = AgentCallback(agent, independent_key="motor", dependent_key="det", report_on_event=False)
    RE.subscribe(cb)
    RE(scan([hw.det], hw.motor, 1, 3, num=3))
    assert agent.n_reports == 1


def test_no_reports(RE, hw):
    agent = DummyAgent(verbose=False)
    cb = AgentCallback(
        agent, independent_key="motor", dependent_key="det", report_on_event=False, report_on_stop=False
    )
    RE.subscribe(cb)
    RE(scan([hw.det], hw.motor, 1, 3, num=3))
    assert agent.n_reports == 0
    assert len(agent.dependents) > 0


def test_stream_name(RE, hw):
    agent = DummyAgent(verbose=True)
    cb = AgentCallback(agent, independent_key="motor", dependent_key="det", stream_name="background")
    RE.subscribe(cb)
    RE(scan([hw.det], hw.motor, 1, 3, num=3))
    assert agent.last_summary is None


def test_class_safety(RE, hw):
    class ErrorAgent:
        def tell(self, x, y):
            pass

        def report(self):
            raise RuntimeError

    RE.subscribe(AgentCallback(ErrorAgent(), independent_key="motor", dependent_key="det"))
    RE(scan([hw.det], hw.motor, 1, 3, num=3))
